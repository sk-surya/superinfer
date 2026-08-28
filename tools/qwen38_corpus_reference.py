#!/usr/bin/env python3
"""Stream a Qwen3.8 reference layer once across several corpus cases.

The model remains an independent Transformers oracle.  Cases retain separate hidden rows and
DynamicCache instances; only the checkpoint layer load/dequantization is shared.  This changes
the order of independent case evaluation, not the order of layers or token positions within a
case.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def canonical_token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_cases(corpus: dict[str, Any], selected: Sequence[str] | None = None) -> list[dict[str, Any]]:
    cases = corpus.get("cases")
    if not isinstance(cases, list):
        raise ValueError("corpus cases must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("every corpus case needs a string id")
        case_id = case["id"]
        if case_id in by_id:
            raise ValueError(f"duplicate corpus case: {case_id}")
        by_id[case_id] = case
    if selected is None:
        return list(by_id.values())
    unknown = [case_id for case_id in selected if case_id not in by_id]
    if unknown:
        raise ValueError(f"unknown corpus case(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in selected]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_cases(model_dir: Path, cases: Sequence[dict[str, Any]], output_dir: Path,
               round_activations: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    reference = __import__("tools.qwen38_nvfp4_e2e_reference", fromlist=["_nvfp4", "_rms_norm", "_tensor"])
    root = json.loads((model_dir / "config.json").read_text())
    config = Qwen3_5TextConfig.from_dict(root["text_config"])
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    torch.set_num_threads(max(1, min(32, torch.get_num_threads())))

    normalized_cases: list[tuple[str, list[int]]] = []
    for case in cases:
        case_id = str(case["id"])
        token_ids = case.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"case {case_id} has no token_ids")
        if not all(isinstance(token, int) and token >= 0 for token in token_ids):
            raise ValueError(f"case {case_id} has invalid token_ids")
        normalized_cases.append((case_id, token_ids))

    # Load embeddings once. The expensive checkpoint work that dominates the old oracle is the
    # repeated layer construction/dequantization; that work is shared below, while each case
    # still owns an independent cache and hidden-state list.
    with safe_open(str(model_dir / index["model.language_model.embed_tokens.weight"]),
                   framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("model.language_model.embed_tokens.weight").to(torch.float32)
    states = [
        {
            "id": case_id,
            "tokens": token_ids,
            "hidden": [embedding[token].reshape(1, 1, -1) for token in token_ids],
            "cache": DynamicCache(config=config),
        }
        for case_id, token_ids in normalized_cases
    ]
    del embedding
    rotary = Qwen3_5TextRotaryEmbedding(config)
    output_rows: dict[str, list[bytes]] = {case_id: [] for case_id, _ in normalized_cases}
    greedy: dict[str, list[int]] = {case_id: [] for case_id, _ in normalized_cases}

    with torch.inference_mode():
        for layer_index in range(config.num_hidden_layers):
            layer = Qwen3_5DecoderLayer(config, layer_index).eval()
            prefix = f"model.language_model.layers.{layer_index}."
            state_dict: dict[str, torch.Tensor] = {}
            for key in layer.state_dict():
                source_name = prefix + key
                if source_name not in index:
                    raise KeyError(source_name)
                if source_name.endswith(".weight") and source_name + "_scale" in index:
                    state_dict[key] = reference._nvfp4(model_dir, index, source_name)
                else:
                    with safe_open(str(model_dir / index[source_name]), framework="pt", device="cpu") as handle:
                        state_dict[key] = handle.get_tensor(source_name).to(torch.float32)
            layer.load_state_dict(state_dict, strict=True)
            del state_dict

            for state in states:
                next_hidden: list[torch.Tensor] = []
                for position, hidden in enumerate(state["hidden"]):
                    position_ids = torch.full((1, 1), position, dtype=torch.long)
                    position_embeddings = rotary(hidden, position_ids)
                    causal_mask = torch.zeros((1, 1, 1, position + 1), dtype=torch.float32)
                    output = layer(hidden, position_embeddings=position_embeddings,
                                   attention_mask=causal_mask, position_ids=position_ids,
                                   past_key_values=state["cache"])
                    updated = output[0] if isinstance(output, tuple) else output
                    if round_activations:
                        updated = updated.to(torch.bfloat16).to(torch.float32)
                    next_hidden.append(updated)
                state["hidden"] = next_hidden
            del layer
            gc.collect()

        final_norm = reference._tensor(model_dir, index, "model.language_model.norm.weight").to(torch.float32)
        lm_head = reference._nvfp4(model_dir, index, "lm_head.weight")
        for state in states:
            for hidden in state["hidden"]:
                normalized = reference._rms_norm(hidden, final_norm, config.rms_norm_eps)
                logits = F.linear(normalized.reshape(1, -1), lm_head).reshape(-1).contiguous()
                values = logits.cpu().numpy().astype("float32").tobytes()
                output_rows[state["id"]].append(values)
                greedy[state["id"]].append(int(torch.argmax(logits).item()))

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case_id, token_ids in normalized_cases:
        payload = b"".join(output_rows[case_id])
        output = output_dir / f"{case_id}.f32"
        output.write_bytes(payload)
        diagnostics = {
            "model": "Qwen3.8-27B-NVFP4-RTX5090",
            "reference": "Transformers Qwen3_5DecoderLayer, layer-streamed across corpus cases",
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "tokens": token_ids,
            "layers": config.num_hidden_layers,
            "steps": len(token_ids),
            "logits_per_step": 248320,
            "greedy_sequence": greedy[case_id],
            "output_sha256": _sha256(payload),
            "token_ids_sha256": canonical_token_hash(token_ids),
            "round_activations": round_activations,
            "evaluation_order": "one checkpoint layer across all selected cases, then next layer",
        }
        diagnostics_path = output.with_suffix(".json")
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        results.append({"id": case_id, "output_name": output.name, **diagnostics})
    return {
        "schema": "superinfer.qwen38.reference.corpus.v1",
        "status": "pass",
        "model": model_dir.name,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--round-activations", action="store_true")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    report = _run_cases(args.model_dir, select_cases(corpus, args.case_ids), args.output_dir,
                        args.round_activations)
    (args.output_dir / "corpus-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [case["id"] for case in report["cases"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
