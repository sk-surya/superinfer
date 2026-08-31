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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu",
                        help="Transformers oracle device, for example cpu or cuda")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument(
        "--round-activations", action="store_true",
        help="round each layer output through BF16 before the next layer",
    )
    parser.add_argument(
        "--round-kv", action="store_true",
        help="store every DynamicCache K/V update as BF16, then read it as FP32",
    )
    parser.add_argument(
        "--round-linear-state", action="store_true",
        help="store linear-attention convolution state as BF16, while retaining FP32 recurrent state",
    )
    parser.add_argument("--hidden-output", type=Path,
                        help="optional FP32 output for one final-normalized hidden row")
    parser.add_argument("--hidden-case",
                        help="corpus case whose normalized hidden row should be captured")
    parser.add_argument("--hidden-step", type=int,
                        help="zero-based token position to capture from --hidden-case")
    parser.add_argument("--boundaries-output", type=Path,
                        help="optional FP32 output for post-layer hidden rows")
    parser.add_argument("--boundary-case",
                        help="corpus case whose post-layer rows should be captured")
    parser.add_argument("--boundary-step", type=int,
                        help="zero-based token position whose post-layer rows should be captured")
    parser.add_argument("--attention-boundaries-output", type=Path,
                        help="optional FP32 output for post-token-mixer residual rows")
    parser.add_argument("--attention-boundary-case",
                        help="corpus case whose post-token-mixer rows should be captured")
    parser.add_argument("--attention-boundary-step", type=int,
                        help="zero-based token position whose post-token-mixer rows should be captured")
    parser.add_argument("--kv-output", type=Path,
                        help="optional FP32 output for one deployment KV cache snapshot")
    parser.add_argument("--kv-case",
                        help="corpus case whose KV cache should be captured")
    parser.add_argument("--kv-step", type=int,
                        help="zero-based token position at which to capture the KV cache")
    parser.add_argument("--kv-layer", type=int,
                        help="decoder layer whose KV cache should be captured")
    return parser


def _run_cases(model_dir: Path, cases: Sequence[dict[str, Any]], output_dir: Path,
               round_activations: bool, round_kv: bool, round_linear_state: bool,
               hidden_output: Path | None = None, hidden_case: str | None = None,
               hidden_step: int | None = None, boundaries_output: Path | None = None,
               boundary_case: str | None = None, boundary_step: int | None = None,
               attention_boundaries_output: Path | None = None,
               attention_boundary_case: str | None = None,
               attention_boundary_step: int | None = None,
               kv_output: Path | None = None, kv_case: str | None = None,
               kv_step: int | None = None, kv_layer: int | None = None,
               device_name: str = "cpu") -> dict[str, Any]:
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
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA reference device but CUDA is unavailable")

    class DeploymentStorageDynamicCache(DynamicCache):
        """DynamicCache with explicit deployment storage contracts for diagnostic qualification."""

        def __init__(self, *args, round_kv: bool, round_linear_state: bool, **kwargs):
            super().__init__(*args, **kwargs)
            self._round_kv = round_kv
            self._round_linear_state = round_linear_state

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            if self._round_kv:
                key_states = key_states.to(torch.bfloat16).to(torch.float32)
                value_states = value_states.to(torch.bfloat16).to(torch.float32)
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        def update_conv_state(self, conv_states, layer_idx, *args, **kwargs):
            if self._round_linear_state:
                conv_states = conv_states.to(torch.bfloat16).to(torch.float32)
            return super().update_conv_state(conv_states, layer_idx, *args, **kwargs)

    def round_linear_cache_state(cache: DynamicCache, layer_idx: int) -> None:
        """Model BF16 causal-convolution state after both prefill and single-token decode."""
        if not round_linear_state:
            return
        layer_cache = cache.layers[layer_idx]
        conv_states = getattr(layer_cache, "conv_states", None)
        if conv_states is not None:
            layer_cache.conv_states = conv_states.to(torch.bfloat16).to(torch.float32)

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
        embedding = handle.get_tensor("model.language_model.embed_tokens.weight").to(
            device=device, dtype=torch.float32)
    states = [
        {
            "id": case_id,
            "tokens": token_ids,
            "hidden": [embedding[token].reshape(1, 1, -1) for token in token_ids],
            "cache": (DeploymentStorageDynamicCache(
                config=config, round_kv=round_kv, round_linear_state=round_linear_state
            ) if round_kv or round_linear_state else DynamicCache(config=config)),
        }
        for case_id, token_ids in normalized_cases
    ]
    del embedding
    rotary = Qwen3_5TextRotaryEmbedding(config)
    output_rows: dict[str, list[bytes]] = {case_id: [] for case_id, _ in normalized_cases}
    greedy: dict[str, list[int]] = {case_id: [] for case_id, _ in normalized_cases}
    boundary_payload = bytearray()
    boundary_count = 0
    attention_boundary_payload = bytearray()
    attention_boundary_count = 0
    kv_payload = bytearray()
    kv_metadata: dict[str, Any] | None = None

    with torch.inference_mode():
        for layer_index in range(config.num_hidden_layers):
            layer = Qwen3_5DecoderLayer(config, layer_index).eval().to(device)
            prefix = f"model.language_model.layers.{layer_index}."
            state_dict: dict[str, torch.Tensor] = {}
            for key in layer.state_dict():
                source_name = prefix + key
                if source_name not in index:
                    raise KeyError(source_name)
                if source_name.endswith(".weight") and source_name + "_scale" in index:
                    state_dict[key] = reference._nvfp4(model_dir, index, source_name).to(device)
                else:
                    with safe_open(str(model_dir / index[source_name]), framework="pt", device="cpu") as handle:
                        state_dict[key] = handle.get_tensor(source_name).to(
                            device=device, dtype=torch.float32)
            layer.load_state_dict(state_dict, strict=True)
            del state_dict

            mixer_outputs: list[torch.Tensor] = []
            mixer_hook = None
            if attention_boundaries_output is not None:
                mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
                if mixer is None:
                    raise ValueError(f"layer {layer_index} has no token mixer")

                def capture_mixer(_module: Any, _inputs: Any, output: Any) -> None:
                    mixer_outputs.append(output[0] if isinstance(output, tuple) else output)

                mixer_hook = mixer.register_forward_hook(capture_mixer)
            for state in states:
                next_hidden: list[torch.Tensor] = []
                for position, hidden in enumerate(state["hidden"]):
                    position_ids = torch.full((1, 1), position, dtype=torch.long, device=device)
                    position_embeddings = rotary(hidden, position_ids)
                    causal_mask = torch.zeros((1, 1, 1, position + 1), dtype=torch.float32,
                                              device=device)
                    output = layer(hidden, position_embeddings=position_embeddings,
                                   attention_mask=causal_mask, position_ids=position_ids,
                                   past_key_values=state["cache"])
                    updated = output[0] if isinstance(output, tuple) else output
                    round_linear_cache_state(state["cache"], layer_index)
                    if (attention_boundaries_output is not None and state["id"] == attention_boundary_case and
                            (attention_boundary_step is None or position == attention_boundary_step)):
                        if len(mixer_outputs) != 1:
                            raise ValueError("token mixer hook did not capture exactly one output")
                        row = (hidden + mixer_outputs.pop()).reshape(-1).contiguous().cpu().numpy()
                        attention_boundary_payload.extend(row.astype("float32").tobytes())
                        attention_boundary_count += 1
                    elif attention_boundaries_output is not None:
                        mixer_outputs.clear()
                    if round_activations:
                        updated = updated.to(torch.bfloat16).to(torch.float32)
                    if (boundaries_output is not None and state["id"] == boundary_case and
                            (boundary_step is None or position == boundary_step)):
                        row = updated.reshape(-1).contiguous().cpu().numpy().astype("float32")
                        boundary_payload.extend(row.tobytes())
                        boundary_count += 1
                    next_hidden.append(updated)
                    if (kv_output is not None and state["id"] == kv_case and
                            layer_index == kv_layer and position == kv_step):
                        layer_cache = state["cache"].layers[layer_index]
                        keys = getattr(layer_cache, "keys", None)
                        values = getattr(layer_cache, "values", None)
                        if keys is None or values is None:
                            raise ValueError(f"layer {layer_index} has no KV cache at step {position}")
                        valid_keys = keys[0, :, :position + 1, :].permute(1, 0, 2).contiguous()
                        valid_values = values[0, :, :position + 1, :].permute(1, 0, 2).contiguous()
                        kv_payload.extend(valid_keys.float().cpu().numpy().tobytes())
                        kv_payload.extend(valid_values.float().cpu().numpy().tobytes())
                        kv_metadata = {
                            "case": state["id"], "layer": layer_index, "step": position,
                            "positions": position + 1,
                            "heads": int(valid_keys.shape[1]),
                            "head_dimension": int(valid_keys.shape[2]),
                        }
                state["hidden"] = next_hidden
            if mixer_hook is not None:
                mixer_hook.remove()
            del layer
            gc.collect()

        final_norm = reference._tensor(model_dir, index, "model.language_model.norm.weight").to(
            device=device, dtype=torch.float32)
        lm_head = reference._nvfp4(model_dir, index, "lm_head.weight").to(device)
        hidden_payload = bytearray()
        hidden_capture_metadata: dict[str, Any] | None = None
        for state in states:
            for position, hidden in enumerate(state["hidden"]):
                normalized = reference._rms_norm(hidden, final_norm, config.rms_norm_eps)
                if (hidden_output is not None and state["id"] == hidden_case and
                        (hidden_step is None or position == hidden_step)):
                    row = normalized.reshape(-1).contiguous().cpu().numpy().astype("float32")
                    hidden_payload.extend(row.tobytes())
                    hidden_capture_metadata = {
                        "case": state["id"],
                        "step": position,
                        "elements": int(row.size),
                    }
                logits = F.linear(normalized.reshape(1, -1), lm_head).reshape(-1).contiguous()
                values = logits.cpu().numpy().astype("float32").tobytes()
                output_rows[state["id"]].append(values)
                greedy[state["id"]].append(int(torch.argmax(logits).item()))

    if hidden_output is not None:
        if not hidden_payload or hidden_capture_metadata is None:
            raise ValueError("requested hidden capture did not match a corpus case and step")
        hidden_output.parent.mkdir(parents=True, exist_ok=True)
        hidden_output.write_bytes(hidden_payload)
        hidden_capture_metadata["sha256"] = _sha256(bytes(hidden_payload))
        hidden_capture_metadata["dtype"] = "float32"
        hidden_output.with_suffix(".json").write_text(
            json.dumps(hidden_capture_metadata, indent=2) + "\n"
        )
    if boundaries_output is not None:
        if not boundary_payload or boundary_count != config.num_hidden_layers:
            raise ValueError("requested boundary capture did not match one complete model step")
        boundaries_output.parent.mkdir(parents=True, exist_ok=True)
        boundaries_output.write_bytes(boundary_payload)
        boundaries_output.with_suffix(".json").write_text(json.dumps({
            "case": boundary_case,
            "step": boundary_step,
            "layers": boundary_count,
            "elements_per_layer": config.hidden_size,
            "dtype": "float32",
            "sha256": _sha256(bytes(boundary_payload)),
        }, indent=2) + "\n")
    if attention_boundaries_output is not None:
        if not attention_boundary_payload or attention_boundary_count != config.num_hidden_layers:
            raise ValueError("requested attention boundary capture did not match one complete model step")
        attention_boundaries_output.parent.mkdir(parents=True, exist_ok=True)
        attention_boundaries_output.write_bytes(attention_boundary_payload)
        attention_boundaries_output.with_suffix(".json").write_text(json.dumps({
            "case": attention_boundary_case,
            "step": attention_boundary_step,
            "layers": attention_boundary_count,
            "elements_per_layer": config.hidden_size,
            "dtype": "float32",
            "contract": "post_token_mixer_residual",
            "sha256": _sha256(bytes(attention_boundary_payload)),
        }, indent=2) + "\n")
    if kv_output is not None:
        if not kv_payload or kv_metadata is None:
            raise ValueError("requested KV capture did not match a corpus case, layer, and step")
        kv_output.parent.mkdir(parents=True, exist_ok=True)
        kv_output.write_bytes(kv_payload)
        kv_metadata.update({
            "dtype": "float32",
            "layout": "[positions, heads, head_dimension] key then value",
            "bytes": len(kv_payload),
            "sha256": _sha256(bytes(kv_payload)),
            "round_kv": round_kv,
        })
        kv_output.with_suffix(".json").write_text(json.dumps(kv_metadata, indent=2) + "\n")

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
            "round_kv": round_kv,
            "round_linear_state": round_linear_state,
            "device": str(device),
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
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.hidden_output is not None and args.hidden_case is None:
        parser.error("--hidden-output requires --hidden-case")
    if args.hidden_step is not None and args.hidden_step < 0:
        parser.error("--hidden-step must be non-negative")
    if args.boundaries_output is not None and args.boundary_case is None:
        parser.error("--boundaries-output requires --boundary-case")
    if args.boundary_step is not None and args.boundary_step < 0:
        parser.error("--boundary-step must be non-negative")
    if args.attention_boundaries_output is not None and args.attention_boundary_case is None:
        parser.error("--attention-boundaries-output requires --attention-boundary-case")
    if args.attention_boundary_step is not None and args.attention_boundary_step < 0:
        parser.error("--attention-boundary-step must be non-negative")
    if args.kv_output is not None and args.kv_case is None:
        parser.error("--kv-output requires --kv-case")
    if args.kv_output is not None and args.kv_step is None:
        parser.error("--kv-output requires --kv-step")
    if args.kv_output is not None and args.kv_layer is None:
        parser.error("--kv-output requires --kv-layer")
    if args.kv_step is not None and args.kv_step < 0:
        parser.error("--kv-step must be non-negative")
    if args.kv_layer is not None and args.kv_layer < 0:
        parser.error("--kv-layer must be non-negative")
    corpus = json.loads(args.corpus.read_text())
    report = _run_cases(args.model_dir, select_cases(corpus, args.case_ids), args.output_dir,
                        args.round_activations, args.round_kv, args.round_linear_state,
                        args.hidden_output,
                        args.hidden_case, args.hidden_step, args.boundaries_output,
                        args.boundary_case, args.boundary_step,
                        args.attention_boundaries_output, args.attention_boundary_case,
                        args.attention_boundary_step, args.kv_output, args.kv_case,
                        args.kv_step, args.kv_layer, args.device)
    (args.output_dir / "corpus-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [case["id"] for case in report["cases"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
