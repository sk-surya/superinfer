#!/usr/bin/env python3
"""Emit a one-token full Qwen3.8 NVFP4 reference result.

The reference streams one decoder layer at a time through the pinned Transformers implementation.
That keeps the CPU working set bounded while preserving the exact layer order and shared recurrent/
KV cache semantics used by the complete model.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)


def _tensor(model_dir: Path, index: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(str(model_dir / index[name]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _nvfp4(model_dir: Path, index: dict[str, str], name: str) -> torch.Tensor:
    packed = _tensor(model_dir, index, name)
    scales = _tensor(model_dir, index, name + "_scale").to(torch.float32)
    tensor_scale = _tensor(model_dir, index, name + "_scale_2").to(torch.float32)
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    values = magnitudes[(codes & 0x07).long()]
    values = torch.where((codes & 0x08) != 0, -values, values)
    return values * scales.to(torch.float32).repeat_interleave(16, dim=1) * tensor_scale


def _load_layer(model_dir: Path, index: dict[str, str], config: Qwen3_5TextConfig,
                layer_index: int) -> Qwen3_5DecoderLayer:
    layer = Qwen3_5DecoderLayer(config, layer_index).eval()
    prefix = f"model.language_model.layers.{layer_index}."
    state: dict[str, torch.Tensor] = {}
    for key in layer.state_dict():
        source_name = prefix + key
        if source_name not in index:
            raise KeyError(source_name)
        if source_name.endswith(".weight") and source_name + "_scale" in index:
            state[key] = _nvfp4(model_dir, index, source_name)
        else:
            state[key] = _tensor(model_dir, index, source_name).to(torch.float32)
    layer.load_state_dict(state, strict=True)
    return layer


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + epsilon) * (weight + 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-output", type=Path,
                        help="optional FP32 output path for the final normalized hidden row")
    parser.add_argument("--boundaries-output", type=Path,
                        help="optional FP32 output path for the post-layer hidden rows")
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--tokens", type=str,
                        help="comma-separated token IDs for a shared-cache continuation")
    parser.add_argument("--round-activations", action="store_true",
                        help="round layer outputs through BF16 to model the deployment contract")
    args = parser.parse_args()

    root = json.loads((args.model_dir / "config.json").read_text())
    # The multimodal Qwen3_5Config embeds a text config and defaults the
    # decoder view to its legacy 32-layer shape when constructed from only
    # the nested dictionary.  Build the actual text configuration directly so
    # DynamicCache receives all 64 authored layer types.
    config = Qwen3_5TextConfig.from_dict(root["text_config"])
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    torch.set_num_threads(max(1, min(32, torch.get_num_threads())))

    tokens = ([int(value) for value in args.tokens.split(",") if value.strip()]
              if args.tokens is not None else [args.token])
    if not tokens:
        raise ValueError("--tokens must contain at least one token ID")
    embedding = _tensor(args.model_dir, index, "model.language_model.embed_tokens.weight").to(torch.float32)
    del embedding
    cache = DynamicCache(config=config)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    logit_rows = []
    hidden_rows = []
    greedy_sequence = []

    with torch.inference_mode():
        boundaries = []
        for position, token in enumerate(tokens):
            hidden = _tensor(args.model_dir, index,
                             "model.language_model.embed_tokens.weight")[token].to(torch.float32)
            hidden = hidden.reshape(1, 1, -1)
            position_ids = torch.full((1, 1), position, dtype=torch.long)
            position_embeddings = rotary(hidden, position_ids)
            causal_mask = torch.zeros((1, 1, 1, position + 1), dtype=torch.float32)
            for layer_index in range(config.num_hidden_layers):
                layer = _load_layer(args.model_dir, index, config, layer_index)
                output = layer(hidden, position_embeddings=position_embeddings,
                               attention_mask=causal_mask, position_ids=position_ids,
                               past_key_values=cache)
                hidden = output[0] if isinstance(output, tuple) else output
                if args.round_activations:
                    hidden = hidden.to(torch.bfloat16).to(torch.float32)
                if args.boundaries_output is not None:
                    boundaries.append(hidden.reshape(-1).contiguous().numpy().astype("float32"))
                del layer
                gc.collect()

            final_norm = _tensor(args.model_dir, index, "model.language_model.norm.weight").to(torch.float32)
            hidden = _rms_norm(hidden, final_norm, config.rms_norm_eps)
            del final_norm
            if args.hidden_output is not None:
                hidden_rows.append(hidden.reshape(-1).contiguous().numpy().astype("float32"))
            lm_head = _nvfp4(args.model_dir, index, "lm_head.weight")
            logits = F.linear(hidden.reshape(1, -1), lm_head)
            logit_rows.append(logits.reshape(-1).contiguous().numpy().astype("float32"))
            greedy_sequence.append(int(torch.argmax(logits, dim=-1).item()))

    if args.boundaries_output is not None:
        args.boundaries_output.parent.mkdir(parents=True, exist_ok=True)
        args.boundaries_output.write_bytes(b"".join(row.tobytes() for row in boundaries))

    if args.hidden_output is not None:
        args.hidden_output.parent.mkdir(parents=True, exist_ok=True)
        args.hidden_output.write_bytes(b"".join(row.tobytes() for row in hidden_rows))
    values = b"".join(row.tobytes() for row in logit_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(values)
    diagnostics = {
        "model": "Qwen3.8-27B-NVFP4-RTX5090",
        "reference": "transformers Qwen3_5DecoderLayer streamed one layer at a time",
        "transformers_version": __import__("transformers").__version__,
        "tokens": tokens,
        "layers": config.num_hidden_layers,
        "steps": len(tokens),
        "logits_per_step": 248320,
        "greedy_sequence": greedy_sequence,
        "checksum": float(sum(float(row.sum()) for row in logit_rows)),
        "output_sha256": __import__("hashlib").sha256(values).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
