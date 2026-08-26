#!/usr/bin/env python3
"""Run a deterministic, checkpoint-backed Qwen3.8 full-attention reference.

This intentionally uses the pinned Transformers implementation and only materializes one layer;
the complete 27B model is not required.  The JSON output is a compact provenance/statistics
record that can be compared with a SuperInfer layer capture once the physical projection path is
available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)


def _load_layer(model_dir: Path, layer_index: int, config: Qwen3_5Config) -> Qwen3_5DecoderLayer:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    layer = Qwen3_5DecoderLayer(config, layer_index).eval()
    state: dict[str, torch.Tensor] = {}
    prefix = f"model.language_model.layers.{layer_index}."
    for key in layer.state_dict():
        tensor_name = prefix + key
        shard = index[tensor_name]
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            state[key] = handle.get_tensor(tensor_name).to(dtype=torch.float32)
    layer.load_state_dict(state, strict=True)
    return layer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sequence_length <= 0 or args.layer < 0:
        raise SystemExit("layer and sequence length must be positive")

    root_config = json.loads((args.model_dir / "config.json").read_text())
    config = Qwen3_5Config.from_dict(root_config["text_config"])
    if not config.attn_output_gate or config.rms_norm_eps != 1.0e-6:
        raise SystemExit("pinned Qwen contract is not the expected gated epsilon=1e-6 variant")

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(32, torch.get_num_threads())))
    layer = _load_layer(args.model_dir, args.layer, config)
    hidden = torch.linspace(
        -0.25,
        0.25,
        args.sequence_length * config.hidden_size,
        dtype=torch.float32,
    ).reshape(1, args.sequence_length, config.hidden_size)
    position_ids = torch.arange(args.sequence_length, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = torch.full(
        (1, 1, args.sequence_length, args.sequence_length), float("-inf"), dtype=torch.float32
    ).triu(1)
    with torch.inference_mode():
        output = layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )
    values = output.detach().cpu().float().contiguous()
    raw = values.numpy().tobytes()
    record = {
        "reference": "transformers",
        "transformers_version": __import__("transformers").__version__,
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "sequence_length": args.sequence_length,
        "hidden_size": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.rms_norm_eps,
        "attn_output_gate": config.attn_output_gate,
        "input_pattern": "torch.linspace(-0.25,0.25)",
        "output_shape": list(values.shape),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "output_min": values.min().item(),
        "output_max": values.max().item(),
        "output_mean": values.mean().item(),
        "output_l2": torch.linalg.vector_norm(values).item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
