#!/usr/bin/env python3
"""Compare an independent Qwen3.8 layer equation path with Transformers.

The test loads only one full-attention layer from the BF16 checkpoint.  The independent path
spells out RMSNorm, gated QKV split, partial RoPE, GQA, sigmoid output gating, and the MLP; it
does not call the Transformers layer for its calculation.  This is the staged oracle for the
future SuperInfer physical command capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)


def load_layer(model_dir: Path, layer_index: int, config: Qwen3_5Config) -> Qwen3_5DecoderLayer:
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    layer = Qwen3_5DecoderLayer(config, layer_index).eval()
    prefix = f"model.language_model.layers.{layer_index}."
    state: dict[str, torch.Tensor] = {}
    for key in layer.state_dict():
        name = prefix + key
        with safe_open(str(model_dir / weight_map[name]), framework="pt", device="cpu") as handle:
            state[key] = handle.get_tensor(name).to(dtype=torch.float32)
    layer.load_state_dict(state, strict=True)
    return layer


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon) * (weight + 1.0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference_layer(
    layer: Qwen3_5DecoderLayer,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    attention = layer.self_attn
    normalized = rms_norm(hidden, layer.input_layernorm.weight, epsilon)
    q_gate = F.linear(normalized, attention.q_proj.weight).view(
        hidden.shape[0], hidden.shape[1], attention.config.num_attention_heads, attention.head_dim * 2
    )
    query, gate = torch.chunk(q_gate, 2, dim=-1)
    key = F.linear(normalized, attention.k_proj.weight).view(
        hidden.shape[0], hidden.shape[1], attention.config.num_key_value_heads, attention.head_dim
    )
    value = F.linear(normalized, attention.v_proj.weight).view(
        hidden.shape[0], hidden.shape[1], attention.config.num_key_value_heads, attention.head_dim
    )
    query = rms_norm(query, attention.q_norm.weight, epsilon)
    key = rms_norm(key, attention.k_norm.weight, epsilon)
    cos, sin = position_embeddings
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]
    query = torch.cat((query[..., :rotary_dim] * cos + rotate_half(query[..., :rotary_dim]) * sin,
                       query[..., rotary_dim:]), dim=-1)
    key = torch.cat((key[..., :rotary_dim] * cos + rotate_half(key[..., :rotary_dim]) * sin,
                     key[..., rotary_dim:]), dim=-1)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2).repeat_interleave(attention.config.num_attention_heads //
                                                attention.config.num_key_value_heads, dim=1)
    value = value.transpose(1, 2).repeat_interleave(attention.config.num_attention_heads //
                                                    attention.config.num_key_value_heads, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (attention.head_dim**0.5)
    positions = hidden.shape[1]
    mask = torch.triu(torch.full((1, 1, positions, positions), float("-inf")), diagonal=1)
    probabilities = torch.softmax(scores + mask, dim=-1, dtype=torch.float32)
    attended = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    attended = attended * torch.sigmoid(gate)
    attended = F.linear(attended.reshape(hidden.shape[0], hidden.shape[1], -1), attention.o_proj.weight)
    residual = hidden + attended
    post = rms_norm(residual, layer.post_attention_layernorm.weight, epsilon)
    mlp = layer.mlp
    feed_forward = F.silu(F.linear(post, mlp.gate_proj.weight)) * F.linear(post, mlp.up_proj.weight)
    return residual + F.linear(feed_forward, mlp.down_proj.weight)


def tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().float().contiguous()
    raw = value.numpy().tobytes()
    return {
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
        "l2": torch.linalg.vector_norm(value).item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = json.loads((args.model_dir / "config.json").read_text())
    config = Qwen3_5Config.from_dict(root["text_config"])
    if not config.attn_output_gate or config.rms_norm_eps != 1.0e-6:
        raise SystemExit("unexpected Qwen contract")
    layer = load_layer(args.model_dir, args.layer, config)
    hidden = torch.linspace(-0.25, 0.25, args.sequence_length * config.hidden_size,
                            dtype=torch.float32).reshape(1, args.sequence_length, config.hidden_size)
    position_ids = torch.arange(args.sequence_length, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = torch.triu(
        torch.full((1, 1, args.sequence_length, args.sequence_length), float("-inf")), diagonal=1
    )
    with torch.inference_mode():
        expected = layer(hidden, position_embeddings=position_embeddings, position_ids=position_ids,
                         attention_mask=causal_mask)
        actual = reference_layer(layer, hidden, position_embeddings, config.rms_norm_eps)
    delta = (actual - expected).abs()
    record = {
        "reference": "transformers-5.12.1",
        "independent_path": "qwen38-rms-gated-qkv-rope-gqa-sigmoid-mlp",
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "sequence_length": args.sequence_length,
        "contract": {
            "rms_norm_eps": config.rms_norm_eps,
            "norm_scale": "one_plus_weight",
            "attn_output_gate": config.attn_output_gate,
            "attention_heads": config.num_attention_heads,
            "key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        },
        "expected": tensor_record(expected),
        "actual": tensor_record(actual),
        "delta_max": delta.max().item(),
        "delta_mean": delta.mean().item(),
        "pass": bool(torch.allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
