#!/usr/bin/env python3
"""Emit a deterministic one-token Qwen layer oracle using the packed NVFP4 checkpoint.

The model module is only used for the pinned Qwen equations. Every dense projection is replaced
with a fresh FP32 dequantization of the artifact's E2M1/FP8/tensor-scale triplet, so the output is
an external semantic reference for the exact quantized payload rather than a BF16-checkpoint
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3_5Config
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
    block_scale = scales.repeat_interleave(16, dim=1)
    return values * block_scale * tensor_scale


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    root = json.loads((args.model_dir / "config.json").read_text())
    config = Qwen3_5Config.from_dict(root["text_config"])
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    layer = Qwen3_5DecoderLayer(config, args.layer).eval()
    prefix = f"model.language_model.layers.{args.layer}."
    state = {key: _tensor(args.model_dir, index, prefix + key).to(torch.float32)
             for key in layer.state_dict()
             if not key.endswith(".weight") or "norm" in key}
    for key in layer.state_dict():
        if key.endswith(".weight") and "norm" not in key:
            state[key] = _nvfp4(args.model_dir, index, prefix + key)
    layer.load_state_dict(state, strict=True)

    hidden = torch.linspace(-0.25, 0.25, config.hidden_size, dtype=torch.float32).reshape(1, 1, -1)
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with torch.inference_mode():
        output = layer(hidden, position_embeddings=position_embeddings,
                       attention_mask=causal_mask, position_ids=position_ids)
    if args.diagnostics:
        with torch.inference_mode():
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            q_gate = torch.nn.functional.linear(normalized, attention.q_proj.weight).reshape(1, 1, 24, 512)
            query, gate = torch.chunk(q_gate, 2, dim=-1)
            key = torch.nn.functional.linear(normalized, attention.k_proj.weight).reshape(1, 1, 4, 256)
            value = torch.nn.functional.linear(normalized, attention.v_proj.weight).reshape(1, 1, 4, 256)
            query = attention.q_norm(query)
            key = attention.k_norm(key)
            cos, sin = position_embeddings
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
            rotary_dim = cos.shape[-1]
            query = torch.cat((query[..., :rotary_dim] * cos + rotate_half(query[..., :rotary_dim]) * sin,
                               query[..., rotary_dim:]), dim=-1)
            key = torch.cat((key[..., :rotary_dim] * cos + rotate_half(key[..., :rotary_dim]) * sin,
                             key[..., rotary_dim:]), dim=-1)
            # Match the physical plan's persistent KV contract: append converts each row to BF16
            # before the decode attention reads it back.
            key = key.to(torch.bfloat16).to(torch.float32)
            value = value.to(torch.bfloat16).to(torch.float32)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2).repeat_interleave(6, dim=1)
            value = value.transpose(1, 2).repeat_interleave(6, dim=1)
            attended = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (256.0 ** 0.5), dim=-1)
            attended = torch.matmul(attended, value).transpose(1, 2).reshape(1, 6144)
            gated = attended * torch.sigmoid(gate.reshape(1, 6144))
            attention_output = torch.nn.functional.linear(gated, attention.o_proj.weight)
            residual = hidden.reshape(1, 5120) + attention_output
            post_norm = layer.post_attention_layernorm(residual)
            gate_projection = torch.nn.functional.linear(post_norm, layer.mlp.gate_proj.weight)
            up_projection = torch.nn.functional.linear(post_norm, layer.mlp.up_proj.weight)
            gated_mlp = torch.nn.functional.silu(gate_projection) * up_projection
            mlp_output = torch.nn.functional.linear(gated_mlp, layer.mlp.down_proj.weight)
            output = (residual + mlp_output).reshape(1, 1, -1)
            diagnostics = {name: tensor.reshape(-1)[:4].tolist() for name, tensor in {
                "normalized": normalized, "q_projection": q_gate, "q_raw": q_gate[..., :256],
                "gate_raw": q_gate[..., 256:512], "gate": gate, "q": query, "q_norm": query,
                "q_rope": query, "attended": attended, "gated": gated,
                "attention_output": attention_output, "residual": residual,
                "post_norm": post_norm, "gate_projection": gate_projection,
                "up_projection": up_projection, "gated_mlp": gated_mlp,
                "mlp_output": mlp_output, "output": output,
            }.items()}
        args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        args.output.with_suffix(".qproj.bin").write_bytes(
            q_gate.reshape(-1).contiguous().numpy().astype("float32").tobytes())
    values = output.reshape(-1).contiguous().numpy().astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(values.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
