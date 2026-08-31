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
from transformers import DynamicCache, Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)


class BF16StorageDynamicCache(DynamicCache):
    """DynamicCache that exposes the deployment's BF16 KV storage contract before attention."""

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        key_states = key_states.to(torch.bfloat16).to(torch.float32)
        value_states = value_states.to(torch.bfloat16).to(torch.float32)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


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


def _read_f32(path: Path) -> torch.Tensor:
    payload = path.read_bytes()
    if len(payload) % 4 != 0:
        raise SystemExit(f"{path} does not contain whole FP32 values")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--input-f32", type=Path,
                        help="optional FP32 hidden input for a single custom-position step")
    parser.add_argument("--kv-f32", type=Path,
                        help="optional FP32 key-then-value cache snapshot for a custom step")
    parser.add_argument("--start-position", type=int,
                        help="position of the custom input; requires --input-f32 and --kv-f32")
    args = parser.parse_args()

    custom_step = args.input_f32 is not None or args.kv_f32 is not None or args.start_position is not None
    if custom_step and (args.input_f32 is None or args.kv_f32 is None or args.start_position is None):
        parser.error("--input-f32, --kv-f32, and --start-position must be supplied together")
    if args.start_position is not None and not 0 <= args.start_position < 262144:
        parser.error("--start-position must be in [0, 262144)")

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

    rotary = Qwen3_5TextRotaryEmbedding(config)
    if args.decode_steps <= 0:
        raise SystemExit("--decode-steps must be positive")
    cache = BF16StorageDynamicCache(config=config)
    outputs = []
    attention_outputs = []
    if args.diagnostics:
        layer.self_attn.register_forward_hook(
            lambda _module, _inputs, output: attention_outputs.append(
                (output[0] if isinstance(output, tuple) else output).reshape(-1).detach().clone()))
    with torch.inference_mode():
        if custom_step:
            input_values = _read_f32(args.input_f32)
            expected_input_elements = config.hidden_size
            if input_values.numel() != expected_input_elements:
                raise SystemExit(
                    f"custom input must contain {expected_input_elements} FP32 values")
            kv_values = _read_f32(args.kv_f32)
            positions = args.start_position
            elements_per_cache = positions * config.num_key_value_heads * config.head_dim
            expected_kv_elements = 2 * elements_per_cache
            if kv_values.numel() != expected_kv_elements:
                raise SystemExit(
                    f"custom KV snapshot must contain {expected_kv_elements} FP32 values")
            if positions:
                key = kv_values[:elements_per_cache].reshape(
                    positions, config.num_key_value_heads, config.head_dim
                ).permute(1, 0, 2).unsqueeze(0)
                value = kv_values[elements_per_cache:].reshape(
                    positions, config.num_key_value_heads, config.head_dim
                ).permute(1, 0, 2).unsqueeze(0)
                cache.update(key, value, args.layer)
            step_values = [positions]
            hidden_values = [input_values.reshape(1, 1, -1)]
        else:
            step_values = list(range(args.decode_steps))
            hidden_values = []
            for step in step_values:
                hidden = torch.linspace(-0.25, 0.25, config.hidden_size,
                                        dtype=torch.float32).reshape(1, 1, -1)
                hidden_values.append(hidden + step * 0.03125)
        for step, hidden in zip(step_values, hidden_values):
            position_ids = torch.full((1, 1), step, dtype=torch.long)
            position_embeddings = rotary(hidden, position_ids)
            causal_mask = torch.zeros((1, 1, 1, step + 1), dtype=torch.float32)
            output = layer(hidden, position_embeddings=position_embeddings,
                           attention_mask=causal_mask, position_ids=position_ids,
                           past_key_values=cache)
            outputs.append(output.reshape(-1).contiguous())
    output = torch.cat(outputs)
    if args.diagnostics:
        torch.cat(attention_outputs).numpy().astype("float32").tofile(
            args.output.with_suffix(".attn.bin"))
    if args.diagnostics and args.decode_steps == 1:
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
