#!/usr/bin/env python3
"""Emit a two-segment Qwen3.8 Gated-DeltaNet layer oracle.

The layer is the pinned Transformers implementation with every packed NVFP4 projection
dequantized from the acceptance checkpoint. Two one-token calls share DynamicCache, so the
resulting output checks both the layer equations and convolution/recurrent state continuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import DynamicCache, Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true",
                        help="also emit intermediate projection and convolution captures")
    args = parser.parse_args()

    model_dir = args.model_dir
    root = json.loads((model_dir / "config.json").read_text())
    config = Qwen3_5Config.from_dict(root["text_config"])
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    layer = Qwen3_5DecoderLayer(config, 0).eval()
    prefix = "model.language_model.layers.0."
    state = {}
    for key in layer.state_dict():
        source_name = prefix + key
        is_packed = source_name in index and source_name + "_scale" in index
        state[key] = (
            _nvfp4(model_dir, index, source_name)
            if is_packed
            else _tensor(model_dir, index, source_name).to(torch.float32)
        )
    layer.load_state_dict(state, strict=True)

    cache = DynamicCache(config=config)
    outputs = []
    attention_outputs = []
    recurrent_states = []
    qkv_projections = []
    convolved_outputs = []
    if args.diagnostics:
        layer.linear_attn.in_proj_qkv.register_forward_hook(
            lambda _module, _inputs, output: qkv_projections.append(output.reshape(-1).detach().clone())
        )
        original_conv_update = layer.linear_attn.causal_conv1d_update

        def capture_conv_update(*inputs, **kwargs):
            output = original_conv_update(*inputs, **kwargs)
            convolved_outputs.append(output.reshape(-1).detach().clone())
            return output

        layer.linear_attn.causal_conv1d_update = capture_conv_update
        original_conv_fn = layer.linear_attn.causal_conv1d_fn
        if original_conv_fn is not None:
            def capture_conv_fn(*inputs, **kwargs):
                output = original_conv_fn(*inputs, **kwargs)
                convolved_outputs.append(output.reshape(-1).detach().clone())
                return output

            layer.linear_attn.causal_conv1d_fn = capture_conv_fn
        else:
            layer.linear_attn.conv1d.register_forward_hook(
                lambda _module, _inputs, output: convolved_outputs.append(
                    (torch.nn.functional.silu(output[:, :, :1])).reshape(-1).detach().clone()))
    layer.linear_attn.register_forward_hook(
        lambda _module, _inputs, output: attention_outputs.append(output.reshape(-1).detach().clone())
    )
    for segment in range(2):
        hidden = torch.linspace(-0.25, 0.25, config.hidden_size, dtype=torch.float32).reshape(1, 1, -1)
        hidden = hidden + segment * 0.03125
        with torch.inference_mode():
            output = layer(
                hidden,
                position_embeddings=(
                    torch.zeros((1, 1, 256), dtype=torch.float32),
                    torch.zeros((1, 1, 256), dtype=torch.float32),
                ),
                past_key_values=cache,
            )
        outputs.append(output.reshape(-1).contiguous())
        recurrent_states.append(cache.layers[0].recurrent_states.reshape(-1).float().detach().clone())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cat(outputs).numpy().astype("float32").tofile(args.output)
    torch.cat(attention_outputs).numpy().astype("float32").tofile(args.output.with_suffix(".attn.bin"))
    torch.cat(recurrent_states).numpy().astype("float32").tofile(args.output.with_suffix(".state.bin"))
    if args.diagnostics:
        torch.cat(qkv_projections).numpy().astype("float32").tofile(
            args.output.with_suffix(".qkv.bin"))
        torch.cat(convolved_outputs).numpy().astype("float32").tofile(
            args.output.with_suffix(".conv.bin"))
    diagnostics = {
        "model": "Qwen3.8-27B-NVFP4-RTX5090",
        "reference": "transformers 5.12.1 Qwen3_5DecoderLayer",
        "layer": 0,
        "segments": 2,
        "segment_lengths": [1, 1],
        "state": {
            "conv_shape": [1, 10240, 4],
            "recurrent_shape": [1, 48, 128, 128],
            "shared_between_segments": True,
        },
        "output_prefix": [output[:4].tolist() for output in outputs],
        "diagnostics": {
            "qkv_projection": str(args.output.with_suffix(".qkv.bin"))
            if args.diagnostics else None,
            "qkv_shape": [10240],
            "segments": len(qkv_projections),
            "convolution": str(args.output.with_suffix(".conv.bin"))
            if args.diagnostics else None,
            "conv_shape": [10240],
            "conv_segments": len(convolved_outputs),
            "z_shape": [6144],
            "a_shape": [48],
            "b_shape": [48],
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
