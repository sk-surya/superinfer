#!/usr/bin/env python3
"""P8 activation-quantization error for NVFP4 (E2M1 + UE4M3 block scale, block 16).

Characterizes the error introduced by quantizing an FP32 activation vector to the
native block-scaled NVFP4 operand contract. Deterministic reference only; no
scale rule is chosen to pass a gate.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Sequence

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_max() -> float:
    # Largest finite UE4M3 (E4M3FN, positive) magnitude.
    return 448.0


def quantize_block(values: Sequence[float], rule: str) -> tuple[list[float], int, float]:
    """Quantize one 16-value block; return dequantized values, clipped count, scale."""
    amax = max(abs(v) for v in values) if values else 0.0
    if rule == "amax_over_6":
        scale = amax / 6.0 if amax > 0.0 else 0.0
    elif rule == "amax_over_7":
        scale = amax / 7.0 if amax > 0.0 else 0.0
    elif rule == "amax_over_4":
        scale = amax / 4.0 if amax > 0.0 else 0.0
    else:
        raise ValueError(f"unknown scale rule: {rule}")
    clipped = 0
    if scale > _e4m3_max():
        scale = _e4m3_max()
    if scale == 0.0:
        return [0.0 for _ in values], 0, 0.0
    out: list[float] = []
    for value in values:
        code = value / scale
        # Nearest E2M1 magnitude with sign.
        sign = -1.0 if code < 0 else 1.0
        magnitude = abs(code)
        if magnitude > 6.0:
            clipped += 1
            magnitude = 6.0
        best = min(E2M1_MAGNITUDES, key=lambda m: abs(m - magnitude))
        out.append(sign * best * scale)
    return out, clipped, scale


def characterize(values: Sequence[float], rule: str, block: int = 16) -> dict[str, Any]:
    dequantized: list[float] = []
    clipped = 0
    blocks = 0
    for start in range(0, len(values), block):
        chunk = list(values[start:start + block])
        out, c, _ = quantize_block(chunk, rule)
        dequantized.extend(out)
        clipped += c
        blocks += 1
    errors = [abs(a - b) for a, b in zip(values, dequantized)]
    reference_norm = math.sqrt(sum(v * v for v in values))
    error_norm = math.sqrt(sum(e * e for e in errors))
    dot = sum(a * b for a, b in zip(values, dequantized))
    dequant_norm = math.sqrt(sum(v * v for v in dequantized))
    cosine = dot / (reference_norm * dequant_norm) if reference_norm and dequant_norm else 1.0
    return {
        "elements": len(values),
        "blocks": blocks,
        "rule": rule,
        "max_abs": max(errors) if errors else 0.0,
        "mean_abs": sum(errors) / len(errors) if errors else 0.0,
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0,
        "relative_l2": error_norm / reference_norm if reference_norm else 0.0,
        "cosine_similarity": cosine,
        "clipped_values": clipped,
        "clipped_fraction": clipped / len(values) if values else 0.0,
        "reference_amax": max(abs(v) for v in values) if values else 0.0,
    }


def load_f32(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 4 != 0:
        raise ValueError(f"{path} is not FP32-aligned")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", type=Path, action="append", required=True,
                        help="one or more FP32 activation captures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=16)
    args = parser.parse_args()
    report: dict[str, Any] = {"schema": "superinfer.p8.activation-quantization.v1", "vectors": []}
    for path in args.vector:
        values = load_f32(path)
        entry: dict[str, Any] = {"path": str(path), "elements": len(values), "rules": {}}
        for rule in ("amax_over_6", "amax_over_7", "amax_over_4"):
            entry["rules"][rule] = characterize(values, rule, args.block)
        # Layer/projection proxy metrics: first half (attention-ish) vs second half.
        half = len(values) // 2
        entry["first_half_rmse_amax6"] = characterize(values[:half], "amax_over_6", args.block)["rmse"]
        entry["second_half_rmse_amax6"] = characterize(values[half:], "amax_over_6", args.block)["rmse"]
        report["vectors"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for entry in report["vectors"]:
        block = entry["rules"]["amax_over_6"]
        print(f"{Path(entry['path']).name}: elements={entry['elements']} "
              f"max_abs={block['max_abs']:.6g} mean_abs={block['mean_abs']:.6g} "
              f"rmse={block['rmse']:.6g} rel_l2={block['relative_l2']:.6g} "
              f"cos={block['cosine_similarity']:.8f} clipped={block['clipped_fraction']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
