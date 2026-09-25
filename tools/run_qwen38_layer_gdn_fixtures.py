#!/usr/bin/env python3
"""Run the Qwen3.8 layer/GDN artifact differential fixtures instead of SKIP.

The C++ tests skip unless their reference captures are supplied. This driver
generates the deterministic references from the pinned safetensors source and
runs the tests with the required environment, so the fixtures actually execute.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL_DIR = Path("/srv/models/hf/Qwen3.8-27B-NVFP4-RTX5090-LMHead4")
DEFAULT_ARTIFACT = Path("build/evidence/qwen38-payload-v1-final-a.sinf")
DEFAULT_BIN_DIR = Path("build/cuda-sm120a/tests")


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--gdn-layer", type=int, default=0)
    parser.add_argument("--gdn-segments", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["SUPERINFER_QWEN38_ARTIFACT"] = str(args.artifact)
    with tempfile.TemporaryDirectory(prefix="qwen38-fixtures-") as temporary:
        work = Path(temporary)
        layer_ref = work / "layer3-reference.f32"
        run([
            sys.executable, str(root / "tools/qwen38_nvfp4_layer_reference.py"),
            "--model-dir", str(args.model_dir), "--layer", str(args.layer),
            "--output", str(layer_ref),
        ], env)
        layer_env = dict(env)
        layer_env["SUPERINFER_QWEN38_REFERENCE_F32"] = str(layer_ref)
        run([str(args.bin_dir / "superinfer_sm120_qwen38_layer_artifact")], layer_env)

        # The C++ test derives companion paths by inserting ".attn"/".state"
        # before a ".bin" suffix, so the reference must be named *.bin.
        gdn_ref = work / "gdn-reference.bin"
        run([
            sys.executable, str(root / "tools/qwen38_nvfp4_gdn_reference.py"),
            "--model-dir", str(args.model_dir), "--layer", str(args.gdn_layer),
            "--segments", str(args.gdn_segments), "--output", str(gdn_ref),
        ], env)
        gdn_env = dict(env)
        gdn_env["SUPERINFER_QWEN38_GDN_REFERENCE_F32"] = str(gdn_ref)
        run([str(args.bin_dir / "superinfer_sm120_qwen38_gdn_artifact")], gdn_env)
    print("layer/GDN fixtures executed and passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
