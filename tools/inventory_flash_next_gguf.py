#!/usr/bin/env python3
"""Generate deterministic, metadata-only evidence for a split Flash-Next GGUF set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from superinfer.convert.flash_next import build_residency_options, inspect_gguf_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = inspect_gguf_source(args.model_dir, args.manifest)
    layer_bytes: dict[int, int] = {}
    for tensor in inventory["tensors"]:
        match = re.match(r"blk\.(\d+)\.", tensor["name"])
        if match is not None:
            layer = int(match.group(1))
            layer_bytes[layer] = layer_bytes.get(layer, 0) + tensor["nbytes"]
    if sorted(layer_bytes) != list(range(48)):
        raise ValueError("expected complete blk.0 through blk.47 layer inventory")
    ple_bytes = inventory["category_bytes"]["ple"]
    computational_bytes = inventory["tensor_payload_bytes"] - ple_bytes
    layers = [(f"layer.{index}", layer_bytes[index]) for index in range(48)]
    vram_categories = dict(inventory["category_bytes"])
    vram_categories["ple"] = 0
    residency = build_residency_options(
        layers,
        vram_categories,
        recipes={"host_ple_fit_first": {key: 1.0 for key in vram_categories}},
        device_budget_bytes=32607 * 1024 * 1024,
        headroom_bytes=4 * 1024 * 1024 * 1024,
    )[0]
    payload = {
        "schema": "superinfer.s03f.flash-next-gguf-candidate.v1",
        "status": "complete_local_packed_candidate_quality_unqualified",
        "identity": {
            "model_repository": "AtomicChat/Qwen3.8-Flash-Next-GGUF",
            "quantization": "AD-4.27bpw-Q4_K_M-M64",
            "model_dir": str(args.model_dir.resolve()),
            "llama_cpp_repository": "https://github.com/ggml-org/llama.cpp",
            "llama_cpp_revision": "6c84c7d5d8833c6e0df69628f75a0f599797934e",
            "manifest_sha256": _sha256(args.manifest),
            "provenance_sha256": _sha256(args.provenance),
        },
        "contract_observed": {
            key: inventory["metadata"][key]
            for key in sorted(inventory["metadata"])
            if key.startswith("qwen4exp.") and not isinstance(inventory["metadata"][key], list)
        },
        "inventory": {
            key: inventory[key]
            for key in ("shard_count", "split_numbers", "tensor_count", "tensor_payload_bytes",
                        "physical_file_bytes", "category_bytes", "tensor_type_counts", "shards", "tensors")
        },
        "residency_observation": {
            "gpu_count": 2,
            "gpu_memory_bytes_each": 32607 * 1024 * 1024,
            "ple_residency": "host_mmap_only",
            "ple_bytes_excluded_from_vram": ple_bytes,
            "computational_tensor_bytes": computational_bytes,
            "computational_fit_without_workspace": computational_bytes < 2 * 32607 * 1024 * 1024,
            "per_layer_tensor_bytes": [layer_bytes[index] for index in range(48)],
            "projection": residency,
            "quality_evidence": "not_available",
            "official_equivalence": False,
        },
        "limitations": [
            "This is an AtomicChat GGUF conversion, not the pinned official HF safetensors artifact.",
            "Packed-byte fit is not a model-quality qualification.",
            "No SuperInfer reference-equivalence evaluation is claimed.",
            "This evidence does not authorize S03F-02 or any expert paging/caching runtime.",
        ],
        "runtime_change": False,
        "s03f_02_status": "engineering_blocked_on_S03",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
