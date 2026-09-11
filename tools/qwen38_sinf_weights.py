#!/usr/bin/env python3
"""Bounded `.sinf` weight provider for the S03-R same-artifact oracle.

Reads exact packed tensor bytes from the deployment artifact SuperInfer
executes, decodes them with PyTorch operations, and feeds the independent
Transformers layer oracle. Numerical execution uses torch; only artifact
parsing helpers (which cannot hide arithmetic behavior) are shared with
project-local code. SuperInfer CUDA kernels are never reused here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tensor_table(artifact: Path) -> tuple[dict[str, Mapping[str, Any]], Any]:
    sys.path.insert(0, str(_repo_root() / "python"))
    from superinfer.artifact import _read_tensor_table

    return _read_tensor_table(artifact)


def _tensor_payload(artifact: Path, tensor: Mapping[str, Any], payload_record: Any) -> bytes:
    sys.path.insert(0, str(_repo_root() / "python"))
    from superinfer.artifact import _read_located_tensor

    return _read_located_tensor(artifact, tensor, payload_record)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SinfWeights:
    """Bounded per-tensor reads from a validated `.sinf` payload artifact."""

    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self._tensors, self._payload = _tensor_table(artifact)

    def names(self) -> list[str]:
        return sorted(self._tensors)

    def __contains__(self, name: object) -> bool:
        return name in self._tensors

    def raw(self, name: str) -> tuple[bytes, Mapping[str, Any]]:
        tensor = self._tensors.get(name)
        if tensor is None:
            raise KeyError(f"tensor is not present in artifact: {name}")
        return _tensor_payload(self.artifact, tensor, self._payload), tensor

    def tensor_count(self) -> int:
        return len(self._tensors)

    def load(self, name: str):
        """Decode one artifact tensor to a CPU torch tensor.

        BF16/F32/F16 payloads become float-capable tensors; packed NVFP4
        weights stay ``uint8`` storage and FP8 block scales stay
        ``float8_e4m3fn`` so the caller dequantizes explicitly.
        """
        import torch

        payload, record = self.raw(name)
        dtype = str(record["dtype"])
        shape = tuple(int(d) for d in record["shape"])
        if dtype == "BF16":
            return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(shape).clone()
        if dtype == "F32":
            return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape).clone()
        if dtype == "F16":
            return torch.frombuffer(bytearray(payload), dtype=torch.float16).reshape(shape).clone()
        if dtype == "U8":
            return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(shape).clone()
        if dtype == "F8_E4M3":
            return torch.frombuffer(bytearray(payload), dtype=torch.float8_e4m3fn).reshape(shape).clone()
        if dtype == "I8":
            return torch.frombuffer(bytearray(payload), dtype=torch.int8).reshape(shape).clone()
        if dtype == "I32":
            return torch.frombuffer(bytearray(payload), dtype=torch.int32).reshape(shape).clone()
        raise ValueError(f"unsupported artifact dtype for {name}: {dtype}")

    def nvfp4(self, name: str):
        """Dequantize one packed NVFP4 weight with torch ops.

        Replicates the ModelOpt NVFP4 spec used by the safetensors oracle:
        low nibble holds the even input, high nibble the odd input, magnitudes
        follow the standard 8-entry codebook, each block scale covers 16
        consecutive inputs and is multiplied by the per-tensor float scale.
        Execution is torch-based and independent of SuperInfer kernels.
        """
        import torch

        packed = self.load(name)
        scales = self.load(name + "_scale").to(torch.float32)
        tensor_scale = self.load(name + "_scale_2").to(torch.float32)
        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
        )
        low = packed & 0x0F
        high = packed >> 4
        codes = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
        values = magnitudes[(codes & 0x07).long()]
        values = torch.where((codes & 0x08) != 0, -values, values)
        return values * scales.repeat_interleave(16, dim=1) * tensor_scale

    def provenance(self, compute_artifact_sha256: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "artifact": str(self.artifact),
            "tensor_count": self.tensor_count(),
        }
        if compute_artifact_sha256:
            record["artifact_sha256"] = sha256_of(self.artifact)
        return record
