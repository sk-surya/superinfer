"""Evidence-first Flash-Next source inventory and offline capacity planning.

This module never loads tensor payloads.  It requires an explicitly supplied contract so an
unavailable or changed upstream model cannot be silently treated as Flash-Next.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class FlashNextValidationError(ValueError):
    """Stable, fail-closed source or capacity diagnostic."""


@dataclass(frozen=True)
class FlashNextContract:
    model_type: str
    layer_count: int
    expert_count: int
    top_k: int
    upstream_repository: str
    upstream_revision: str
    reference_repository: str
    reference_revision: str
    qsa_config: Mapping[str, Any]


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_start: int
    data_end: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


@dataclass(frozen=True)
class FlashNextInventory:
    contract: FlashNextContract
    config: Mapping[str, Any]
    tensors: tuple[TensorRecord, ...]
    file_hashes: Mapping[str, str]

    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["tensors"] = [asdict(tensor) for tensor in self.tensors]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def inventory_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FlashNextValidationError(f"missing_file [{label}]") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FlashNextValidationError(f"invalid_json [{label}]") from error
    if not isinstance(value, dict):
        raise FlashNextValidationError(f"invalid_object [{label}]")
    return value


def _header(path: Path) -> tuple[Mapping[str, Any], int]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise FlashNextValidationError(f"truncated_header [{path.name}]")
            header_size = struct.unpack("<Q", prefix)[0]
            header = json.loads(stream.read(header_size))
            payload_size = path.stat().st_size - 8 - header_size
    except FileNotFoundError as error:
        raise FlashNextValidationError(f"missing_file [{path.name}]") from error
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as error:
        raise FlashNextValidationError(f"invalid_safetensors [{path.name}]") from error
    if not isinstance(header, dict) or payload_size < 0:
        raise FlashNextValidationError(f"invalid_safetensors [{path.name}]")
    return header, payload_size


def validate_source(model_dir: Path, contract: FlashNextContract) -> FlashNextInventory:
    """Validate config/index/header metadata and return a deterministic inventory.

    The caller owns the source directory. This function reads metadata and hashes files only;
    it does not materialize tensor payloads. Repository/revision values are compared exactly.
    """
    for field in ("upstream_revision", "reference_revision"):
        revision = getattr(contract, field)
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise FlashNextValidationError(f"invalid_revision [{field}]")
    config = _read_json(model_dir / "config.json", "config")
    provenance = _read_json(model_dir / "provenance.json", "provenance")
    for field in ("upstream_repository", "upstream_revision", "reference_repository", "reference_revision"):
        if provenance.get(field) != getattr(contract, field):
            raise FlashNextValidationError(f"revision_mismatch [{field}]")
    checks = (("model_type", contract.model_type, config.get("model_type")),
              ("layer_count", contract.layer_count, config.get("num_hidden_layers")),
              ("expert_count", contract.expert_count, config.get("num_experts")),
              ("top_k", contract.top_k, config.get("num_experts_per_tok")))
    for field, expected, actual in checks:
        if actual != expected:
            raise FlashNextValidationError(f"config_mismatch [{field}]")
    if not isinstance(config.get("ple"), dict) or not config["ple"]:
        raise FlashNextValidationError("missing_metadata [ple]")
    qsa = config.get("qsa")
    if not isinstance(qsa, dict) or qsa != dict(contract.qsa_config):
        raise FlashNextValidationError("unexpected_configuration [qsa]")
    index = _read_json(model_dir / "model.safetensors.index.json", "tensor_index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise FlashNextValidationError("invalid_mapping [weight_map]")
    tensors: list[TensorRecord] = []
    shard_headers: dict[str, tuple[Mapping[str, Any], int]] = {}
    for name, shard_value in sorted(weight_map.items()):
        if not isinstance(name, str) or not isinstance(shard_value, str):
            raise FlashNextValidationError("invalid_mapping [weight_map]")
        if shard_value not in shard_headers:
            shard_headers[shard_value] = _header(model_dir / shard_value)
        header, payload_size = shard_headers[shard_value]
        descriptor = header.get(name)
        if not isinstance(descriptor, dict):
            raise FlashNextValidationError(f"tensor_index_mismatch [{name}]")
        dtype, shape, offsets = descriptor.get("dtype"), descriptor.get("shape"), descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise FlashNextValidationError(f"invalid_tensor_record [{name}]")
        start, end = offsets
        if not all(isinstance(value, int) for value in (*shape, start, end)) or start < 0 or end < start or end > payload_size:
            raise FlashNextValidationError(f"invalid_tensor_record [{name}]")
        tensors.append(TensorRecord(name, dtype, tuple(shape), shard_value, start, end))
    files = {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
             for name in sorted({"config.json", "provenance.json", "model.safetensors.index.json", *weight_map.values()})}
    return FlashNextInventory(contract, config, tuple(tensors), files)


def classify_tensor_bytes(inventory: FlashNextInventory) -> dict[str, int]:
    """Classify known tensor-name families; unknown names fail closed."""
    totals: dict[str, int] = {"embedding_lm_head": 0, "ple": 0, "routed_experts": 0}
    for tensor in inventory.tensors:
        name = tensor.name.lower()
        if "ple" in name or "ngram" in name:
            category = "ple"
        elif "expert" in name or ".experts." in name:
            category = "routed_experts"
        elif "embed" in name or "lm_head" in name:
            category = "embedding_lm_head"
        else:
            raise FlashNextValidationError(f"unclassified_tensor [{tensor.name}]")
        totals[category] += tensor.nbytes
    return {key: totals[key] for key in sorted(totals)}


def build_residency_options(
    layers: Sequence[tuple[str, int]], category_bytes: Mapping[str, int],
    *, recipes: Mapping[str, Mapping[str, float]], device_budget_bytes: int,
    headroom_bytes: int,
) -> list[dict[str, Any]]:
    """Project recipes and choose a deterministic contiguous two-device layer partition."""
    if device_budget_bytes <= headroom_bytes:
        raise FlashNextValidationError("invalid_budget [headroom]")
    results: list[dict[str, Any]] = []
    for name in sorted(recipes):
        recipe = recipes[name]
        total = sum(int(category_bytes[key] * recipe.get(key, 1.0)) for key in category_bytes)
        capacity = device_budget_bytes - headroom_bytes
        if total > 2 * capacity:
            raise FlashNextValidationError(f"budget_exceeded [{name}]")
        layer_partition: list[dict[str, int]] = []
        running = 0
        first = 0
        for index, (_, size) in enumerate(layers):
            if running and running + size > capacity:
                layer_partition.append({"device": len(layer_partition), "first_layer": first, "last_layer": index - 1})
                first, running = index, 0
            running += size
        if layers:
            layer_partition.append({"device": len(layer_partition), "first_layer": first, "last_layer": len(layers) - 1})
        if len(layer_partition) != 2:
            raise FlashNextValidationError(f"budget_exceeded [{name}]")
        results.append({"name": name, "projected_bytes": total, "headroom_bytes": headroom_bytes,
                        "full_expert_residency_feasible": True, "partition": layer_partition,
                        "quality_evidence": "not_available"})
    return results


def blocked_source_evidence() -> dict[str, Any]:
    """Return canonical evidence for a checkout without the pinned Flash-Next inputs."""
    return {
        "schema": "superinfer.s03f.flash-next-source-evidence.v1",
        "status": "blocked",
        "reason": "No exact Flash-Next model artifact or reference checkout is available in this workspace.",
        "required": ["config.json", "model.safetensors.index.json", "safetensors shards", "reference revision"],
        "checked_workspace": "/srv/repos/superinfer",
        "observed": [".planning/FLASH-NEXT-DESIGN.md", "S03F-01-PLAN.md"],
        "excluded_near_matches": [
            "/srv/models/hf/Qwen3.8-27B-DFlash2",
            "/srv/ai/models/gguf/qwen3-coder-next",
        ],
        "claims_withheld": ["pinned revisions", "tensor counts", "packed byte totals", "quality equivalence", "residency feasibility"],
    }


__all__ = ["FlashNextContract", "FlashNextInventory", "FlashNextValidationError", "blocked_source_evidence", "build_residency_options", "classify_tensor_bytes", "validate_source"]
