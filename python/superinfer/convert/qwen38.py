"""Strict, dependency-free Qwen3.8 source inventory validation.

The validator reads JSON metadata and safetensors headers only. It never reads model payload bytes;
conversion code can therefore reject a bad source before allocating bulk host/device storage.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class Qwen38ValidationError(ValueError):
    """A source/config/tensor contract violation with a stable field-oriented diagnostic."""

    def __init__(self, code: str, field: str, message: str) -> None:
        self.code = code
        self.field = field
        super().__init__(f"{code} [{field}]: {message}")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_start: int
    data_end: int


@dataclass(frozen=True)
class Qwen38Inventory:
    upstream_repository: str
    upstream_revision: str
    derivative_repository: str
    derivative_revision: str
    config: Mapping[str, Any]
    tensors: tuple[TensorRecord, ...]
    file_hashes: Mapping[str, str]

    def canonical_tensor_bytes(self) -> bytes:
        return json.dumps(
            [asdict(tensor) for tensor in self.tensors],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def manifest(self) -> dict[str, Any]:
        return {
            "model": "Qwen3.8-27B",
            "upstream_repository": self.upstream_repository,
            "upstream_revision": self.upstream_revision,
            "derivative_repository": self.derivative_repository,
            "derivative_revision": self.derivative_revision,
            "architecture": self.config["architectures"][0],
            "config": self.config,
            "tensor_count": len(self.tensors),
            "tensor_inventory_sha256": hashlib.sha256(self.canonical_tensor_bytes()).hexdigest(),
            "file_sha256": dict(sorted(self.file_hashes.items())),
            "license": "apache-2.0",
        }


_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "hf_quant_config.json",
)


def _read_json(path: Path, field: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise Qwen38ValidationError("missing_file", field, str(path.name)) from error
    except json.JSONDecodeError as error:
        raise Qwen38ValidationError("invalid_json", field, str(error)) from error


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise Qwen38ValidationError("invalid_type", field, "expected an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Qwen38ValidationError("invalid_value", field, "expected a positive integer")
    return value


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    if architectures != ["Qwen3_5ForConditionalGeneration"]:
        raise Qwen38ValidationError(
            "unsupported_architecture", "architectures", "expected Qwen3_5ForConditionalGeneration"
        )
    if config.get("model_type") != "qwen3_5":
        raise Qwen38ValidationError("unsupported_model_type", "model_type", "expected qwen3_5")
    text = _required_mapping(config.get("text_config"), "text_config")
    expected = {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
    }
    for field, expected_value in expected.items():
        actual = text.get(field)
        if actual != expected_value:
            raise Qwen38ValidationError("config_mismatch", f"text_config.{field}", f"expected {expected_value}, got {actual!r}")
    layers = text.get("layer_types")
    if not isinstance(layers, list) or len(layers) != 64 or any(
        layer not in {"linear_attention", "full_attention"} for layer in layers
    ):
        raise Qwen38ValidationError("config_mismatch", "text_config.layer_types", "expected 64 known layer types")
    if text.get("full_attention_interval") != 4:
        raise Qwen38ValidationError("config_mismatch", "text_config.full_attention_interval", "expected 4")
    epsilon = text.get("rms_norm_eps")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not 0.0 < epsilon < 1.0:
        raise Qwen38ValidationError("invalid_value", "text_config.rms_norm_eps", "expected 0 < epsilon < 1")
    for field in ("bos_token_id", "eos_token_id", "vocab_size"):
        _positive_int(text.get(field), f"text_config.{field}")
    return dict(config)


def _validate_tokenizer(model_dir: Path) -> None:
    tokenizer = _required_mapping(
        _read_json(model_dir / "tokenizer_config.json", "tokenizer_config"),
        "tokenizer_config",
    )
    if tokenizer.get("tokenizer_class") != "Qwen2Tokenizer":
        raise Qwen38ValidationError(
            "tokenizer_mismatch", "tokenizer_config.tokenizer_class", "expected Qwen2Tokenizer"
        )
    if tokenizer.get("model_max_length") != 262144:
        raise Qwen38ValidationError(
            "tokenizer_mismatch", "tokenizer_config.model_max_length", "expected 262144"
        )
    if tokenizer.get("eos_token") != "<|im_end|>":
        raise Qwen38ValidationError(
            "tokenizer_mismatch", "tokenizer_config.eos_token", "expected <|im_end|>"
        )


def _tensor_role(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "normalization"
    if "linear_attn" in name or "self_attn" in name or "full_attention" in name:
        return "attention"
    if ".mlp." in name or ".ffn." in name:
        return "feed_forward"
    return "weight"


def _safetensors_header(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise Qwen38ValidationError("truncated_tensor_file", path.name, "missing header size")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size == 0 or header_size > 128 * 1024 * 1024:
                raise Qwen38ValidationError("invalid_tensor_header", path.name, "header size is unsafe")
            header = stream.read(header_size)
    except OSError as error:
        raise Qwen38ValidationError("tensor_io", path.name, str(error)) from error
    if len(header) != header_size:
        raise Qwen38ValidationError("truncated_tensor_file", path.name, "header is truncated")
    try:
        value = json.loads(header)
    except json.JSONDecodeError as error:
        raise Qwen38ValidationError("invalid_tensor_header", path.name, str(error)) from error
    return _required_mapping(value, path.name)


def _inventory(model_dir: Path, index: Mapping[str, Any]) -> tuple[TensorRecord, ...]:
    weight_map = _required_mapping(index.get("weight_map"), "model.safetensors.index.json.weight_map")
    grouped: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard.endswith(".safetensors"):
            raise Qwen38ValidationError("invalid_tensor_index", "weight_map", "tensor names/shards are invalid")
        grouped.setdefault(shard, []).append(name)
    records: list[TensorRecord] = []
    for shard, expected_names in sorted(grouped.items()):
        path = model_dir / shard
        header = _safetensors_header(path)
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            raw_header_size = stream.read(8)
        header_size = struct.unpack("<Q", raw_header_size)[0]
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        if set(tensors) != set(expected_names):
            raise Qwen38ValidationError("tensor_index_mismatch", shard, "index and safetensors names differ")
        for name in sorted(expected_names):
            descriptor = _required_mapping(tensors[name], f"{shard}:{name}")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if not isinstance(dtype, str) or not isinstance(shape, list) or not all(
                isinstance(dimension, int) and dimension > 0 for dimension in shape
            ) or not isinstance(offsets, list) or len(offsets) != 2 or not all(
                isinstance(offset, int) and offset >= 0 for offset in offsets
            ):
                raise Qwen38ValidationError("invalid_tensor_descriptor", f"{shard}:{name}", "invalid dtype, shape, or offsets")
            start, end = offsets
            if start >= end or 8 + header_size + end > file_size:
                raise Qwen38ValidationError("invalid_tensor_descriptor", f"{shard}:{name}", "payload offsets exceed shard")
            records.append(TensorRecord(name, _tensor_role(name), dtype, tuple(shape), shard, start, end))
    names = [record.name for record in records]
    if len(names) != len(set(names)):
        raise Qwen38ValidationError("duplicate_tensor", "weight_map", "tensor names are not unique")
    return tuple(records)


def validate_source(
    model_dir: Path,
    *,
    upstream_revision: str,
    derivative_revision: str,
    upstream_repository: str = "Qwen/Qwen3.8-27B",
    derivative_repository: str = "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090",
) -> Qwen38Inventory:
    """Validate a pinned source directory and return deterministic metadata only."""

    if not upstream_revision or len(upstream_revision) != 40 or not all(
        character in "0123456789abcdef" for character in upstream_revision
    ):
        raise Qwen38ValidationError("invalid_revision", "upstream_revision", "expected a 40-character lowercase SHA")
    if not derivative_revision or len(derivative_revision) != 40 or not all(
        character in "0123456789abcdef" for character in derivative_revision
    ):
        raise Qwen38ValidationError("invalid_revision", "derivative_revision", "expected a 40-character lowercase SHA")
    config = _validate_config(_required_mapping(_read_json(model_dir / "config.json", "config"), "config"))
    _validate_tokenizer(model_dir)
    index = _required_mapping(_read_json(model_dir / "model.safetensors.index.json", "tensor_index"), "tensor_index")
    tensors = _inventory(model_dir, index)
    file_hashes: dict[str, str] = {}
    for filename in _REQUIRED_FILES:
        path = model_dir / filename
        if not path.is_file():
            raise Qwen38ValidationError("missing_file", filename, "required provenance input is missing")
        file_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not (model_dir / "chat_template.jinja").read_text(encoding="utf-8"):
        raise Qwen38ValidationError("invalid_template", "chat_template.jinja", "template is empty")
    return Qwen38Inventory(
        upstream_repository,
        upstream_revision,
        derivative_repository,
        derivative_revision,
        config,
        tensors,
        file_hashes,
    )
