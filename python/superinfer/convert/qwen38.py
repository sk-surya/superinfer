"""Strict, dependency-free Qwen3.8 source inventory and provenance validation.

The validator parses JSON metadata and safetensors headers before conversion, then streams indexed
shards to authenticate their payload hashes without materializing model weights in host memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from superinfer.artifact import write_artifact, write_streaming_artifact


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
    quantization: Mapping[str, Any]
    tensors: tuple[TensorRecord, ...]
    file_hashes: Mapping[str, str]

    def canonical_tensor_bytes(self) -> bytes:
        return json.dumps(
            [asdict(tensor) for tensor in self.tensors],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def normalized_tensor_mapping(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic model-to-artifact tensor records without source paths."""

        return tuple(
            {
                "name": tensor.name,
                "role": tensor.role,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "source_shard": tensor.shard,
                "data_start": tensor.data_start,
                "data_end": tensor.data_end,
            }
            for tensor in self.tensors
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "model": "Qwen3.8-27B",
            "upstream_repository": self.upstream_repository,
            "upstream_revision": self.upstream_revision,
            "derivative_repository": self.derivative_repository,
            "derivative_revision": self.derivative_revision,
            "architecture": self.config["architectures"][0],
            "config": self.config,
            "quantization": self.quantization,
            "tensor_count": len(self.tensors),
            "tensor_inventory_sha256": hashlib.sha256(self.canonical_tensor_bytes()).hexdigest(),
            "tensor_mapping": self.normalized_tensor_mapping(),
            "file_sha256": dict(sorted(self.file_hashes.items())),
            "license": "apache-2.0",
        }


PINNED_UPSTREAM_REPOSITORY = "Qwen/Qwen3.8-27B"
PINNED_UPSTREAM_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PINNED_DERIVATIVE_REPOSITORY = "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
PINNED_DERIVATIVE_REVISION = "0cc27958cefbbe231782ec8511de8c4eb5233348"
PINNED_TENSOR_COUNT = 2402
PINNED_TENSOR_INVENTORY_SHA256 = "7342659a53eecbb04c47b5de89d957ca47cb021970cb252575b8b9161d0a84fc"
PINNED_FILE_SHA256 = {
    "config.json": "78f65e03f2ac08a39320bf4a2633f1ae1526144da0fba1904b7371e682c304ea",
    "generation_config.json": "7ae9e193dbcef99733ccf647c95ef668c35d1a80a8aa88a51ee40a9bcacf5a74",
    "model.safetensors.index.json": "4f0c8847dd549636c873737a4703ff1f215a98ec6d5e90b082b31e9e26f4e765",
    "model-00001-of-00003.safetensors": "cdd37b0e61eccc8a3d7d08f9d1a4f52856a9d88e4e8b42089bd18a970e3a01ec",
    "model-00002-of-00003.safetensors": "4b547449a2b23c6cd414da0cf65ff9d7e17ad9aa2b119beedcbba14f649eb1dd",
    "model-00003-of-00003.safetensors": "9ce944d534eabdd493076a3a52c7ebd31f41c135b340a1ea95c5a695e6f1f6b2",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "c873857aae349387312ff4cb76d4a17a8d5ed79f89523146200a1570739132db",
    "chat_template.jinja": "0a20a4673d45476ed88dcb4a60b6af35ca202ae470fee6fd2bd419758aabd9ab",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "hf_quant_config.json": "2c30a0d7e08c5eede4a273c9862aa90f49adfda1cd661dd564742749de9c1a2b",
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
    except OSError as error:
        raise Qwen38ValidationError("metadata_io", field, str(error)) from error
    except json.JSONDecodeError as error:
        raise Qwen38ValidationError("invalid_json", field, str(error)) from error
    except UnicodeError as error:
        raise Qwen38ValidationError("invalid_encoding", field, str(error)) from error


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
    expected_layers = ["linear_attention"] * 64
    expected_layers[3::4] = ["full_attention"] * 16
    if layers != expected_layers:
        raise Qwen38ValidationError("config_mismatch", "text_config.layer_types", "expected the pinned 3-linear/1-full schedule")
    if text.get("full_attention_interval") != 4:
        raise Qwen38ValidationError("config_mismatch", "text_config.full_attention_interval", "expected 4")
    for field, expected_value in {
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
    }.items():
        if text.get(field) != expected_value:
            raise Qwen38ValidationError("config_mismatch", f"text_config.{field}", f"expected {expected_value}")
    epsilon = text.get("rms_norm_eps")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not 0.0 < epsilon < 1.0:
        raise Qwen38ValidationError("invalid_value", "text_config.rms_norm_eps", "expected 0 < epsilon < 1")
    for field in ("bos_token_id", "eos_token_id", "vocab_size"):
        _positive_int(text.get(field), f"text_config.{field}")
    return dict(config)


def _validate_tokenizer(model_dir: Path) -> None:
    tokenizer_json = _required_mapping(_read_json(model_dir / "tokenizer.json", "tokenizer"), "tokenizer")
    model = _required_mapping(tokenizer_json.get("model"), "tokenizer.model")
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict) or not isinstance(model.get("merges"), list):
        raise Qwen38ValidationError("tokenizer_mismatch", "tokenizer.model", "expected a BPE model with vocab and merges")
    vocab = _required_mapping(_read_json(model_dir / "vocab.json", "vocab"), "vocab")
    if len(vocab) != len(model["vocab"]):
        raise Qwen38ValidationError("tokenizer_mismatch", "vocab", "tokenizer.json and vocab.json sizes differ")
    added_tokens = tokenizer_json.get("added_tokens")
    if not isinstance(added_tokens, list) or not any(
        isinstance(token, dict) and token.get("content") == "<|im_end|>" for token in added_tokens
    ):
        raise Qwen38ValidationError("tokenizer_mismatch", "tokenizer.added_tokens", "<|im_end|> is absent")
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


def _validate_quantization(model_dir: Path) -> dict[str, Any]:
    value = _required_mapping(_read_json(model_dir / "hf_quant_config.json", "hf_quant_config"), "hf_quant_config")
    producer = _required_mapping(value.get("producer"), "hf_quant_config.producer")
    quantization = _required_mapping(value.get("quantization"), "hf_quant_config.quantization")
    if producer.get("name") != "modelopt" or not isinstance(producer.get("version"), str):
        raise Qwen38ValidationError("quantization_mismatch", "hf_quant_config.producer", "expected ModelOpt producer")
    if quantization.get("quant_algo") != "NVFP4":
        raise Qwen38ValidationError("quantization_mismatch", "hf_quant_config.quantization.quant_algo", "expected NVFP4")
    if quantization.get("group_size") != 16:
        raise Qwen38ValidationError("quantization_mismatch", "hf_quant_config.quantization.group_size", "expected 16")
    if quantization.get("kv_cache_quant_algo") != "FP8":
        raise Qwen38ValidationError(
            "quantization_mismatch", "hf_quant_config.quantization.kv_cache_quant_algo", "expected FP8"
        )
    excluded = quantization.get("exclude_modules")
    if not isinstance(excluded, list) or not all(isinstance(entry, str) and entry for entry in excluded):
        raise Qwen38ValidationError(
            "quantization_mismatch", "hf_quant_config.quantization.exclude_modules", "expected non-empty strings"
        )
    return {
        "algorithm": "NVFP4",
        "group_size": 16,
        "kv_cache_algorithm": "FP8",
        "producer": {"name": "modelopt", "version": producer["version"]},
        "excluded_module_patterns": sorted(excluded),
    }
def _tensor_role(name: str) -> str:
    if name.endswith(".weight_scale") or name.endswith(".weight_scale_2") or name.endswith(".input_scale"):
        return "scale"
    if name.endswith(".bias") or name.endswith(".dt_bias"):
        return "bias"
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
    except (OSError, struct.error) as error:
        raise Qwen38ValidationError("tensor_io", path.name, str(error)) from error
    if len(header) != header_size:
        raise Qwen38ValidationError("truncated_tensor_file", path.name, "header is truncated")
    try:
        value = json.loads(header)
    except json.JSONDecodeError as error:
        raise Qwen38ValidationError("invalid_tensor_header", path.name, str(error)) from error
    return _required_mapping(value, path.name)


def _safe_shard_path(model_dir: Path, shard: str) -> Path:
    candidate = Path(shard)
    if candidate.is_absolute() or candidate.name != shard or ".." in candidate.parts:
        raise Qwen38ValidationError("invalid_tensor_index", "weight_map", "shard must be a plain basename")
    root = model_dir.resolve()
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Qwen38ValidationError("invalid_tensor_index", shard, "shard resolves outside model directory") from error
    if not path.is_file():
        raise Qwen38ValidationError("missing_file", shard, "indexed shard is absent")
    return path


def _file_sha256(path: Path, field: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Qwen38ValidationError("tensor_io", field, str(error)) from error
    return digest.hexdigest()


def _file_sha256_entry(entry: tuple[Path, str]) -> tuple[str, str]:
    """Hash one validation input; kept module-level for executor portability."""

    path, field = entry
    return field, _file_sha256(path, field)


def _parallel_file_hashes(model_dir: Path, filenames: tuple[str, ...]) -> dict[str, str]:
    """Hash independent source files concurrently while preserving deterministic output."""

    entries: list[tuple[Path, str]] = []
    for filename in filenames:
        path = model_dir / filename
        if not path.is_file():
            raise Qwen38ValidationError("missing_file", filename, "required provenance input is missing")
        entries.append((path, filename))
    if not entries:
        return {}
    worker_count = min(len(entries), max(1, int((os.cpu_count() or 1) * 0.9)))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="sinf-hash") as executor:
        return dict(executor.map(_file_sha256_entry, entries))


def _inventory(model_dir: Path, index: Mapping[str, Any]) -> tuple[TensorRecord, ...]:
    weight_map = _required_mapping(index.get("weight_map"), "model.safetensors.index.json.weight_map")
    grouped: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard.endswith(".safetensors"):
            raise Qwen38ValidationError("invalid_tensor_index", "weight_map", "tensor names/shards are invalid")
        grouped.setdefault(shard, []).append(name)
    records: list[TensorRecord] = []
    for shard, expected_names in sorted(grouped.items()):
        path = _safe_shard_path(model_dir, shard)
        header = _safetensors_header(path)
        try:
            file_size = path.stat().st_size
            with path.open("rb") as stream:
                raw_header_size = stream.read(8)
            header_size = struct.unpack("<Q", raw_header_size)[0]
        except (OSError, struct.error) as error:
            raise Qwen38ValidationError("tensor_io", shard, str(error)) from error
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
    upstream_revision: str = PINNED_UPSTREAM_REVISION,
    derivative_revision: str = PINNED_DERIVATIVE_REVISION,
    upstream_repository: str = PINNED_UPSTREAM_REPOSITORY,
    derivative_repository: str = PINNED_DERIVATIVE_REPOSITORY,
    enforce_pinned: bool = True,
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
    if enforce_pinned and (
        upstream_repository != PINNED_UPSTREAM_REPOSITORY
        or upstream_revision != PINNED_UPSTREAM_REVISION
        or derivative_repository != PINNED_DERIVATIVE_REPOSITORY
        or derivative_revision != PINNED_DERIVATIVE_REVISION
    ):
        raise Qwen38ValidationError("source_identity_mismatch", "source", "repository and revision are immutable")
    config = _validate_config(_required_mapping(_read_json(model_dir / "config.json", "config"), "config"))
    _validate_tokenizer(model_dir)
    quantization = _validate_quantization(model_dir)
    index = _required_mapping(_read_json(model_dir / "model.safetensors.index.json", "tensor_index"), "tensor_index")
    tensors = _inventory(model_dir, index)
    quantization["observed_tensor_dtypes"] = sorted({tensor.dtype for tensor in tensors})
    file_hashes: dict[str, str]
    shard_names = sorted({record.shard for record in tensors})
    file_hashes = _parallel_file_hashes(model_dir, (*_REQUIRED_FILES, *shard_names))
    try:
        template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise Qwen38ValidationError("invalid_template", "chat_template.jinja", str(error)) from error
    if not template:
        raise Qwen38ValidationError("invalid_template", "chat_template.jinja", "template is empty")
    inventory = Qwen38Inventory(
        upstream_repository,
        upstream_revision,
        derivative_repository,
        derivative_revision,
        config,
        quantization,
        tensors,
        file_hashes,
    )
    if enforce_pinned:
        if len(inventory.tensors) != PINNED_TENSOR_COUNT:
            raise Qwen38ValidationError("tensor_schema_mismatch", "tensor_count", f"expected {PINNED_TENSOR_COUNT}")
        if hashlib.sha256(inventory.canonical_tensor_bytes()).hexdigest() != PINNED_TENSOR_INVENTORY_SHA256:
            raise Qwen38ValidationError("tensor_schema_mismatch", "tensor_inventory_sha256", "pinned tensor inventory differs")
        for filename, expected in PINNED_FILE_SHA256.items():
            if inventory.file_hashes.get(filename) != expected:
                raise Qwen38ValidationError("source_hash_mismatch", filename, "pinned source hash differs")
    return inventory


_QWEN38_OPERATION_CAPABILITIES = (
    "embedding",
    "rms_norm",
    "gated_delta_attention",
    "attention",
    "residual",
    "gated_dense_ffn",
    "lm_head",
)
_SM120_EXECUTABLE_BASELINE = frozenset({"rms_norm", "residual"})


def _operation_capabilities() -> list[dict[str, str]]:
    return [
        {
            "semantic_operation": operation,
            "target_capability": operation,
            "baseline_status": "executable" if operation in _SM120_EXECUTABLE_BASELINE else "unavailable",
        }
        for operation in _QWEN38_OPERATION_CAPABILITIES
    ]


def _memory_ledger_for_directory(
    model_dir: Path, inventory: Qwen38Inventory, context: int
) -> dict[str, int]:
    shard_bytes = sum(
        (model_dir / name).stat().st_size
        for name in inventory.file_hashes
        if name.endswith(".safetensors")
    )
    full_attention_layers = 16
    linear_attention_layers = 48
    kv_bytes = full_attention_layers * context * 4 * 256 * 2
    delta_state_bytes = linear_attention_layers * 48 * 128 * 128 * 4
    convolution_state_bytes = linear_attention_layers * 4 * 10240 * 2
    activation_bytes = 8 * 5120 * 2
    workspace_bytes = 256 * 1024 * 1024
    device_budget_bytes = 32 * 1024**3
    required_bytes = (
        shard_bytes
        + kv_bytes
        + delta_state_bytes
        + convolution_state_bytes
        + activation_bytes
        + workspace_bytes
    )
    return {
        "weights": shard_bytes,
        "full_attention_kv": kv_bytes,
        "gated_delta_state": delta_state_bytes,
        "convolution_state": convolution_state_bytes,
        "activation_safety": activation_bytes,
        "workspace": workspace_bytes,
        "device_budget": device_budget_bytes,
        "required": required_bytes,
        "margin": device_budget_bytes - required_bytes,
    }


def write_qwen38_metadata_artifact(
    model_dir: Path,
    output: Path,
    *,
    max_context: int | None = None,
    enforce_pinned: bool = True,
) -> None:
    """Write deterministic source metadata and an explicit pre-execution coverage ledger.

    This recipe intentionally contains no model-weight payload. It is the reproducible conversion
    checkpoint used before a physical plan is accepted; the manifest says exactly which baseline
    capabilities remain unavailable rather than manufacturing executable kernel records.
    """
    inventory = validate_source(model_dir, enforce_pinned=enforce_pinned)
    text_config = inventory.config["text_config"]
    assert isinstance(text_config, dict)
    context = max_context if max_context is not None else int(text_config["max_position_embeddings"])
    if context <= 0 or context > int(text_config["max_position_embeddings"]):
        raise Qwen38ValidationError("invalid_value", "max_context", "context exceeds pinned model capacity")

    manifest = inventory.manifest()
    manifest["conversion"] = {
        "recipe": "qwen38-metadata-v1",
        "max_context": context,
        "payload_included": False,
        "physical_execution_status": "pending-baseline-provider-coverage",
        "operation_capabilities": _operation_capabilities(),
        "memory_ledger_bytes": _memory_ledger_for_directory(model_dir, inventory, context),
    }
    tensor_table = list(inventory.normalized_tensor_mapping())
    physical_plan = json.dumps(
        {
            "version": 1,
            "target": "sm_120a",
            "kernel_catalog": "baseline-v1",
            "status": "pending-baseline-provider-coverage",
            "commands": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    write_artifact(
        {
            "manifest": manifest,
            "tensors": tensor_table,
            "physical_plan": physical_plan,
            "payload_hex": "",
        },
        output,
    )


def write_qwen38_payload_artifact(
    model_dir: Path,
    output: Path,
    *,
    max_context: int | None = None,
    enforce_pinned: bool = True,
) -> None:
    """Write a deterministic `.sinf` with indexed source shards copied in bounded chunks."""
    inventory = validate_source(model_dir, enforce_pinned=enforce_pinned)
    text_config = inventory.config["text_config"]
    if not isinstance(text_config, dict):
        raise Qwen38ValidationError("invalid_type", "text_config", "expected an object")
    context = max_context if max_context is not None else int(text_config["max_position_embeddings"])
    if context <= 0 or context > int(text_config["max_position_embeddings"]):
        raise Qwen38ValidationError("invalid_value", "max_context", "context exceeds pinned model capacity")

    shard_names = sorted({record.shard for record in inventory.tensors})
    payload_files = tuple(model_dir / name for name in shard_names)
    shard_table: list[dict[str, Any]] = []
    tensor_table: list[dict[str, Any]] = []
    payload_offset = 0
    header_sizes: dict[str, int] = {}
    for shard in shard_names:
        path = _safe_shard_path(model_dir, shard)
        try:
            with path.open("rb") as stream:
                raw_size = stream.read(8)
            header_size = struct.unpack("<Q", raw_size)[0]
            shard_size = path.stat().st_size
        except (OSError, struct.error) as error:
            raise Qwen38ValidationError("tensor_io", shard, str(error)) from error
        header_sizes[shard] = header_size
        shard_table.append({
            "name": shard,
            "payload_offset": payload_offset,
            "bytes": shard_size,
            "sha256": inventory.file_hashes[shard],
        })
        payload_offset += shard_size
    for tensor in inventory.tensors:
        shard_base = next(item["payload_offset"] for item in shard_table if item["name"] == tensor.shard)
        tensor_table.append({
            **asdict(tensor),
            "artifact_payload_offset": shard_base + 8 + header_sizes[tensor.shard] + tensor.data_start,
            "artifact_payload_end": shard_base + 8 + header_sizes[tensor.shard] + tensor.data_end,
        })

    manifest = inventory.manifest()
    manifest["conversion"] = {
        "recipe": "qwen38-payload-v1",
        "max_context": context,
        "payload_included": True,
        "physical_execution_status": "pending-baseline-provider-coverage",
        "operation_capabilities": _operation_capabilities(),
        "memory_ledger_bytes": _memory_ledger_for_directory(model_dir, inventory, context),
        "payload_shards": shard_table,
    }
    physical_plan = json.dumps(
        {
            "version": 1,
            "target": "sm_120a",
            "kernel_catalog": "baseline-v1",
            "status": "pending-baseline-provider-coverage",
            "commands": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    write_streaming_artifact(manifest, tensor_table, physical_plan, payload_files, output)
