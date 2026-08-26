"""Dependency-free `.sinf` v1 writer, reader, and inspector."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAGIC = b"SINF"
FORMAT_MAJOR = 1
FORMAT_MINOR = 0
HEADER = struct.Struct("<4sHHIIQQ")
DIRECTORY = struct.Struct("<IIQQQ")
INTEGRITY = struct.Struct("<IIQ")
REQUIRED = 1
SECTION_NAMES = {
    1: "manifest",
    2: "tensor_table",
    3: "physical_plan",
    4: "payload",
    5: "integrity",
}
MAXIMUM_ARTIFACT_BYTES = 1 << 35
STREAMING_INSPECTION_THRESHOLD_BYTES = 1 << 30


@dataclass(frozen=True)
class PhysicalTensorDescriptor:
    """Validated physical view metadata for one artifact tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    alignment: int
    encoding: str
    payload_offset: int
    storage_bytes: int


@dataclass(frozen=True)
class TypedTensor:
    """One bounded tensor materialization and the contract used to consume it."""

    descriptor: PhysicalTensorDescriptor
    data: bytes


class ArtifactError(ValueError):
    """Raised when an artifact fails structural, version, or checksum validation."""


def _checksum(data: bytes) -> int:
    result = 1469598103934665603
    for byte in data:
        result ^= byte
        result = (result * 1099511628211) & ((1 << 64) - 1)
    return result


def _checksum_stream(stream: Any, size: int) -> int:
    result = 1469598103934665603
    remaining = size
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk or len(chunk) > remaining:
            raise ArtifactError("truncated artifact section")
        for byte in chunk:
            result ^= byte
            result = (result * 1099511628211) & ((1 << 64) - 1)
        remaining -= len(chunk)
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _align8(value: int) -> int:
    return (value + 7) & ~7


def write_artifact(source: Mapping[str, Any], path: Path) -> None:
    """Write a normalized source mapping atomically and deterministically."""

    sections = [
        (1, _canonical_json(source["manifest"])),
        (2, _canonical_json(source["tensors"])),
        (3, str(source["physical_plan"]).encode("utf-8")),
        (4, bytes.fromhex(str(source.get("payload_hex", "")))),
    ]
    integrity = b"".join(
        INTEGRITY.pack(kind, 0, _checksum(data)) for kind, data in sections
    )
    sections.append((5, integrity))
    output = bytearray(HEADER.size + len(sections) * DIRECTORY.size)
    records: list[tuple[int, int, int, int, int]] = []
    for kind, data in sections:
        offset = _align8(len(output))
        output.extend(b"\0" * (offset - len(output)))
        section_offset = len(output)
        output.extend(data)
        records.append((kind, REQUIRED, section_offset, len(data), _checksum(data)))
    HEADER.pack_into(output, 0, MAGIC, FORMAT_MAJOR, FORMAT_MINOR, HEADER.size, len(records),
                     HEADER.size, len(output))
    for index, record in enumerate(records):
        DIRECTORY.pack_into(output, HEADER.size + index * DIRECTORY.size, *record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(output)
    os.replace(temporary, path)


def write_streaming_artifact(
    manifest: Mapping[str, Any],
    tensors: Any,
    physical_plan: str,
    payload_files: Sequence[Path],
    path: Path,
) -> None:
    """Write a deterministic artifact while copying payload files in bounded chunks."""

    section_data = [
        _canonical_json(manifest),
        _canonical_json(tensors),
        physical_plan.encode("utf-8"),
    ]
    payload_size = 0
    payload_checksum = 1469598103934665603
    for payload_path in payload_files:
        with payload_path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                payload_size += len(chunk)
                for byte in chunk:
                    payload_checksum ^= byte
                    payload_checksum = (payload_checksum * 1099511628211) & ((1 << 64) - 1)
    section_sizes = [len(value) for value in section_data] + [payload_size]
    checksums = [_checksum(value) for value in section_data] + [payload_checksum]
    integrity = b"".join(INTEGRITY.pack(index + 1, 0, checksum) for index, checksum in enumerate(checksums))
    section_sizes.append(len(integrity))
    directory_end = HEADER.size + len(section_sizes) * DIRECTORY.size
    records: list[tuple[int, int, int, int, int]] = []
    cursor = directory_end
    for index, (size, checksum) in enumerate(zip(section_sizes, checksums + [_checksum(integrity)]), start=1):
        cursor = _align8(cursor)
        records.append((index, REQUIRED, cursor, size, checksum))
        cursor += size
    total_size = cursor
    path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = path.with_name(path.name + ".partial")
    with output_temporary.open("wb") as stream:
        stream.write(b"\0" * directory_end)
        for value in section_data:
            aligned = _align8(stream.tell())
            stream.write(b"\0" * (aligned - stream.tell()))
            stream.write(value)
        aligned = _align8(stream.tell())
        stream.write(b"\0" * (aligned - stream.tell()))
        for payload_path in payload_files:
            with payload_path.open("rb") as payload_stream:
                while True:
                    chunk = payload_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
        aligned = _align8(stream.tell())
        stream.write(b"\0" * (aligned - stream.tell()))
        stream.write(integrity)
        stream.flush()
        if stream.tell() != total_size:
            raise ArtifactError("streaming artifact size accounting mismatch")
        stream.seek(0)
        stream.write(HEADER.pack(MAGIC, FORMAT_MAJOR, FORMAT_MINOR, HEADER.size, len(records),
                                 HEADER.size, total_size))
        stream.seek(HEADER.size)
        for record in records:
            stream.write(DIRECTORY.pack(*record))
    os.replace(output_temporary, path)


def _validated_sections(data: bytes) -> dict[int, bytes]:
    if len(data) < HEADER.size or data[:4] != MAGIC:
        raise ArtifactError("invalid artifact magic or truncated header")
    (
        _magic,
        major,
        minor,
        header_size,
        section_count,
        directory_offset,
        total_size,
    ) = HEADER.unpack_from(data)
    if major != FORMAT_MAJOR or minor > FORMAT_MINOR:
        raise ArtifactError("unsupported artifact format version")
    if (
        header_size != HEADER.size
        or not 0 < section_count <= 1024
        or directory_offset != HEADER.size
    ):
        raise ArtifactError("invalid artifact header bounds")
    if total_size != len(data) or total_size > MAXIMUM_ARTIFACT_BYTES:
        raise ArtifactError("invalid artifact total size")
    directory_end = directory_offset + section_count * DIRECTORY.size
    if directory_end > len(data):
        raise ArtifactError("truncated artifact section directory")
    records: list[tuple[int, int, int, int, int]] = []
    for index in range(section_count):
        record = DIRECTORY.unpack_from(data, directory_offset + index * DIRECTORY.size)
        kind, flags, offset, size, checksum = record
        if kind not in SECTION_NAMES and flags & REQUIRED:
            raise ArtifactError("unknown required section")
        if offset % 8 or offset < directory_end or offset > len(data) or size > len(data) - offset:
            raise ArtifactError("invalid artifact section offset or size")
        if kind in SECTION_NAMES:
            records.append(record)
    ordered = sorted(records, key=lambda record: record[2])
    for previous, current in zip(ordered, ordered[1:]):
        if previous[2] + previous[3] > current[2]:
            raise ArtifactError("artifact sections overlap")
    required_kinds = {1, 2, 3, 4, 5}
    if {record[0] for record in records} != required_kinds or len(records) != len(required_kinds):
        raise ArtifactError("artifact required section is missing or duplicated")
    sections: dict[int, bytes] = {}
    for kind, _flags, offset, size, checksum in records:
        payload = data[offset : offset + size]
        if _checksum(payload) != checksum:
            raise ArtifactError("artifact section checksum mismatch")
        sections[kind] = payload
    expected_integrity = [
        INTEGRITY.unpack_from(sections[5], index * INTEGRITY.size)
        for index in range(len(sections[5]) // INTEGRITY.size)
    ]
    if len(sections[5]) != 4 * INTEGRITY.size:
        raise ArtifactError("artifact integrity table size mismatch")
    for kind, _reserved, checksum in expected_integrity:
        if kind not in sections or kind == 5 or _checksum(sections[kind]) != checksum:
            raise ArtifactError("artifact integrity table mismatch")
    return sections


def inspect_artifact(path: Path) -> dict[str, Any]:
    """Validate an artifact and return a stable machine-readable summary."""

    if path.stat().st_size > STREAMING_INSPECTION_THRESHOLD_BYTES:
        return _inspect_streaming_artifact(path)
    data = path.read_bytes()
    sections = _validated_sections(data)
    manifest = json.loads(sections[1])
    tensors = json.loads(sections[2])
    return {
        "format_version": FORMAT_MAJOR,
        "sections": [SECTION_NAMES[index] for index in (1, 2, 3, 4, 5)],
        "manifest": manifest,
        "tensor_count": len(tensors),
        "payload_bytes": len(sections[4]),
    }


def _read_tensor_table(path: Path) -> tuple[dict[str, Mapping[str, Any]], tuple[int, int, int, int, int]]:
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ArtifactError("truncated artifact header")
        _magic, _major, _minor, _header_size, section_count, directory_offset, _total_size = HEADER.unpack(header)
        stream.seek(directory_offset)
        directory = stream.read(section_count * DIRECTORY.size)
        if len(directory) != section_count * DIRECTORY.size:
            raise ArtifactError("truncated artifact section directory")
        records = {
            record[0]: record
            for record in (
                DIRECTORY.unpack_from(directory, index * DIRECTORY.size)
                for index in range(section_count)
            )
            if record[0] in {2, 4}
        }
        tensor_record = records[2]
        stream.seek(tensor_record[2])
        tensor_table_bytes = stream.read(tensor_record[3])
        tensors = json.loads(tensor_table_bytes)
        if not isinstance(tensors, list):
            raise ArtifactError("artifact tensor table is not a list")
        named = {}
        for tensor in tensors:
            if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
                raise ArtifactError("artifact tensor table record is invalid")
            if tensor["name"] in named:
                raise ArtifactError(f"duplicate artifact tensor: {tensor['name']}")
            named[tensor["name"]] = tensor
        return named, records[4]


def _locate_tensor(path: Path, tensor_name: str) -> tuple[Mapping[str, Any], tuple[int, int, int, int, int]]:
    if not tensor_name:
        raise ArtifactError("tensor name is empty")
    inspect_artifact(path)
    tensors, payload_record = _read_tensor_table(path)
    tensor = tensors.get(tensor_name)
    if tensor is None:
        raise ArtifactError(f"tensor is not present: {tensor_name}")
    if "artifact_payload_offset" not in tensor or "artifact_payload_end" not in tensor:
        raise ArtifactError("tensor payload offsets are absent")
    try:
        start = int(tensor["artifact_payload_offset"])
        end = int(tensor["artifact_payload_end"])
    except (TypeError, ValueError) as error:
        raise ArtifactError("tensor payload offsets are invalid") from error
    if start < 0 or end < start or end > payload_record[3]:
        raise ArtifactError("tensor payload offsets are outside payload")
    return tensor, payload_record


def _read_located_tensor(path: Path, tensor: Mapping[str, Any],
                         payload_record: tuple[int, int, int, int, int]) -> bytes:
    start = int(tensor["artifact_payload_offset"])
    end = int(tensor["artifact_payload_end"])
    with path.open("rb") as stream:
        stream.seek(payload_record[2] + start)
        payload = stream.read(end - start)
    if len(payload) != end - start:
        raise ArtifactError("truncated tensor payload")
    return payload


class ValidatedArtifact:
    """Reusable validated view that checks an artifact once and reads bounded tensor ranges."""

    def __init__(self, path: Path) -> None:
        self.path = path
        inspect_artifact(path)
        self._tensors, self._payload_record = _read_tensor_table(path)

    def _locate(self, tensor_name: str) -> Mapping[str, Any]:
        if not tensor_name:
            raise ArtifactError("tensor name is empty")
        tensor = self._tensors.get(tensor_name)
        if tensor is None:
            raise ArtifactError(f"tensor is not present: {tensor_name}")
        if "artifact_payload_offset" not in tensor or "artifact_payload_end" not in tensor:
            raise ArtifactError("tensor payload offsets are absent")
        try:
            start = int(tensor["artifact_payload_offset"])
            end = int(tensor["artifact_payload_end"])
        except (TypeError, ValueError) as error:
            raise ArtifactError("tensor payload offsets are invalid") from error
        if start < 0 or end < start or end > self._payload_record[3]:
            raise ArtifactError("tensor payload offsets are outside payload")
        return tensor

    def read_tensor_payload(self, tensor_name: str) -> bytes:
        tensor = self._locate(tensor_name)
        return _read_located_tensor(self.path, tensor, self._payload_record)

    def read_typed_tensor(self, tensor_name: str) -> TypedTensor:
        tensor = self._locate(tensor_name)
        return _typed_tensor_from_record(self.path, tensor_name, tensor, self._payload_record)


def read_tensor_payload(path: Path, tensor_name: str) -> bytes:
    """Read one tensor from a validated payload artifact using its relative tensor-table range.

    Validation is performed before the seek, so callers do not consume bytes from a corrupted
    section. Only the requested tensor bytes are materialized; the full payload is never loaded.
    """

    tensor, payload_record = _locate_tensor(path, tensor_name)
    return _read_located_tensor(path, tensor, payload_record)


def _typed_tensor_from_record(path: Path, tensor_name: str, tensor: Mapping[str, Any],
                              payload_record: tuple[int, int, int, int, int]) -> TypedTensor:
    """Materialize one located tensor after validating its physical contract."""

    try:
        source_dtype = str(tensor["dtype"])
        shape_value = tensor["shape"]
        logical_shape = tuple(int(dimension) for dimension in shape_value)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError("tensor physical descriptor is incomplete") from error
    if not isinstance(shape_value, list) or any(dimension <= 0 for dimension in logical_shape):
        raise ArtifactError("tensor physical shape is invalid")
    shape = logical_shape if logical_shape else (1,)
    dtype_map = {
        "F32": ("f32", 4),
        "F16": ("f16", 2),
        "BF16": ("bf16", 2),
        "I8": ("int8", 1),
        "I32": ("int32", 4),
        "U8": ("u8", 1),
        "F8_E4M3": ("u8", 1),
    }
    if source_dtype not in dtype_map:
        raise ArtifactError(f"unsupported tensor dtype: {source_dtype}")
    dtype, bytes_per_element = dtype_map[source_dtype]
    elements = 1
    for dimension in logical_shape:
        elements *= dimension
    encoding = "none"
    expected_bytes = elements * bytes_per_element
    if source_dtype == "U8" and tensor_name.endswith(".weight"):
        if len(logical_shape) != 2 or logical_shape[1] == 0 or logical_shape[1] % 8 != 0:
            raise ArtifactError("packed NVFP4 weight shape is invalid")
        encoding = "nvfp4_packed"
        if logical_shape[1] > ((1 << 64) - 1) // 2:
            raise ArtifactError("packed NVFP4 logical shape overflows")
        shape = (logical_shape[0], logical_shape[1] * 2)
        expected_bytes = elements
    elif source_dtype == "F8_E4M3":
        encoding = "fp8_e4m3_group_scale"
    payload = _read_located_tensor(path, tensor, payload_record)
    if len(payload) != expected_bytes:
        raise ArtifactError("tensor payload bytes do not match physical descriptor")
    declared_contract = {
        "physical_dtype": dtype,
        "layout": "row_major",
        "alignment": 256,
        "storage_encoding": encoding,
        "storage_bytes": len(payload),
        "logical_shape": list(shape),
    }
    for field, expected in declared_contract.items():
        if field in tensor and tensor[field] != expected:
            raise ArtifactError(f"tensor {field} disagrees with its source dtype or payload")
    descriptor = PhysicalTensorDescriptor(
        name=tensor_name,
        dtype=dtype,
        shape=shape,
        layout="row_major",
        alignment=256,
        encoding=encoding,
        payload_offset=int(tensor["artifact_payload_offset"]),
        storage_bytes=len(payload),
    )
    return TypedTensor(descriptor, payload)


def read_typed_tensor(path: Path, tensor_name: str) -> TypedTensor:
    """Materialize one tensor after validating its physical dtype, shape, and encoding.

    The returned bytes are bounded to one tensor. Packed NVFP4 weights retain their logical
    element shape while exposing ``u8`` storage and an explicit packed encoding; callers must
    not reinterpret the bytes as a dense floating-point matrix.
    """

    tensor, payload_record = _locate_tensor(path, tensor_name)
    return _typed_tensor_from_record(path, tensor_name, tensor, payload_record)


def _inspect_streaming_artifact(path: Path) -> dict[str, Any]:
    """Inspect a large artifact without materializing its payload section."""

    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size or header[:4] != MAGIC:
            raise ArtifactError("invalid artifact magic or truncated header")
        (
            _magic,
            major,
            minor,
            header_size,
            section_count,
            directory_offset,
            total_size,
        ) = HEADER.unpack(header)
        file_size = path.stat().st_size
        if major != FORMAT_MAJOR or minor > FORMAT_MINOR:
            raise ArtifactError("unsupported artifact format version")
        if (
            header_size != HEADER.size
            or not 0 < section_count <= 1024
            or directory_offset != HEADER.size
            or total_size != file_size
            or total_size > MAXIMUM_ARTIFACT_BYTES
        ):
            raise ArtifactError("invalid artifact header bounds")
        directory_end = directory_offset + section_count * DIRECTORY.size
        if directory_end > file_size:
            raise ArtifactError("truncated artifact section directory")
        directory = stream.read(section_count * DIRECTORY.size)
        if len(directory) != section_count * DIRECTORY.size:
            raise ArtifactError("truncated artifact section directory")
        records: list[tuple[int, int, int, int, int]] = []
        for index in range(section_count):
            record = DIRECTORY.unpack_from(directory, index * DIRECTORY.size)
            kind, flags, offset, size, checksum = record
            if kind not in SECTION_NAMES and flags & REQUIRED:
                raise ArtifactError("unknown required section")
            if offset % 8 or offset < directory_end or offset > file_size or size > file_size - offset:
                raise ArtifactError("invalid artifact section offset or size")
            if kind in SECTION_NAMES:
                records.append(record)
        ordered = sorted(records, key=lambda record: record[2])
        for previous, current in zip(ordered, ordered[1:]):
            if previous[2] + previous[3] > current[2]:
                raise ArtifactError("artifact sections overlap")
        if {record[0] for record in records} != {1, 2, 3, 4, 5} or len(records) != 5:
            raise ArtifactError("artifact required section is missing or duplicated")

        sections: dict[int, bytes] = {}
        checksums: dict[int, int] = {}
        for kind, _flags, offset, size, checksum in records:
            stream.seek(offset)
            if kind == 4:
                actual = _checksum_stream(stream, size)
            else:
                payload = stream.read(size)
                if len(payload) != size:
                    raise ArtifactError("truncated artifact section")
                sections[kind] = payload
                actual = _checksum(payload)
            if actual != checksum:
                raise ArtifactError("artifact section checksum mismatch")
            checksums[kind] = actual

        integrity = sections[5]
        if len(integrity) != 4 * INTEGRITY.size:
            raise ArtifactError("artifact integrity table size mismatch")
        for index in range(4):
            kind, _reserved, checksum = INTEGRITY.unpack_from(integrity, index * INTEGRITY.size)
            if kind not in checksums or kind == 5 or checksums[kind] != checksum:
                raise ArtifactError("artifact integrity table mismatch")
        manifest = json.loads(sections[1])
        tensors = json.loads(sections[2])
        return {
            "format_version": FORMAT_MAJOR,
            "sections": [SECTION_NAMES[index] for index in (1, 2, 3, 4, 5)],
            "manifest": manifest,
            "tensor_count": len(tensors),
            "payload_bytes": next(record[3] for record in records if record[0] == 4),
        }
