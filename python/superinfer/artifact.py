"""Dependency-free `.sinf` v1 writer, reader, and inspector."""

from __future__ import annotations

import json
import os
import struct
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


def read_tensor_payload(path: Path, tensor_name: str) -> bytes:
    """Read one tensor from a validated payload artifact using its relative tensor-table range.

    Validation is performed before the seek, so callers do not consume bytes from a corrupted
    section. Only the requested tensor bytes are materialized; the full payload is never loaded.
    """

    if not tensor_name:
        raise ArtifactError("tensor name is empty")
    inspect_artifact(path)
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
        tensor = next((item for item in tensors if item.get("name") == tensor_name), None)
        if tensor is None:
            raise ArtifactError(f"tensor is not present: {tensor_name}")
        if "artifact_payload_offset" not in tensor or "artifact_payload_end" not in tensor:
            raise ArtifactError("tensor payload offsets are absent")
        try:
            start = int(tensor["artifact_payload_offset"])
            end = int(tensor["artifact_payload_end"])
        except (TypeError, ValueError) as error:
            raise ArtifactError("tensor payload offsets are invalid") from error
        payload_size = records[4][3]
        if start < 0 or end < start or end > payload_size:
            raise ArtifactError("tensor payload offsets are outside payload")
        stream.seek(records[4][2] + start)
        payload = stream.read(end - start)
        if len(payload) != end - start:
            raise ArtifactError("truncated tensor payload")
        return payload


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
