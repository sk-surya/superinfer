# `.sinf` v1 format

V0 uses a small fixed little-endian binary container rather than a generated schema dependency.
The choice keeps the reader CPU-only, dependency-light, mmap-friendly after validation, easy to
fuzz, and byte-deterministic across C++ and Python implementations. The format is not an ABI
promise; the major version changes on incompatible layout changes.

## Fixed layout

The 32-byte header is `<4sHHIIQQ`:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 4 | ASCII `SINF` magic |
| 4 | 2 | format major, currently 1 |
| 6 | 2 | format minor, currently 0 |
| 8 | 4 | header size, 32 |
| 12 | 4 | section count, bounded to 1024 |
| 16 | 8 | directory offset, currently 32 |
| 24 | 8 | total file size |

Each 32-byte directory entry is `<IIQQQ`: section kind, flags, aligned offset, byte size, and
FNV-1a 64-bit checksum. All section offsets are 8-byte aligned and must be after the directory.
Known required sections are manifest (1), tensor table (2), physical plan (3), payload (4), and
integrity (5). Unknown optional sections may be skipped; unknown required sections fail closed.

The manifest and tensor table are canonical UTF-8 JSON (sorted keys, compact separators). The
physical plan is canonical text in S01 and becomes a structured schema in the following backend
work. Payload bytes are opaque to the parser. The integrity section contains four `<IIQ>` records
mapping each non-integrity kind to its checksum. Readers validate all structure, bounds, overlap,
checksums, required sections, and integrity records before returning read-only views.

The writer builds all bytes first and writes to `path.partial`, flushes/closes it, then atomically
renames it to the requested path. No timestamp, pointer, path, or process identity enters the
deterministic content. FNV-1a is an integrity/checksum mechanism, not a cryptographic signature;
the reserved signature/attestation slot remains a future compatibility extension.

