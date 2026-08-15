# S01: Artifact and IR — Context

**Status:** Planned
**Depends on:** S00
**Critical-path role:** Creates the deterministic compilation/deployment contract used by every model and backend.

<domain>
## Phase Boundary

Implement canonical Semantic IR, target-aware Lowered IR, immutable Physical Plan schema, verified pass manager, `.sinf` format/reader/writer, StoragePolicy, and CPU converter/inspection tooling. Target model-specific import and device execution remain out of scope; use synthetic fixtures.
</domain>

<decisions>
## Locked Decisions

- [D-002] Exactly three durable representations.
- [D-005] `.sinf` is versioned, sectioned, checksummed, inspectable, and fully validated before allocation.
- [D-010] provenance and evidence metadata are first-class.
- Same pinned inputs/configuration/tool build produce byte-identical output.
- Unknown required capability fails closed; unknown optional section can be skipped safely.
- Artifact parser and verifier are CPU-only and fuzzable.
- Python converter orchestration does not define model semantics; frontends own that contract.

### Executor Discretion

- Choose a serialization/schema technology after measuring determinism, mmap/zero-copy ergonomics, forwards compatibility, code size, and fuzzability.
- Choose canonical JSON/CBOR/binary encoding for manifests so long as byte reproducibility and schema validation hold.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — three representations and logical `.sinf` sections.
- `.planning/REQUIREMENTS.md` — ARCH-001–003/007, FMT-001–006, MOD-002/003.
- `.planning/QUALITY.md` — parser, golden, property, fuzz, and compatibility tests.
- `.planning/DECISIONS.md` — D-002, D-005, D-010, D-012.
</canonical_refs>

<deferred>
## Deferred

Qwen/Gemma tensor names, device code, tuned kernel identifiers, real tokenizer integration, and GPU materialization are deferred to S02/S03/S07.
</deferred>
