---
phase: "S01-artifact-ir"
plan: "S01-03"
type: "feature"
wave: 3
depends_on: [S01-02]
files_modified:
  - schemas/sinf/**
  - include/superinfer/artifact/**
  - src/artifact/**
  - python/superinfer/{convert,inspect}/**
  - tests/{unit,property,golden,fuzz,integration}/artifact/**
  - docs/sinf-format.md
autonomous: true
requirements_addressed: [FMT-001, FMT-002, FMT-003, FMT-004, FMT-005, FMT-006]
must_haves:
  truths:
    - "A `.sinf` is deterministic, inspectable, integrity-checked, and validated before allocation."
    - "Storage details terminate at StoragePolicy/device views, not kernels."
    - "Corrupt or incompatible artifacts fail closed with useful diagnostics."
  artifacts:
    - "Published V0 `.sinf` binary/schema specification"
    - "Atomic writer, defensive reader, manifest and section integrity implementation"
    - "CPU converter shell plus inspect/validate CLI and fuzz corpus"
---

# S01-03 — `.sinf` Artifact, StoragePolicy, and Tooling

## Objective

Turn a verified synthetic Physical Plan and tensor inventory into a deterministic deployment artifact and make it safely inspectable without a GPU.

## Tasks

1. **Resolve and document the format encoding**
   - Compare candidate schema encodings using deterministic bytes, forwards compatibility, checked/zero-copy access, build impact, inspectability, and fuzz support.
   - Record the selection and compatibility rules in `.planning/DECISIONS.md`.
   - Specify fixed header, endianness, section directory, alignments, required/optional flags, manifest, tensors, plan, tokenizer/config metadata, integrity table, and reserved/signature behavior.

2. **Implement manifest and section schemas**
   - Include source identity/revision/licenses, converter/compiler/toolchain IDs, target fingerprint, quantization, pass pipeline, tokenizer/template hashes, tensor hashes, and format/schema versions.
   - Canonicalize maps/order/floats/text and explicitly exclude timestamps or normalize them outside reproducible content.
   - Define stable capability IDs and required/optional negotiation.

3. **Implement atomic deterministic writer**
   - Precompute checked offsets/sizes/alignment, stream payloads, hash contents, fsync/close as appropriate, and atomically replace final output.
   - Never leave a partially written file at the requested artifact path.
   - Produce identical bytes from identical synthetic inputs in separate processes.

4. **Implement defensive reader and validator**
   - Validate magic/version, total size, section count, checked arithmetic, alignment, overlap/alias policy, hashes, required capabilities, manifest schema, tensor metadata, and embedded Physical Plan before storage materialization.
   - Bound allocations, recursion/nesting, strings, sections, tensors, and commands.
   - Expose zero-copy/mmap-friendly read-only views only after full structural validation.

5. **Implement StoragePolicy baseline**
   - Provide aligned contiguous and memory-mapped host policies with explicit owners and typed tensor views.
   - Define device materialization plan contract for S02 without calling CUDA here.
   - Test mapping lifetime, unaligned/corrupt payload rejection, and duplicate/alias rules.

6. **Create converter shell and inspect/validate CLI**
   - Python converter accepts a normalized synthetic source inventory/frontend output, compiler options, target profile file, and output path; it invokes versioned schema/binding boundaries.
   - Inspector prints canonical summary/sections/tensors/plan resources/provenance and supports machine-readable JSON.
   - Validator returns stable exit classes for corruption, incompatibility, semantic/plan failure, and I/O.

7. **Build golden, negative, fuzz and reproducibility suites**
   - Golden minimal/dense/MoE artifacts and manifests.
   - Generate systematic truncation/corruption/offset-overflow/unknown-capability/checksum fixtures.
   - Add parser fuzz target and bounded CI smoke corpus.
   - Run conversion twice in isolated directories and compare full hashes/bytes.

## Verification

- All artifact tests pass under sanitizers and parser fuzz-smoke.
- `superinfer inspect` and `validate` operate with CUDA disabled.
- A corrupt artifact never reaches StoragePolicy materialization.
- Format docs reconstruct every byte/compatibility rule and match golden fixtures.

## Completion Evidence

- `.sinf` V0 format document and schema compatibility table.
- Golden artifact hashes and deterministic rebuild transcript.
- Negative/fuzz corpus summary and stable CLI JSON examples.
