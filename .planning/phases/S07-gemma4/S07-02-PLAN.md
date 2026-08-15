---
phase: "S07-gemma4"
plan: "S07-02"
type: "feature"
wave: 2
depends_on: [S07-01]
files_modified:
  - include/superinfer/{ir,compiler,kernels}/**
  - src/{ir,compiler}/**
  - backends/sm120/{compiler,kernels}/**
  - python/superinfer/convert/**
  - tests/{unit,gpu,golden,integration}/gemma4/**
autonomous: false
requirements_addressed: [MOD-005, ARCH-005, ARCH-006, KER-002, FMT-003, FMT-004, QUA-004]
must_haves:
  truths:
    - "All Gemma needs are implemented through generic passes/providers/storage/strategy contracts."
    - "Provider selection remains capability-based and Qwen behavior stays compatible."
    - "The resulting `.sinf` fits and validates for the declared RTX 5090 scope."
  artifacts:
    - "Generic IR/pass/provider additions with Qwen regression coverage"
    - "Deterministic Gemma `.sinf` conversion and memory ledger"
    - "Updated format/capability compatibility fixtures if required"
---

# S07-02 — Generic Lowering/Provider Gaps and Gemma Artifact

## Objective

Close the generic compiler/provider/storage gaps revealed by Gemma and produce a validated, deterministic artifact without touching the physical executor.

## Tasks

1. **Implement approved generic semantic/lowering additions**
   - Add operation attributes/verifiers/reference math from S07-01 gap decisions.
   - Add deterministic GraphPass/lowering behavior with explicit preconditions/postconditions/invalidation.
   - Update Semantic/Lowered/Physical dumps and compatibility versions intentionally.

2. **Extend provider capabilities, not model branches**
   - Add baseline correct candidates for new operation/layout/dtype/shape needs.
   - Reuse S04 providers where capability envelopes apply; add measured/tuned variants only if needed for feasible execution.
   - Test selector positive/negative/fallback paths using generic fixtures and both model shapes.

3. **Resolve storage/quantization and memory envelope**
   - Produce full Gemma weight/KV/activation/workspace ledger against target capacity.
   - Choose explicit initial quantization/context policy; record numerical effects and manifest fields.
   - Add StoragePolicy behavior only if general and versioned.

4. **Compile deterministic Gemma artifact**
   - Integrate source streaming/tensor mapping, compiler pipeline, strategy, provider catalog and `.sinf` provenance.
   - Validate complete artifact/plan/target capabilities before materialization.
   - Convert twice from pinned inputs and compare full artifact bytes/hashes.

5. **Maintain compatibility and Qwen regression**
   - Update reader/writer/capability compatibility fixtures for intentional schema changes.
   - Rebuild/run the pinned Qwen artifact/correctness subset and confirm unchanged behavior or documented compatible rebuild.
   - Hash/diff executor sources/dependencies and fail on any change.

## Verification

- Generic operation/provider differential and boundary suites pass.
- Gemma `.sinf` validates, fits declared memory scope and is byte-reproducible.
- Qwen correctness/compatibility subset passes.
- Executor hash/dependency test proves no model-specific or any executor edits.

## Completion Evidence

- Gemma operation coverage matrix and memory ledger.
- Artifact recipe/hash/manifest and compatibility report.
- Qwen regression plus executor unchanged evidence.
