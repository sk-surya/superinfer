---
phase: "S03-qwen38-e2e"
plan: "S03-02"
type: "feature"
wave: 2
depends_on: [S03-01, S02-02]
files_modified:
  - backends/sm120/compiler/passes/**
  - backends/sm120/kernels/baseline/**
  - include/superinfer/decode/**
  - src/decode/**
  - python/superinfer/convert/**
  - tests/{integration,gpu,golden}/qwen38/**
autonomous: false
requirements_addressed: [MOD-001, MOD-003, DEC-001, KER-001, KER-005, KER-006, FMT-003, BCK-004]
must_haves:
  truths:
    - "The complete Qwen graph lowers through generic passes/providers and fits the declared RTX 5090 envelope."
    - "Greedy decode executes from a prebuilt strategy/state plan with correct KV updates."
    - "Conversion is deterministic and runtime remains model-blind/allocation-free in steady state."
  artifacts:
    - "Qwen-specific target compilation pipeline configuration"
    - "Complete baseline provider coverage and explicit KV-cache layout"
    - "Deterministic Qwen `.sinf` artifact recipe and memory ledger"
---

# S03-02 — Qwen Lowering, Baseline Coverage, and `.sinf` Integration

## Objective

Compile the full pinned Qwen Semantic IR into an executable baseline Physical Plan and deterministic `.sinf` that fits the declared RTX 5090 workload.

## Tasks

1. **Close generic operation/lowering coverage**
   - Map every Qwen Semantic IR operation to target-aware Lowered IR and a baseline provider capability.
   - Implement only generic passes/providers; parameterize by operation attributes, shapes, layouts and target profile.
   - Maintain an operation matrix from semantic node -> pass -> lowered op -> kernel capability -> reference test.

2. **Select and validate initial storage/quantization policy**
   - Produce a full memory ledger for weights, scales/metadata, maximum KV context, activations, workspaces and safety margin.
   - Choose the smallest explicit, testable V0 configuration that fits one RTX 5090 and preserves acceptable reference agreement.
   - Record quantization algorithms/layouts/scales in `.sinf`; compare conversion statistics and layer-level error to the trusted reference.

3. **Implement explicit Qwen KV-cache plan**
   - Define logical-to-physical KV axes, dtype/quantization, alignment, capacity, append/update, local/sliding behavior if required by pinned semantics, bounds and reset.
   - Lower attention/provider capabilities against this layout and version it in the Physical Plan.
   - Test multi-step updates, max boundary, reset/reuse, invalid position and rollback-ready invariants.

4. **Implement greedy DecodeStrategy baseline**
   - Declare logits/sampling commands, token/state buffers, stop/max-token handling and KV transition requirements at compile time.
   - Bind all resources at session construction and keep steady decode allocation-free.
   - Add sampling strategy shell only if needed for reference tests; speculative behavior remains deferred.

5. **Integrate deterministic conversion**
   - Stream source shards through validated mapping/StoragePolicy without unbounded host copies.
   - Compile target plan using pinned target profile/provider catalog and write provenance/quantization/tensor hashes.
   - Support resumable staging only if atomic final artifact/determinism remain provable.

6. **Create full-graph staged differential harness**
   - Compare selected embeddings/layer boundaries/router outputs/attention/FFN/final logits for small prompt cases.
   - Localize first divergence and retain bounded tensors/statistics without leaking full model weights.
   - Use dtype/quantization-specific numerical contracts approved before tests are run.

## Verification

- Two clean conversions of pinned inputs produce byte-identical `.sinf` hashes.
- Inspector/validator reports source/provenance, memory ledger, KV layout and complete plan validity.
- Full-graph small prompt matches approved intermediate/final error contracts.
- Runtime trace after warmup contains no forbidden hot-path events or Qwen model identifiers.

## Completion Evidence

- Conversion manifest/command, artifact hash and size.
- RTX 5090 memory ledger with declared max workload and margin.
- Operation coverage matrix and staged differential report.
