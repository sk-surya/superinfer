---
phase: "S04-kernel-portfolio"
plan: "S04-02"
type: "performance"
wave: 1
depends_on: [S03-03]
files_modified:
  - backends/sm120/kernels/{attention,kv}/**
  - backends/sm120/compiler/passes/attention/**
  - tests/{unit,gpu,bench}/kernels/{attention,kv}/**
  - benchmarks/manifests/{micro,component}/attention/**
autonomous: false
requirements_addressed: [KER-002, KER-003, KER-004, KER-005, KER-006, BCK-002, BCK-004, BEN-002]
must_haves:
  truths:
    - "Prefill and decode attention have separate capability-selected portfolios."
    - "KV layout/update is versioned, bounds-safe, and verified over long sequences."
    - "Attention optimization does not absorb speculative decode policy."
  artifacts:
    - "Prefill/decode attention and KV-update provider families"
    - "KV layout compatibility/invariant suite"
    - "Attention selection and promotion evidence"
---

# S04-02 — Attention and KV-Cache Portfolio

## Objective

Optimize prefill and iterative decode attention as separate workload families while preserving the explicit KV contract and long-context correctness.

## Tasks

1. **Characterize attention workloads**
   - Generate manifests over pinned Qwen head/group dimensions, prompt lengths, decode context lengths, batch/concurrency-one V0 scope, and declared KV dtype/layout.
   - Profile baseline memory traffic, reductions, occupancy, register/shared-memory use and end-to-end share.
   - Separate prefill, first decode and steady decode conclusions.

2. **Implement prefill attention candidates**
   - Explore bounded tiling/online-softmax/staging variants appropriate to semantic attention/local masks.
   - Declare mask/causal/local/GQA/layout/dtype/alignment envelopes and stable accumulation behavior.
   - Test odd/boundary sequence lengths, fully/partially masked tiles, and large logits.

3. **Implement decode attention/KV update candidates**
   - Fuse KV append/update with attention only when state visibility and rollback contract remain explicit.
   - Explore head grouping, split/reduction strategy, persistent/cache-aware scheduling, and context-length regimes.
   - Precompute launch/scratch choices at compile time; no runtime tuning/model switch.

4. **Harden KV layout compatibility**
   - Version layout in Physical Plan/provider capability and reject mismatches.
   - Test address mapping, capacity boundary, reset/reuse, long sequence, local-window behavior if present, and future rollback checkpoints.
   - Add sentinel/canary and sanitizer cases around every arena boundary.

5. **Integrate deterministic selection and fallback**
   - Select distinct candidates by operation/layout/dtype/shape/context bucket authored by the Physical Plan.
   - Ensure out-of-envelope cases choose the correct baseline or fail before launch.
   - Dump selection rationale and measured tuning identity.

6. **Promote with full-model gates**
   - Run kernel differentials against independent attention reference across random/boundary/model shapes.
   - Run S03 corpus including long-context state hashes and hot-path trace.
   - Compare component and end-to-end diagnostic impact under stable environment.

## Verification

- Attention/KV differential matrix and sanitizer runs pass.
- Long-context Qwen outputs/state hashes remain within approved contract.
- Speculative/proposal/acceptance logic does not appear in attention providers.
- Selection never depends on model name and always has an explicit fallback/out-of-range result.

## Completion Evidence

- Prefill/decode capability and measured envelope tables.
- KV layout/version/invariant specification.
- Promoted/rejected candidate records with raw evidence.
