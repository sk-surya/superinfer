---
phase: "S04-kernel-portfolio"
plan: "S04-03"
type: "performance"
wave: 2
depends_on: [S04-01, S04-02]
files_modified:
  - backends/sm120/kernels/{ffn,moe}/**
  - backends/sm120/compiler/passes/{ffn,moe,fusion}/**
  - tests/{unit,gpu,bench}/kernels/{ffn,moe}/**
  - benchmarks/manifests/{micro,component}/moe/**
  - artifacts/S04/**
autonomous: false
requirements_addressed: [KER-002, KER-003, KER-004, KER-005, BCK-002, BCK-004, BEN-002]
must_haves:
  truths:
    - "FFN/MoE routing, expert compute and combine are correct across sparse low-token decode cases."
    - "Fusion boundaries and provider selection remain inspectable in the Physical Plan."
    - "The portfolio improves Qwen under controlled component/E2E measurement without hidden regressions."
  artifacts:
    - "Gated FFN and MoE provider families"
    - "Complete Qwen kernel operation matrix"
    - "S04 full-model correctness/performance qualification bundle"
---

# S04-03 — FFN/MoE Portfolio and Full-Model Qualification

## Objective

Complete the Qwen-critical kernel portfolio, focus on sparse/imbalanced decode behavior where applicable to the pinned model, and qualify the combined optimized plan.

## Tasks

1. **Profile FFN/MoE baseline and distributions**
   - Capture actual pinned-model operation shapes and, where MoE is present, expert selection/load distributions over the correctness/benchmark corpora without publishing prompt-sensitive raw content.
   - Separate prefill high-token and decode low-token regimes.
   - Identify routing/permutation, expert compute, activation/gating and combine bottlenecks.

2. **Implement gated FFN candidates**
   - Explore activation/gate multiplication and adjacent projection epilogues/fusions with explicit accumulation and shape envelopes.
   - Retain non-fused baseline and test extreme activation values/boundary widths.

3. **Implement MoE routing and data-movement candidates when required**
   - Optimize logits/top-k, stable tie behavior, expert counts/offset prefix sums, permutation/scatter and combine.
   - Check all indexes/count sums/arena bounds; deterministic mode has stable ordering.
   - Cover zero/one/many tokens per expert and severe imbalance.

4. **Implement grouped expert compute candidates when required**
   - Provide variants for prefill token groups and decode sparse GEMV/grouped workloads.
   - Declare capacity, alignment, workspace and expert-layout requirements.
   - Avoid dispatch loops that allocate or branch on model identity; bounded plan-authored expert work is allowed.

5. **Complete fusion/selector integration**
   - Evaluate combined passes for Qwen graphs and ensure verifier postconditions/memory planner account for lifetimes/workspaces.
   - Generate a complete operation coverage report: baseline, specialized, selected envelope, fallback and evidence ID.
   - Reject candidates that win microbenchmarks but regress declared end-to-end cases.

6. **Qualify the combined optimized artifact**
   - Recompile `.sinf` with pinned provider/tuning catalog.
   - Run all unit/differential/boundary/sanitizer/determinism/S03 acceptance/hot-path gates.
   - Capture controlled diagnostic prefill/decode measurements for S05/S06 baseline; do not publish competitive claims yet.

## Verification

- Complete operation matrix has no unverified specialized entry.
- S03 corpus and state traces pass twice on target hardware.
- Combined plan memory/resource bounds fit declared RTX 5090 configuration.
- Full-model diagnostic improvements are repeatable and raw-data backed.

## Completion Evidence

- Final S04 provider/tuning catalog IDs and optimized artifact hash.
- Complete promotion/operation matrix.
- Full-model correctness, trace, memory and diagnostic performance bundle.
