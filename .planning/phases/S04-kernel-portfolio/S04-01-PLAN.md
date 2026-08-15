---
phase: "S04-kernel-portfolio"
plan: "S04-01"
type: "performance"
wave: 1
depends_on: [S03-03]
files_modified:
  - backends/sm120/kernels/{linear,norm,rope,embedding,sampling}/**
  - backends/sm120/compiler/passes/fusion/**
  - tests/{unit,gpu,bench}/kernels/**
  - benchmarks/manifests/micro/**
autonomous: false
requirements_addressed: [KER-002, KER-003, KER-004, KER-005, BCK-002, BCK-004, BEN-002]
must_haves:
  truths:
    - "Core dense/support kernels have explicit capability and numerical envelopes."
    - "Fusions are compiler decisions materialized into the plan, not runtime branches."
    - "Winners improve representative regions and do not regress Qwen correctness."
  artifacts:
    - "Linear/norm/RoPE/embedding/LM-head/sampling provider families"
    - "Differential and boundary matrix plus micro/component results"
    - "Promotion records and selector/fallback rules"
---

# S04-01 — Dense and Support Kernel Portfolio

## Objective

Build measured `sm120` candidates for dense projections and ubiquitous support operations, including only fusions that improve the declared Qwen workloads.

## Tasks

1. **Baseline and profile representative regions**
   - Create immutable micro/component manifests for actual Qwen shapes across prefill/decode context regimes.
   - Capture baseline latency, bandwidth/compute estimates, occupancy/resource metrics, end-to-end share and correctness.
   - Form a written hypothesis and applicability envelope before each kernel family.

2. **Implement dense GEMM/GEMV candidate families**
   - Cover declared storage/accumulation dtypes, aligned Qwen projection shapes and low-M decode regimes.
   - Explore tiling, persistent scheduling, weight layout and epilogue fusion in bounded variants.
   - Declare alignment, shape divisibility, workspace, determinism and numerical contract; keep baseline fallback.

3. **Implement normalization/residual and RoPE candidates**
   - Add RMSNorm and proven residual-norm fusions with stable reductions and explicit accumulation.
   - Add RoPE candidates matching exact semantic axes/positions and boundary positions.
   - Evaluate fusion with adjacent regions only through a GraphPass with declared postconditions.

4. **Implement embedding, LM-head and sampling candidates**
   - Optimize lookup/projection/top-k/greedy primitives needed by baseline DecodeStrategy.
   - Avoid unnecessary full-logit materialization only when plan semantics and correctness are preserved.
   - Cover vocabulary boundaries, ties, NaNs/infinities per numerical contract and deterministic selection.

5. **Integrate selector and target lowering**
   - Feed candidate capabilities and measured tuning records into deterministic compile-time selection.
   - Explain chosen/rejected candidates in plan dumps.
   - Test applicable/inapplicable/fallback paths without model-name switches.

6. **Promote only verified winners**
   - Run randomized/boundary/model-shape differential tests, sanitizer where supported, repeatability, micro and component measurement.
   - Re-run full Qwen acceptance and hot-path trace.
   - Record candidate patch, manifests, raw results and scoped promotion decision.

## Verification

- Every registered capability has positive/negative selector and differential cases.
- Full S03 correctness corpus passes unchanged.
- No new steady-state allocation/sync/model branching.
- Report performance separately for micro, component and end-to-end diagnostic effect.

## Completion Evidence

- Kernel registry matrix and plan selection explanation.
- One promotion/rejection record per candidate family.
- Before/after raw samples for declared Qwen regions.
