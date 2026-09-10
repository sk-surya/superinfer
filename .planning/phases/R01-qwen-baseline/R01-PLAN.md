---
phase: "R01-qwen-baseline"
plan: "R01"
type: "internal-performance-baseline"
depends_on: [S03-R]
autonomous: true
requirements_addressed: [BEN-001, BEN-002, BEN-003, QUA-002, QUA-003]
---

# R01 — First Qwen Baseline and Critical-Path Profile

## Objective

Measure the current accepted Qwen3.8-27B SuperInfer path before broad optimization and identify exactly one dominant actionable steady-state decode bottleneck.

## Preconditions

- S03 is closed under either the corrected historical contract or an independently justified D-021 contract.
- The exact accepted `.sinf` artifact and correctness evidence are pinned.
- The measured RTX 5090 is not shared with another workload during accepted samples.

## Required measurements

For fixed, declared concurrency=1 workloads retain:

- decode TPOT and tokens/s;
- prefill time/tokens/s and TTFT boundary;
- cold artifact load/materialization separately;
- peak device memory;
- raw sample distributions and warmup policy;
- kernel/region GPU time and launch counts;
- synchronization/transfer indicators;
- exact commit/artifact/corpus/device/CUDA/driver/power/clock state.

Use Nsight Systems to identify coarse critical-path structure. Use Nsight Compute only on the leading actionable candidate(s) where counters can discriminate an optimization hypothesis.

## Selection rule

Rank regions by critical-path contribution. Select exactly one R02 target: the largest actionable contributor unless overlap/serialization/launch/transfer evidence shows that reducing it cannot materially change end-to-end decode latency.

The R01 summary must state the selected target, its measured share, an Amdahl upper bound on E2E improvement if eliminated, and the specific source/provider path to inspect for R02.

## Non-goals

- no full S04 portfolio;
- no Flash-Next work;
- no full autoresearch;
- no public comparative claim required;
- no optimization during baseline capture.

## Exit evidence

- versioned benchmark manifest;
- raw baseline samples;
- profiler captures or stable summarized evidence plus commands;
- ranked bottleneck table;
- exactly one R02 target;
- `.planning/phases/R01-qwen-baseline/R01-SUMMARY.md`.

After R01, write the exact-file/function R02 child plan from measured evidence and execute it autonomously.