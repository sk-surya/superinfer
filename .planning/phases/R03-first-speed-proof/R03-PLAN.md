---
phase: "R03-first-speed-proof"
plan: "R03"
type: "performance-proof"
depends_on: [R02]
autonomous: true
requirements_addressed: [BEN-001, BEN-002, BEN-003, BEN-004, QUA-002, QUA-003]
---

# R03 — First Reproduced End-to-End Qwen Speed Proof

## Objective

Prove that the R02 profiler-selected optimization improves real Qwen3.8-27B steady-state decode end to end, not merely a microbenchmark.

## Preconditions

- R01 baseline/profile is valid and identifies one R02 target.
- R02 implementation retains its correctness fallback/differential path.
- Full Qwen correctness passes under the S03 accepted contract.

## Required comparison

Re-run baseline and optimized candidate under the same benchmark manifest and valid hardware envelope. Report:

- target-region before/after latency or throughput;
- target-region share of decode time before/after;
- end-to-end decode TPOT/tokens/s before/after;
- absolute and percentage E2E delta;
- predicted Amdahl ceiling from R01 versus realized delta;
- peak memory and launch/synchronization/transfer changes;
- correctness result and evidence identity.

Repeat in a second fresh session. The sign of the end-to-end improvement must reproduce.

## Decision

### PASS

R03 passes only when correctness is green and a positive E2E decode improvement reproduces in the second fresh session under a valid environment.

### REPROFILE

If the target microbenchmark/region improves materially but E2E improvement is negligible or non-reproducible, record the local win but do not declare the recovery sprint successful. Re-profile the optimized runtime and select the new dominant critical-path target; do not stack speculative optimizations on the old target.

### FAIL

Any correctness regression, undeclared fallback/transfer, invalid environment, or negative reproduced E2E result blocks promotion.

## Exit evidence

- two fresh-session same-manifest before/after result bundles;
- full correctness evidence;
- `.planning/phases/R03-first-speed-proof/R03-SUMMARY.md` with local and E2E deltas;
- updated `.planning/STATE.md`.

R03 PASS is the first recovery stopping condition and the first direct evidence that SuperInfer's hardware-specialization thesis improves real model execution.