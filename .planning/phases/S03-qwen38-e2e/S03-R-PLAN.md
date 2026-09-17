---
phase: "S03-qwen38-e2e"
plan: "S03-R"
type: "correctness-recovery"
depends_on: [S03-03]
autonomous: true
requirements_addressed: [MOD-001, MOD-004, DEC-001, KER-006, BCK-004, QUA-002, QUA-003]
---

# S03-R — Decisive Same-Artifact Correctness Closure

## Objective

End open-ended S03 numerical archaeology. Determine whether the remaining long-run logit drift is a SuperInfer-specific execution defect or a mismatch between the historical source-reference threshold and the actual quantized deployment arithmetic.

## Inputs

- `.planning/RESULTS-FIRST-RECOVERY-DESIGN.md`
- `.planning/RESULTS-FIRST-RECOVERY-PLAN.md`, Tasks 2–4
- `.planning/phases/S03-qwen38-e2e/S03-03-REVIEW-LATEST.md`
- `tools/qwen38_nvfp4_e2e_reference.py`
- `tools/qwen38_s03_acceptance.py`
- existing S03 artifact, acceptance corpus, layer traces, and deployment-v8 evidence

## Required experiment

Construct an independent same-artifact oracle over the exact `.sinf` packed tensor semantics. Compare that oracle with SuperInfer and, secondarily, the pinned source/reference model across the existing corpus plus bounded additional long-drift coverage.

Record greedy agreement, top-k overlap, argmax margins, max/mean/RMSE, Jensen-Shannon divergence, error-to-margin ratio, selected intermediate boundary errors, repeatability hashes, exact artifact identity, and environment identity.

## Decision

Exactly one outcome is permitted:

### A — contract supersession candidate

Use only when the same-artifact oracle and SuperInfer agree under independently justified local/layer contracts and behavioral stability, while the remaining material error is attributable to source/reference versus quantized deployment semantics. Derive and review a scoped D-021 model-level quantized acceptance contract; do not weaken local kernel/layer gates. Rerun fresh-session acceptance. Close S03 only on that evidence.

### B — real SuperInfer discrepancy

Use when same-artifact execution and SuperInfer materially diverge. Preserve the historical contract, identify the first reproducible divergence boundary, add the smallest failing differential, and debug only that boundary until fixed. No unrelated precision/rounding hypothesis work is allowed.

“Inconclusive” is a valid experiment result but not a closure result; if evidence is inconclusive, design one discriminating experiment rather than returning to broad hypothesis search.

## Exit evidence

- machine-readable `artifacts/S03R/qwen38-same-artifact-acceptance-v1.json` or a versioned successor;
- same-artifact provenance/tensor identity evidence;
- two fresh-session executions for the closure result;
- `S03-R-SUMMARY.md` stating Outcome A or B and why;
- if A, D-021 plus fresh acceptance under it;
- if B, an automated failing differential at the first proven divergence.

## Stop rule

Once S03 closes, immediately enter R01 baseline/profile. Do not start Flash-Next or broad kernel work.