# Results-First Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. The repo owner explicitly authorized autonomous execution without waiting for further approval gates. Preserve understanding packets, but do not pause for user answers.

**Goal:** Close Qwen S03 on defensible numerical evidence, capture the first reproducible Qwen RTX 5090 baseline, optimize exactly one profiler-selected decode bottleneck, and reproduce a positive end-to-end speedup before resuming Flash-Next.

**Architecture:** Keep the existing `.sinf`/Physical Plan runtime unchanged unless S03-R proves a real same-artifact discrepancy. Extend the existing independent NVFP4 reference and S03 acceptance tooling to distinguish source-model quantization drift from SuperInfer-specific execution error. After correctness closes, add a minimal internal benchmark/profiling lane, select one measured critical-path target, optimize only that target behind its existing fallback/differential boundary, and verify end-to-end impact.

**Tech Stack:** C++20, CUDA C++ / `sm_120a`, Python 3.12+, PyTorch/Transformers reference tooling, Nsight Systems/Compute or equivalent NVIDIA profiling tools already available on ai90, JSON evidence artifacts.

**Spec:** `.planning/RESULTS-FIRST-RECOVERY-DESIGN.md`

## Global Constraints

- D-006 remains binding: correctness before promotion or performance claims.
- D-019 remains binding: Flash-Next residency/quality assumptions stay blocked pending official evidence.
- Do not substitute token agreement alone for numerical evidence.
- Do not weaken operation/kernel differential tolerances to close S03.
- Do not implement S03F runtime work before the recovery sprint's first performance proof.
- Do not implement the full S04 kernel portfolio before R01 identifies the dominant actionable bottleneck.
- Do not expand full autoresearch before at least one manual profiler -> optimization -> correctness -> end-to-end-speedup loop succeeds.
- Preserve `.sinf` as the deployment contract and keep the executor model-family agnostic.
- Preserve the existing user-owned GPU workload unless the repo owner explicitly changes that constraint; select the available RTX 5090 deterministically and record device identity.
- Preserve all understanding-gate history under D-014; no gate is marked user-passed without user evidence.
- Every completion claim requires fresh verification evidence.

---

### Task 1: Reconcile the canonical plan to results-first ordering

**Files:**
- Modify: `.planning/DECISIONS.md`
- Modify: `.planning/PROJECT.md`
- Modify: `.planning/ROADMAP.md`
- Modify: `.planning/STATE.md`
- Modify only if needed for consistency: `.planning/QUALITY.md`
- Modify only if needed for early internal measurement semantics: `.planning/BENCHMARKS.md`
- Preserve: `.planning/UNDERSTANDING.md`
- Preserve: `.planning/UNDERSTANDING-GATES.md`

**Interfaces:**
- Consumes: approved `.planning/RESULTS-FIRST-RECOVERY-DESIGN.md`
- Produces: one canonical critical path: `S03-R -> R01 -> R02 -> R03 -> broader kernels/autoresearch/Flash-Next`

- [ ] **Step 1: Add D-020 results-first ordering decision**

Append an ADR with these exact semantics:

```markdown
## D-020 — Qwen performance proof precedes Flash-Next implementation

**Status:** Accepted
**Supersedes:** D-017 ordering only
**Decision:** After S03 correctness closes, execute an early Qwen results-first lane consisting of a reproducible baseline/profile, one profiler-selected bottleneck optimization, and a reproduced end-to-end speedup before S03F-02+ Flash-Next runtime implementation. S03F-01 research evidence remains retained. Flash-Next remains a required V0 architecture proof unless separately superseded.
**Why:** The current highest-value uncertainty is whether SuperInfer's hardware-specialization thesis produces measurable end-to-end Qwen improvement. Flash-Next is a broad six-plan architecture expansion and is independently quality-constrained by D-019; placing it before the first Qwen performance feedback loop delays the most decision-useful evidence.
**Consequence:** D-006 correctness remains binding. D-019 remains binding. The first internal Qwen baseline may be captured immediately after S03 closes; broad S04 work and full autoresearch remain deferred until R03 proves one end-to-end optimization loop.
```

- [ ] **Step 2: Rewrite project critical path text**

In `.planning/PROJECT.md`, replace the “two correctness proofs followed by optimization” critical-path description with:

```text
Qwen3.8-27B correctness
  -> S03-R decisive same-artifact closure
  -> R01 reproducible baseline + profile
  -> R02 one measured bottleneck optimization
  -> R03 reproduced end-to-end speedup

then

broader kernel/autoresearch work and Flash-Next architecture proof
  -> S03F-02+ only under D-019 evidence
  -> later model-family/release work
```

Keep V0 requirements for Flash-Next and Gemma intact.

- [ ] **Step 3: Reorder ROADMAP without deleting historical phases**

Keep S00-S03 history. Insert recovery checkpoints S03-R/R01/R02/R03 between S03 and the old S03F/S04 continuation. Mark S03F-01 research complete/retained and S03F-02+ postponed until R03. Do not renumber historical evidence files.

- [ ] **Step 4: Make STATE.md operationally unambiguous**

Set active lane to `S03-R`, next gate to same-artifact decision, and next result gate to `R01 Qwen baseline`. Record that legacy S03-03 diagnostic hypotheses are frozen unless same-artifact evidence identifies a concrete divergence.

- [ ] **Step 5: Verify planning consistency**

Run:

```bash
rg -n "S03F.*before|Flash-Next.*before|two correctness proofs|S03F-02|S04" .planning/PROJECT.md .planning/ROADMAP.md .planning/STATE.md .planning/DECISIONS.md .planning/QUALITY.md .planning/BENCHMARKS.md
python tools/validate.py --full
```

Expected: no surviving statement claims S03F-02+ blocks the first Qwen performance checkpoint; repository validation exits 0.

- [ ] **Step 6: Commit**

```bash
git add .planning/DECISIONS.md .planning/PROJECT.md .planning/ROADMAP.md .planning/STATE.md .planning/QUALITY.md .planning/BENCHMARKS.md
git commit -m "plan: prioritize first Qwen performance proof"
```

---

### Task 2: Add independent same-artifact comparison metrics

**Files:**
- Create: `tools/qwen38_same_artifact_metrics.py`
- Create: `tests/unit/test_qwen38_same_artifact_metrics.py`
- Modify: `tools/qwen38_nvfp4_e2e_reference.py`

**Interfaces:**
- Consumes: FP32 logit rows from the existing streamed NVFP4 reference and SuperInfer capture files.
- Produces: `compare_distribution(reference, candidate, top_k) -> dict[str, float | int | bool]` and per-step JSON-ready metrics including max/mean/RMSE, top-k overlap, greedy agreement, argmax-margin safety, and Jensen-Shannon divergence.

- [ ] **Step 1: Write failing unit tests for distribution metrics**

Create deterministic tests covering identical logits, harmless common shift, a changed non-winning tail logit, and an argmax flip. Tests must assert:

```python
assert metrics["greedy_match"] is True
assert metrics["top_k_overlap"] == 1.0
assert metrics["js_divergence"] == 0.0
assert metrics["candidate_argmax_margin"] > 0.0
assert metrics["max_error_over_margin"] >= 0.0
```

For the argmax-flip fixture, assert `greedy_match is False` and `max_error_over_margin >= 1.0`.

- [ ] **Step 2: Run the focused test and confirm RED**

```bash
python -m pytest tests/unit/test_qwen38_same_artifact_metrics.py -q
```

Expected: fail because `tools.qwen38_same_artifact_metrics` does not yet exist.

- [ ] **Step 3: Implement stable metrics**

Implement with standard-library math only where practical. Compute softmax after subtracting the row max, compute Jensen-Shannon divergence in probability space with a small documented epsilon only for `log`, compute exact top-k index overlap, and record both reference/candidate winner margins. Do not infer an acceptance threshold in this module.

- [ ] **Step 4: Run GREEN and existing Python checks**

```bash
python -m pytest tests/unit/test_qwen38_same_artifact_metrics.py -q
python tools/check_python.py
```

Expected: all pass.

- [ ] **Step 5: Extend the streamed reference to accept `.sinf`-derived packed inputs**

Do not reuse SuperInfer execution kernels. Reuse only artifact parsing/metadata helpers that cannot hide numerical execution behavior. If the current tool cannot directly consume `.sinf`, add an explicit extraction step that proves tensor payload hashes/packing metadata match the artifact used by SuperInfer. Record those hashes in the reference JSON.

- [ ] **Step 6: Commit**

```bash
git add tools/qwen38_same_artifact_metrics.py tests/unit/test_qwen38_same_artifact_metrics.py tools/qwen38_nvfp4_e2e_reference.py
git commit -m "test(S03): add same-artifact distribution oracle"
```

---

### Task 3: Build the bounded S03-R acceptance runner

**Files:**
- Create: `tools/qwen38_s03r_acceptance.py`
- Modify: `tools/qwen38_s03_acceptance.py` only to factor reusable subprocess/capture helpers; preserve old report semantics.
- Create: `tests/unit/test_qwen38_s03r_acceptance.py`
- Create: `tests/corpora/qwen38/results-first-v1.json`
- Evidence output: `artifacts/S03R/`

**Interfaces:**
- Consumes: `.sinf` artifact, existing SuperInfer Qwen acceptance executable, exact same-artifact reference, expanded pinned corpus.
- Produces: `artifacts/S03R/qwen38-same-artifact-acceptance-v1.json` with row-level and aggregate metrics plus a machine-readable decision summary.

- [ ] **Step 1: Add RED tests for report aggregation and decision states**

Test three synthetic reports: `contract_supersede_candidate`, `real_superinfer_discrepancy`, and `inconclusive`. The runner must never silently convert inconclusive evidence into pass.

- [ ] **Step 2: Add the expanded corpus**

Retain all current S03 cases. Add seeded prompt/token sequences that cover short, medium, and longer continuation lengths and at least one regime beyond the previously observed drift onset. Keep the total experiment bounded enough to run repeatedly on ai90. Pin all token IDs and hash the corpus.

- [ ] **Step 3: Implement runner**

For every row/step retain:

```text
greedy_match
top_k_overlap
reference_argmax_margin
candidate_argmax_margin
max_abs
mean_abs
rmse
js_divergence
max_error_over_margin
reference_capture_sha256
candidate_capture_sha256
```

Aggregate quantiles for numerical/distribution metrics. Preserve selected existing layer-boundary traces for cases that enter the long-drift regime.

- [ ] **Step 4: Execute on the qualified RTX 5090 in two fresh sessions**

Use the existing real artifact and executable. Record exact device/environment identity and ensure the other user-owned GPU workload is not disturbed. Save raw logs/captures or hashes sufficient to reproduce the report.

- [ ] **Step 5: Produce exactly one S03-R decision**

`contract_supersede_candidate` is allowed only if same-artifact local/layer differentials stay within their independently justified contracts, greedy output is stable on the expanded corpus, and the remaining material mismatch is between deployment-quantized semantics and the source-model oracle rather than between SuperInfer and same-artifact execution.

`real_superinfer_discrepancy` is required if the same-artifact oracle and SuperInfer materially diverge at a traced boundary or behaviorally unsafe output row.

- [ ] **Step 6: Commit runner/corpus/evidence**

```bash
git add tools/qwen38_s03r_acceptance.py tools/qwen38_s03_acceptance.py tests/unit/test_qwen38_s03r_acceptance.py tests/corpora/qwen38/results-first-v1.json artifacts/S03R
git commit -m "evidence(S03R): run decisive same-artifact acceptance"
```

---

### Task 4: Close S03 by evidence, not by schedule

**Files:**
- Modify: `.planning/phases/S03-qwen38-e2e/S03-03-REVIEW-LATEST.md`
- Create: `.planning/phases/S03-qwen38-e2e/S03-R-SUMMARY.md`
- Modify: `.planning/DECISIONS.md` only for Outcome A
- Modify: `.planning/STATE.md`

**Interfaces:**
- Consumes: Task 3 decision artifact.
- Produces: either a closed S03 with a scoped superseding numerical ADR, or a concrete first-divergence bug with a failing differential test.

- [ ] **Step 1A: If Outcome A, derive the replacement model-level contract from evidence**

Do not choose round thresholds merely because the current run passes. Derive thresholds from same-artifact distributions with explicit safety margin and preserve hard local differential gates. Record corpus scope, quantiles, worst cases, margin behavior, and escalation rule.

- [ ] **Step 2A: Add D-021 quantized Qwen model-level acceptance contract**

D-021 must supersede only the historical full-model source-reference `max_abs <= 0.5` gate, not local kernel/layer contracts. Require deterministic greedy agreement on acceptance prompts plus bounded distributional/model-level metrics and explicit outlier review.

- [ ] **Step 3A: Rerun fresh-session S03 acceptance under D-021 and close only on pass**

Store raw report and hashes. Update `STATE.md` to make R01 active.

- [ ] **Step 1B: If Outcome B, freeze all unrelated diagnostics**

Identify the earliest reproducible same-artifact boundary divergence, add the smallest failing differential test, and invoke systematic debugging against only that boundary. Do not proceed to R01 until fixed and fresh S03-R acceptance passes.

- [ ] **Step 4: Run repository verification**

```bash
python tools/validate.py --full
```

Also run the exact GPU correctness command(s) named by the updated S03-R summary.

- [ ] **Step 5: Commit**

Use either:

```bash
git commit -am "accept(S03): qualify Qwen quantized execution contract"
```

or, for a real implementation fix, a scoped correctness commit followed by a separate evidence/acceptance commit.

---

### Task 5: Capture the first ugly Qwen baseline and profile

**Files:**
- Create: `benchmarks/manifests/qwen38-results-first-v1.json`
- Create: `tools/qwen38_results_first_benchmark.py`
- Create: `tests/unit/test_qwen38_results_first_benchmark.py`
- Evidence output: `benchmarks/runs/R01-*/`
- Create: `.planning/phases/R01-qwen-baseline/R01-SUMMARY.md`

**Interfaces:**
- Consumes: S03-accepted artifact/runtime and fixed benchmark manifest.
- Produces: stable internal baseline with decode TPOT/tok/s, prefill/TTFT, peak VRAM, kernel/region profile, launch/sync indicators, and ranked actionable decode contributors.

- [ ] **Step 1: Add a RED manifest/report-schema test**

Require exact commit, dirty state, artifact hash, CUDA/driver, GPU identity, power/clock policy, prompt/output lengths, concurrency, warmup/sample count, correctness artifact, and timing boundaries.

- [ ] **Step 2: Implement the minimal runner**

Separate cold load/materialization from prefill and steady-state decode. Retain every raw sample. Use GPU events for scoped device timings and monotonic host time for user-visible latency boundaries. Do not add an external baseline dependency.

- [ ] **Step 3: Capture controlled R01 baseline**

Run concurrency=1 on at least two declared prompt lengths and one fixed decode length. Warm to stable clocks/latency. Record median and robust spread, not a single best sample.

- [ ] **Step 4: Capture system/kernel profile**

Use Nsight Systems first to identify launch, synchronization, transfer, and coarse kernel-time structure. Use Nsight Compute only on the top candidate regions where microarchitectural counters will affect the optimization choice. Avoid profiling every kernel indiscriminately.

- [ ] **Step 5: Rank actionable bottlenecks**

The summary must contain a table with `region`, `% decode critical-path time`, `launch count`, `dominant resource symptom`, `expected E2E ceiling if eliminated`, and `actionable now?`. Select exactly one R02 target.

- [ ] **Step 6: Commit baseline evidence**

```bash
git add benchmarks tools/qwen38_results_first_benchmark.py tests/unit/test_qwen38_results_first_benchmark.py .planning/phases/R01-qwen-baseline/R01-SUMMARY.md
git commit -m "bench(R01): capture Qwen results-first baseline"
```

---

### Task 6: Generate and execute a one-target R02 child plan autonomously

**Files:**
- Create: `.planning/phases/R02-first-bottleneck/R02-PLAN.md`
- Modify: only the provider/kernel/runtime files that R01 proves are on the selected critical path.
- Add tests in the corresponding owning `tests/gpu`, `tests/unit`, or `tests/integration` subtree.
- Evidence output: `benchmarks/runs/R02-*/`

**Interfaces:**
- Consumes: ranked R01 profile and selected single target.
- Produces: one optimized candidate behind a retained baseline/fallback and differential gate.

- [ ] **Step 1: Write R02-PLAN from actual profile evidence**

The plan must name exact source files/functions after inspection. It must include a roofline/resource hypothesis, a predicted local speedup, a predicted maximum end-to-end speedup from Amdahl's law, a correctness differential, and a rollback boundary.

- [ ] **Step 2: Do not ask the repo owner for approval**

The repo owner has pre-authorized autonomous continuation within this design. Self-review the child plan, then execute it.

- [ ] **Step 3: Use TDD/differential-first implementation**

Add or strengthen a failing/characterizing differential before replacing the measured path. Keep the incumbent provider selectable until promotion evidence passes.

- [ ] **Step 4: Measure local target before/after under identical shape/config**

Reject improvements that depend on changed semantics, undeclared fallback, or unmatched workload.

- [ ] **Step 5: Run full Qwen correctness before E2E promotion**

Use the S03 accepted contract and existing local kernel/layer differentials.

- [ ] **Step 6: Commit implementation separately from promotion evidence**

Keep optimization code and benchmark/acceptance evidence auditable.

---

### Task 7: R03 reproduced end-to-end performance gate

**Files:**
- Evidence output: `benchmarks/runs/R03-*/`
- Create: `.planning/phases/R03-first-speed-proof/R03-SUMMARY.md`
- Modify: `.planning/STATE.md`
- Modify: `.planning/ROADMAP.md` only to record completed recovery gate / next evidence-driven target.

**Interfaces:**
- Consumes: R01 baseline manifest and promoted R02 candidate.
- Produces: same-manifest before/after report with local and end-to-end deltas reproduced in a second fresh session.

- [ ] **Step 1: Re-run baseline and candidate under the same benchmark manifest**

Do not compare against stale R01 timing if environment drift is material. Record temperature/power/clock validity.

- [ ] **Step 2: Verify correctness first**

If correctness fails, R03 fails regardless of speed.

- [ ] **Step 3: Require positive reproduced E2E decode improvement**

Record local target speedup and end-to-end decode speedup separately. If local speedup is large but E2E gain is negligible, mark the hypothesis locally successful but R03 not satisfied; immediately re-profile instead of stacking more work on the same target.

- [ ] **Step 4: Run a second fresh-session reproduction**

The sign of the end-to-end improvement must reproduce. Report median/spread and environment validity for both runs.

- [ ] **Step 5: Update state by evidence**

If R03 passes, recovery sprint first stopping condition is met. Choose the next lane from fresh profile evidence: another high-value Qwen bottleneck, minimal autoresearch scaffolding based on the proven loop, or resume S03F when D-019 evidence is available. Do not default mechanically to the old phase order.

- [ ] **Step 6: Final verification**

```bash
python tools/validate.py --full
# plus all GPU correctness commands recorded by S03-R and R02
# plus the exact R01/R03 benchmark reproduction commands
```

Check `git status --short`; preserve unrelated/untracked user artifacts.

- [ ] **Step 7: Commit final recovery evidence**

```bash
git add benchmarks/runs/R03-* .planning/phases/R03-first-speed-proof/R03-SUMMARY.md .planning/STATE.md .planning/ROADMAP.md
git commit -m "bench(R03): reproduce first Qwen end-to-end speedup"
```

## Stop Condition

Stop the recovery sprint only when one of these is true:

1. **Success:** S03 is defensibly closed, R01 baseline/profile exists, one R02 target is optimized and correctness-gated, and R03 reproduces a positive end-to-end decode improvement in a second fresh session.
2. **Hard blocker:** same-artifact evidence proves a real correctness discrepancy that cannot be resolved inside the bounded S03-R debugging lane, or hardware/artifact availability prevents measurement. Record the blocker precisely with reproduction evidence; do not substitute additional architecture work.

The agent must report only these state fields at the end of each major gate: `current gate`, `evidence`, `decision`, `blocker if any`, `next action`. Do not pause for user approval.