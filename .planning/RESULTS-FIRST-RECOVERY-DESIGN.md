# Results-First Recovery Sprint Design

**Status:** Approved in chat; written-spec review pending
**Date:** 2026-09-09
**Target branch:** `sol/results-first-recovery`
**Base:** `work/ultraplan-v0` @ `cc638254903877c4e26d374f714beede1ee8350a`

## Objective

Change the V0 critical path so SuperInfer produces decision-useful Qwen performance evidence as soon as correctness is defensible. The immediate milestone is not completion of a fixed number of plans. It is:

> Qwen3.8-27B generates correctly under an independently justified quantized-model contract, current end-to-end performance is measured reproducibly, and one profiler-selected optimization produces a reproduced end-to-end speed improvement without correctness regression.

## Why the current ordering must change

The current roadmap requires:

`S03 Qwen correctness -> S03F Flash-Next -> S04 kernels -> S05 autoresearch -> S06 performance proof`

That ordering optimizes for architecture breadth before validating the project's central performance thesis. S03F is a six-plan expansion into multi-device placement, PLE, MoE, QSA, gated residuals, and dual-GPU execution. It is also independently constrained by D-019 because official Flash-Next artifact/reference evidence is not yet sufficient for quality qualification.

For the current objective, Flash-Next is therefore not on the shortest path to the next high-information result.

## New critical path

The recovery sequence is:

1. **S03-R — decisive Qwen correctness closure experiment**
2. **R01 — current Qwen end-to-end baseline and profiler decomposition**
3. **R02 — optimize exactly one profiler-selected dominant decode bottleneck**
4. **R03 — reproduce end-to-end improvement and decide whether to continue kernel work or re-rank bottlenecks**
5. **Then** resume broader kernel portfolio/autoresearch and Flash-Next architecture work under the reordered roadmap

Flash-Next and full autoresearch machinery are removed from the path to the first Qwen speed result.

## S03-R — decisive same-artifact correctness experiment

### Problem statement

Current S03 evidence is deterministic and token-correct on the acceptance corpus, while two longer cases fail the historical model-level `max_abs <= 0.5` logit threshold. Extensive localization has ruled out many candidate implementation defects and points instead to deterministic accumulated cross-layer numerical amplification.

The remaining question is no longer “which precision tweak should we try next?” It is:

> Does the current SuperInfer execution materially disagree with an independent implementation operating on the exact same packed artifact semantics, or is the historical source-model `max_abs <= 0.5` gate an inappropriate model-level equivalence criterion for this quantized path?

### Required oracle

Build an independent **same-artifact oracle** that consumes the actual `.sinf` packed tensors / quantization metadata and reproduces the declared deployment storage semantics without reusing SuperInfer execution kernels as its own oracle.

The oracle must be independent enough that agreement cannot be explained by common implementation code for the arithmetic under test. Project-local parsing/utilities may be shared only where they do not duplicate the numerical execution path being validated.

### Corpus

Retain the existing S03 acceptance corpus and expand it enough to characterize the error distribution rather than adjudicate correctness from a few rows. At minimum include:

- current short/plain case;
- Unicode/special-token cases;
- current chat-template continuation;
- current varied-length continuation;
- additional seeded prompts spanning materially different prompt/output lengths;
- at least one continuation long enough to expose the previously observed accumulated-drift regime;
- fresh-process repeatability for selected cases.

The exact corpus size is chosen by the executor to keep the experiment bounded, but it must be large enough to estimate tail behavior and top-token stability rather than merely reproduce the known failures.

### Measurements

Compare SuperInfer, the same-artifact oracle, and where useful the pinned external/source reference. Record per row/step:

- greedy argmax token equality;
- top-k overlap for a fixed documented `k`;
- argmax logit margin and whether numerical error could plausibly flip the winner;
- max absolute logit error;
- mean absolute logit error;
- RMSE;
- probability-space divergence using a documented stable metric such as KL or Jensen-Shannon on a bounded/top-k-normalized support;
- selected intermediate hidden-state errors at already-instrumented boundaries sufficient to detect a genuine execution discrepancy;
- repeatability hashes.

Do not replace numerical evidence with token agreement alone.

### Decision rule

S03-R has exactly two valid outcomes.

#### Outcome A — historical contract superseded

If the same-artifact oracle agrees with SuperInfer within independently derived operation/layer tolerances, preserves deterministic greedy behavior across the expanded corpus, and the remaining source-reference discrepancy is attributable to the quantized/deployment arithmetic contract rather than a SuperInfer-specific defect, author a new ADR that supersedes the historical model-level `max_abs <= 0.5` criterion.

The replacement contract must be evidence-derived and scoped. It must include:

- required greedy-token agreement for deterministic acceptance prompts;
- operation/layer numerical tolerances where those are diagnostically meaningful;
- model-level distributional metrics and thresholds derived from the observed same-artifact error distribution;
- argmax-margin safety evidence;
- explicit treatment of outliers and failure escalation;
- no weakening of kernel/local differential gates merely to pass the model-level test.

Then rerun fresh-session S03 acceptance under the superseding contract and close S03 only if it passes.

#### Outcome B — real SuperInfer discrepancy found

If SuperInfer materially disagrees with the independent same-artifact oracle, do not change the numerical contract. Localize the first boundary where the implementations diverge, write a minimal failing differential, and fix that root cause before S03 closes.

No further speculative precision/rounding experiments are permitted unless they are downstream of a demonstrated same-artifact divergence.

## R01 — ugly baseline first

Immediately after S03 closes, benchmark the current unoptimized Qwen3.8-27B path before broad kernel work.

### Required results

For declared prompt/output shapes and concurrency=1, retain at minimum:

- decode tokens/s and TPOT;
- prefill tokens/s and TTFT;
- peak device memory;
- artifact load/materialization time separately from steady-state inference;
- per-kernel or per-region GPU-time decomposition;
- bytes moved / effective bandwidth where measurable;
- launch count and synchronization indicators;
- top five contributors to steady-state decode GPU time;
- exact commit, artifact hash, GPU topology, clocks/power policy, CUDA/driver, prompt corpus, runtime configuration, and correctness result.

The baseline is allowed to be slow. Its purpose is to reveal the optimization regime.

### Baseline comparison

If an external engine baseline can be run with matched-enough semantics, record it. Do not block R01 on obtaining a perfectly matched external comparison. The first mandatory comparison is SuperInfer-before versus SuperInfer-after under identical semantics.

## R02 — optimize exactly one bottleneck

Select the target mechanically from R01 evidence.

Default selection criterion:

> Choose the single region/kernel family with the largest actionable share of steady-state decode time, unless measurement shows that optimizing it cannot materially affect end-to-end latency because of overlap, serialization, transfer, launch, or another critical-path constraint.

Examples may include NVFP4 projection/GEMV, attention/KV, GDN/FFN, launch overhead, memory movement, or another measured region. No target is selected in advance.

### Scope rule

R02 may change only what is required to improve the selected bottleneck and its immediate integration path. Do not implement the entire S04 portfolio.

Every optimized candidate must retain the independent differential path required by D-006. The incumbent baseline remains available as a fallback until promotion evidence passes.

## R03 — first performance proof gate

R02 is not considered successful because a microbenchmark gets faster.

Promotion requires fresh-process evidence showing:

1. the selected kernel/region improves materially on its target workload;
2. full Qwen correctness remains within the accepted S03 contract;
3. end-to-end decode performance improves under the same benchmark manifest;
4. the improvement reproduces in a second run/session;
5. no undeclared fallback, transfer, synchronization, or memory regression invalidates the comparison.

Record both the local speedup and the end-to-end speedup.

If the local speedup is substantial but end-to-end gain is negligible, do not stack more optimizations on that hypothesis. Re-profile and select the new critical bottleneck.

## Roadmap and ADR changes required after written-spec approval

The implementation plan must update canonical planning state so it no longer claims Flash-Next blocks the first Qwen performance work.

Required planning changes:

- add an ADR superseding **D-017 ordering only**;
- preserve D-017's architectural rationale and Flash-Next scope, but move S03F off the path to the first Qwen performance checkpoint;
- preserve D-019 and its evidence requirements;
- update `ROADMAP.md` with S03-R / R01 / R02 / R03 or equivalent named recovery phases/checkpoints;
- update `STATE.md` so the active lane is S03-R and the next gate is the Qwen baseline;
- update `PROJECT.md` milestone wording if necessary so “results-first” ordering is not contradicted by the project summary;
- update `BENCHMARKS.md` only where needed to permit the early internal Qwen baseline/performance checkpoint while retaining stricter public-claim/release requirements;
- update `QUALITY.md` only if required by the new evidence-derived S03 contract process;
- keep understanding-gate history intact under D-014; do not mark any historical gate user-passed.

## Autoresearch ordering

Do not build or expand the full autoresearch/promotion machinery before the first manual profiler -> optimization -> correctness -> end-to-end-speedup loop has succeeded.

After one or two manual optimization cycles expose the real experiment shape, use that evidence to finalize the autoresearch schema and promotion objective. This avoids automating an incorrect or low-value search objective.

## Flash-Next ordering

Flash-Next remains a required V0 architecture proof unless separately superseded later. It is postponed, not abandoned.

Resume S03F implementation after the first Qwen performance proof unless new evidence makes another ordering clearly higher-value. D-019 remains binding: no expert residency/paging/quality assumptions without the required official artifact/reference evidence.

## Invariants

The recovery sprint must preserve these invariants:

- correctness before promotion;
- no performance claim from a numerically invalid run;
- independent oracle for optimized paths;
- no token-agreement-only correctness substitution;
- no silent loosening of local kernel/operation tolerances;
- benchmark manifests and raw evidence remain reproducible;
- no hidden expert paging/offload or undeclared transfers;
- no model-name branching added to the generic executor;
- no understanding gate marked passed for the user;
- Flash-Next architecture work remains tracked rather than deleted.

## Non-goals

This recovery sprint does **not** attempt to:

- finish the complete S04 kernel portfolio;
- implement Flash-Next runtime support;
- implement full autoresearch;
- optimize batch/continuous serving;
- add arbitrary distributed execution;
- publish comparative performance claims before the existing benchmark/release evidence requirements are satisfied;
- weaken correctness merely to obtain a graph.

## Acceptance criteria

The recovery sprint reaches its first stopping point only when all of the following evidence exists:

1. S03 is closed through either a root-cause fix or an independently reviewed superseding quantized-model contract;
2. a reproducible current Qwen baseline exists on the qualified RTX 5090;
3. profiler evidence identifies the dominant actionable decode bottleneck;
4. one targeted optimization is implemented behind a correct fallback/promotion boundary;
5. correctness passes after that optimization;
6. a second-session benchmark reproduces a positive end-to-end decode improvement;
7. the report states both local and end-to-end deltas and retains raw evidence.

At that point the project has its first direct evidence that SuperInfer's hardware-specialization thesis can improve real model execution, and the next phase is chosen from fresh profiler evidence rather than the old phase count.