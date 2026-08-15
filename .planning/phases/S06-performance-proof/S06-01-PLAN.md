---
phase: "S06-performance-proof"
plan: "S06-01"
type: "feature"
wave: 1
depends_on: [S05-02, S05-03]
files_modified:
  - schemas/benchmarks/**
  - python/superinfer/bench/**
  - benchmarks/{manifests,baselines}/**
  - tests/{unit,integration}/bench/**
  - .github/workflows/benchmark.yml
  - .planning/{UNDERSTANDING.md,STATE.md}
autonomous: false
understanding_gate: "D-entry"
requirements_addressed: [BEN-001, BEN-002, BEN-003, BEN-004, QUA-002, QUA-003, GOV-001, GOV-002, GOV-004]
must_haves:
  truths:
    - "The runner refuses incomplete, semantically mismatched, incorrect or environmentally invalid runs."
    - "Prefill, decode and load/compile boundaries are explicit in raw samples."
    - "Baseline invocations are pinned, inspectable and matched where compared."
  artifacts:
    - "Versioned benchmark/environment/correctness/sample schemas"
    - "Controlled RTX 5090 runner and validity preflight"
    - "Pinned SuperInfer and baseline manifests/adapters"
---

# S06-01 — Controlled Benchmark Runner and Baseline Adapters

## Objective

Make benchmark execution a schema-validated, correctness-gated measurement process that can reproduce equivalent SuperInfer and baseline workloads.

## Gate D Entry Contract

Read the Gate D packet and current ledger before implementation. Benchmark mechanics may proceed within the recorded window, but this plan must not silently cross a later L2 boundary when Gate D or an older gate would make debt exceed one. Emit the phase-transition `UNDERSTANDING STATUS` with the concrete proposal-width experiment and blocked boundary.

## Tasks

1. **Implement benchmark/evidence schemas**
   - Encode every required manifest/environment/correctness/sample/summary field from `.planning/BENCHMARKS.md` with compatibility/version rules.
   - Represent timing boundaries, prompt corpus hash, token counts, concurrency, sampling/speculation, artifact/model identity and baseline equivalence explicitly.
   - Add typed invalidity/rejection reasons and immutable raw record hashes.

2. **Implement RTX 5090 environment preflight**
   - Verify GPU/driver/toolchain identity, exclusive process policy, power/clock settings, temperature window, memory availability and absence of prior CUDA errors.
   - Sample environment during warmup/measurement and invalidate runs that leave the envelope.
   - Redact machine-private identifiers while preserving stable reproducibility facts.

3. **Implement workload runner and timing boundaries**
   - Load a pinned prompt-token corpus, validate tokenizer/artifact identity and execute declared warmup/samples with timeouts.
   - Capture host monotonic user-visible timings and device-event component timings without accidental synchronization.
   - Separate cold load, materialization, prefill/TTFT, first decode, steady decode and total output.
   - Store every sample and rejection; compute no final comparison in the runner.

4. **Integrate correctness and hot-path gates**
   - Run/attach the S03 acceptance subset and artifact validation before eligible measurement.
   - Assert selected output tokens/state and no forbidden trace events/fallbacks.
   - Bind evidence by hash to the exact artifact, plan/catalog, workload and source commit.

5. **Create baseline adapter contract**
   - Each adapter installs/resolves a pinned engine version, converts/loads the declared model configuration, prints exact invocation and emits normalized raw events without hiding native raw output.
   - Add equivalence audit for weights/quantization, prompt IDs, sampling/speculation, lengths/concurrency, timing boundaries and GPU policy.
   - Mark unmatched dimensions and prevent them from entering a matched comparison table.

6. **Create the first benchmark grid**
   - Choose a small, useful prompt/context grid for single-request prefill and decode with fixed output length and declared strategy.
   - Estimate total run time and repetition budget; include second-session confirmation.
   - Version the manifest; do not edit it after data collection begins—create a new version instead.

7. **Add controlled benchmark workflow**
   - Require explicit/manual or scheduled trigger on an exclusive runner, environment preflight, timeout and immutable artifact upload.
   - Prevent ordinary PR code from producing public claims without review/approval.

## Verification

- Schema/runner tests reject every missing/mismatched/invalid field class.
- Synthetic timing source tests verify boundaries and no double counting.
- Known thermal drift, fallback, incorrect token and baseline mismatch fixtures are ineligible.
- Dry run and two RTX 5090 sessions produce complete raw bundles for the pinned grid.

## Completion Evidence

- Validated run manifests and adapter equivalence reports.
- Two raw target run bundles plus intentionally rejected fixture bundles.
- Exact clean-run operator command and expected duration/resources.
