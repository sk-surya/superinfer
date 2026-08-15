# Benchmark and Evidence Protocol

## Purpose

Benchmarks support engineering decisions and public claims. They are valid only when correctness, workload equivalence, environment validity, raw evidence, and reproduction metadata are present.

## Benchmark Classes

1. **Microbenchmarks** — one kernel/fused region over an explicit shape/layout/dtype envelope.
2. **Component benchmarks** — prefill block, decode block, MoE layer, sampling, transfer/materialization.
3. **End-to-end single request** — TTFT and per-token latency for declared prompt/output shapes.
4. **End-to-end throughput** — aggregate tokens/s at declared concurrency/batch with latency percentiles.
5. **Compile/conversion** — converter time, compiler time, tuning time, artifact size, peak host/device memory.

Microbenchmarks never substitute for end-to-end claims.

## Required Run Manifest

Every run pins:

- SuperInfer commit and dirty-state hash;
- `.sinf` artifact hash, source model/revision, quantization and tokenizer identity;
- backend/kernel catalog/tuning database versions;
- GPU SKU, UUID-redacted stable identifier, compute capability, VBIOS when obtainable, driver, CUDA/toolchain;
- power limit, clock policy, persistence mode, temperature at start/end, fan policy when controlled;
- host CPU, RAM, OS/kernel, relevant environment variables, process affinity and competing load note;
- prompt corpus hash, prompt lengths/distribution, output lengths, batch/concurrency, KV state, decode strategy, sampling seed/parameters;
- warmup count, measured sample count, synchronization method, timing source, timeout;
- correctness suite/result and tolerance contract;
- baseline engine commit/version and exact invocation for comparisons.

Schema validation fails before execution if required facts are missing.

## Measurement Rules

- Separate cold artifact load, first prefill, steady-state prefill, and decode.
- TTFT includes the explicitly declared boundaries; report whether tokenization and host-device transfer are included.
- Decode reports TPOT/inter-token latency distribution and tokens/s. Aggregate throughput also reports per-request latency percentiles.
- Warm until clocks/temperature and latency stabilize, then sample; retain every raw sample and rejection reason.
- Use GPU events for scoped device regions and host monotonic time for user-visible latency; do not mix without labeling.
- Avoid device-wide synchronization unless it is part of the measured contract.
- Report median, p5/p95 or robust spread, sample count, and confidence/repeat policy; avoid unsupported precision.
- Re-run winners in a fresh process and, for release claims, a second session.

## Correctness Gate

A benchmark result is ineligible when:

- artifact or output validation fails;
- reference logits/tokens exceed their numerical contract;
- output semantics differ from the baseline comparison;
- the process reports CUDA errors, fallback not declared by the manifest, or unstable nondeterminism;
- temperature/power/clocks violate the run envelope;
- warmup or sample minimum is not met;
- raw evidence is missing or schema-invalid.

## First RTX 5090 Graph

The S06 public graph must prioritize clarity over breadth. Minimum panels:

1. Qwen3.8-27B single-request decode tokens/s (or TPOT) across declared context lengths;
2. prefill tokens/s or TTFT across declared prompt lengths;
3. peak device memory for the same cases;
4. optional kernel-stage breakdown clearly labeled as diagnostic, not end-to-end.

The graph title/caption names artifact quantization, output length, decode strategy, concurrency, GPU power/clock policy, SuperInfer commit, and baseline versions. Correctness status is visible. A generated table accompanies the chart for accessibility and precise values.

## Baseline Comparison

Adapters must print and retain the exact command/config. Match:

- model weights/quantization or clearly label differences;
- tokenizer/chat template and prompt tokens;
- context and requested output token counts;
- greedy/sampling/speculative semantics;
- batch/concurrency and stop conditions;
- inclusion/exclusion boundaries for timing;
- power/clock policy and GPU.

If a baseline cannot match a feature, publish separate results instead of normalizing incompatible numbers into one ranking.

## Evidence Layout

```text
benchmarks/
  manifests/
  baselines/
  runs/<run-id>/
    manifest.json
    environment.json
    correctness.json
    samples.jsonl
    summary.json
    stdout.log
  reports/<report-id>/
    report.json
    table.csv
    graph.svg
    README.md
```

Large raw bundles may be stored in release/object storage, but checksums and a retrieval manifest remain in the repository.

## Regression Policy

- Correctness regression: always blocking.
- Hot-path allocation/new synchronization: always blocking.
- Performance regression: blocking when it exceeds the plan's practical threshold on a stable controlled lane; otherwise flagged with evidence.
- Noise or environment invalidity: rerun, never average away.
- A faster result may not be promoted if it regresses a required workload outside the candidate's declared applicability envelope.

## Reproduction Contract

A clean checkout with documented dependencies must be able to validate the manifest, acquire or point to the exact artifact, run the workload, rebuild the summary, and generate byte-stable tables/graph structure (timestamps and run IDs normalized). Release evidence includes a reproduction transcript and verifier output.
