# Benchmark and Evidence Protocol

## Purpose

Benchmarks support engineering decisions and public claims. They are valid only when correctness, workload equivalence, environment validity, raw evidence, placement/residency semantics, and reproduction metadata are present.

S03F is primarily a correctness phase. Its transfer/residency measurements are diagnostic evidence until S06; they are not performance claims.

## Benchmark Classes

1. **Microbenchmarks** — one kernel/fused region over an explicit shape/layout/dtype envelope.
2. **Component benchmarks** — prefill block, decode block, MoE layer, QSA/indexer, PLE gather/prefetch, transfer/materialization.
3. **End-to-end single request** — TTFT and per-token latency for declared prompt/output shapes.
4. **End-to-end throughput** — aggregate tokens/s at declared concurrency/batch with latency percentiles.
5. **Compile/conversion** — converter time, compiler time, tuning time, artifact size, peak host/per-device memory.

Microbenchmarks never substitute for end-to-end claims.

## Required Run Manifest

Every run pins:

- SuperInfer commit and dirty-state hash;
- `.sinf` artifact hash, source model/revision, quantization and tokenizer identity;
- backend/kernel catalog/tuning database versions;
- every GPU SKU/device ordinal, UUID-redacted stable identifier, compute capability, VBIOS when obtainable, driver, CUDA/toolchain;
- GPU topology/peer-access result and declared transfer path for multi-device runs;
- per-device power limit, clock policy, persistence mode, temperature start/end, fan policy when controlled;
- host CPU, RAM, OS/kernel, relevant environment variables, process affinity and competing load note;
- placement plan hash, per-domain residency bytes, host/mmap PLE bytes, device expert bytes and configured workspace/headroom;
- prompt corpus hash, prompt lengths/distribution, output lengths, batch/concurrency, KV/recurrent state, decode strategy, sampling seed/parameters;
- warmup count, measured sample count, synchronization method, timing source, timeout;
- correctness suite/result and tolerance contract;
- baseline engine commit/version and exact invocation for comparisons.

Schema validation fails before execution if required facts are missing.

## Measurement Rules

- Separate cold artifact load, host mapping, device materialization, first prefill, steady-state prefill and decode.
- TTFT includes explicitly declared boundaries; report whether tokenization, PLE lookup and host/device transfers are included.
- Decode reports TPOT/inter-token latency distribution and tokens/s. Aggregate throughput also reports per-request latency percentiles.
- Multi-device runs separately retain cross-device transfer bytes/count/latency and PLE H2D bytes/count; do not fold these away before raw evidence is stored.
- Report per-device peak memory and host-resident mapped bytes, not just aggregate VRAM.
- Warm until clocks/temperature and latency stabilize, then sample; retain every raw sample and rejection reason.
- Use GPU events for scoped device/transfer regions and host monotonic time for user-visible latency; do not mix without labeling.
- Avoid device-wide synchronization unless it is part of the measured contract.
- Report median, p5/p95 or robust spread, sample count and confidence/repeat policy; avoid unsupported precision.
- Re-run winners in a fresh process and, for release claims, a second session.

## Correctness Gate

A benchmark result is ineligible when:

- artifact or output validation fails;
- reference logits/tokens exceed their numerical contract;
- output semantics differ from the baseline comparison;
- device placement/residency differs from the declared manifest;
- an expert-weight, full PLE-table, or cross-device transfer occurs that was not authored in the Physical Plan/accepted policy;
- the process reports CUDA errors, undeclared fallback or unstable nondeterminism;
- temperature/power/clocks violate the run envelope;
- warmup or sample minimum is not met;
- raw evidence is missing or schema-invalid.

## S03F Diagnostic Evidence

Before performance optimization, S03F retains enough physical evidence to prove the architecture is behaving as intended:

- exact packed bytes per host/device category;
- compiler-selected contiguous partition and headroom;
- peer-access probe and actual transfer path;
- activation bytes crossing the GPU boundary;
- PLE sparse-gather and H2D bytes/token;
- expert-weight transfer count during steady-state decode;
- per-device state/workspace allocation;
- selected intermediate differential results.

These measurements validate execution semantics and capacity assumptions. S03F summaries must not label them as wins over another engine.

## First RTX 5090 Graph

The S06 public graph prioritizes clarity over breadth. Minimum Qwen panels:

1. Qwen3.8-27B single-request decode tokens/s (or TPOT) across declared context lengths;
2. prefill tokens/s or TTFT across declared prompt lengths;
3. peak device memory for the same cases;
4. optional kernel-stage breakdown clearly labeled diagnostic.

A Flash-Next dual-GPU report may be added only when baseline semantics, quantization, PLE residency and device topology are sufficiently matched and reproducible. It must show placement/residency facts alongside throughput rather than presenting one aggregate tokens/s number detached from the memory strategy.

The graph title/caption names artifact quantization, output length, decode strategy, concurrency, GPU count/topology, power/clock policy, SuperInfer commit and baseline versions. Correctness status is visible. A generated table accompanies charts for accessibility and precise values.

## Baseline Comparison

Adapters must print and retain the exact command/config. Match or explicitly disclose differences in:

- model weights/quantization;
- PLE/N-gram representation/residency and whether host access is included;
- tokenizer/chat template and prompt tokens;
- context and requested output token counts;
- greedy/sampling/speculative semantics;
- batch/concurrency and stop conditions;
- device count/topology and expert placement/offload policy;
- inclusion/exclusion boundaries for timing;
- power/clock policy.

If a baseline cannot match a feature, publish separate results instead of normalizing incompatible numbers into one ranking.

## Evidence Layout

```text
benchmarks/
  manifests/
  baselines/
  runs/<run-id>/
    manifest.json
    environment.json
    placement.json
    correctness.json
    samples.jsonl
    transfers.jsonl
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
- Hot-path allocation/new implicit synchronization/unplanned transfer: always blocking.
- Placement/residency drift from the declared plan: always blocking until explained/versioned.
- Performance regression: blocking when it exceeds the plan's practical threshold on a stable controlled lane; otherwise flagged with evidence.
- Noise or environment invalidity: rerun, never average away.
- A faster result may not be promoted if it regresses a required workload outside the candidate's declared applicability envelope.

## Reproduction Contract

A clean checkout with documented dependencies must be able to validate the manifest, acquire or point to the exact artifact, reproduce the placement/residency plan, run the workload, rebuild the summary and generate stable report structure. Multi-device evidence must reproduce or explicitly fail topology validation rather than silently changing transfer strategy.
