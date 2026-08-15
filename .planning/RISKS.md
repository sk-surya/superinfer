# Risk Register

| ID | Risk | Likelihood | Impact | Early signal | Mitigation / owner | Triggered response |
|---|---|---:|---:|---|---|---|
| R-01 | Qwen3.8-27B does not fit target VRAM at useful context under chosen representation | Medium | Critical | memory ledger exceeds budget in S03 | S01/S02 owners model weights, KV, workspace before full integration; support explicit quantization/layout policies | narrow context/quantization target transparently; do not hide host offload in benchmark |
| R-02 | CUDA/toolchain lacks or destabilizes intended `sm_120a` features | Medium | High | target probe/compiler failures | S00 pins minimum toolchain; S02 builds capability probes and baseline fallbacks | use supported primitives first; record deferred feature path |
| R-03 | Architecture becomes model-specific despite generic interfaces | High | High | model-name checks or executor edits during S03 | fitness tests; five extension surfaces; Gemma no-executor-change gate | stop and refactor before S07; record superseding decision only if unavoidable |
| R-04 | `.sinf` schema churn makes artifacts unusable | Medium | High | unversioned fields or golden rewrites | compatibility policy, sectioned format, reader/writer matrix | bump format intentionally, provide diagnostic/migration tooling |
| R-05 | Unsafe artifact inputs cause overflow/OOB/device faults | Medium | Critical | parser fuzz crashes, invalid plan reaches allocation | fail-closed CPU validation, checked arithmetic, fuzzing, plan verifier | block release and optimization; add minimized regression fixture |
| R-06 | Megakernel pursuit increases register pressure or harms important shapes | High | Medium | microbenchmark wins but E2E/shape regressions | kernel portfolio and applicability envelopes; end-to-end gate | keep smaller kernels/fallback; promote only scoped winner |
| R-07 | Autoresearch optimizes benchmark loopholes or wrong outputs | High | Critical | speedups correlate with changed tokens/workload/tolerances | immutable benchmark semantics, independent correctness-first gates, patch allowlist/budgets | reject candidate, retain evidence, strengthen schema/test |
| R-08 | RTX 5090 benchmark noise produces false claims | High | High | unstable clocks/thermals, wide variance | dedicated lane, environment capture, invalid-run rejection, repeated sessions | withhold graph/claim until validity and repeat policy pass |
| R-09 | Baseline comparison is semantically mismatched | High | High | differing quantization, sampling, context, or timing boundaries | explicit adapters and audit checklist | publish separate labeled results or omit ranking |
| R-10 | GPU access delays critical acceptance | Medium | High | S02 CPU work complete without target runs | schedule GPU windows early; keep CPU/reference work independent | mark hardware-dependent evidence pending; never simulate performance proof |
| R-11 | External model names/revisions change rapidly | High | Medium | config/tensor drift from assumptions | pin immutable repository revisions and hashes in phase | update frontend fixtures via explicit support revision, not silent tolerance |
| R-12 | Kernel correctness bugs appear only at long context/boundary shapes | High | Critical | divergence after many decode steps | boundary/random/long-context tests, canaries, sanitizer, KV invariant checks | bisect to baseline provider, capture failing seed/artifact |
| R-13 | Python control-plane code leaks into runtime dependency | Medium | High | runtime requires Python to load/execute | `.sinf` deployment contract and dependency tests | block merge; move logic to converter/compiler schema boundary |
| R-14 | Contributor velocity suffers from excessive abstractions | Medium | Medium | simple feature crosses many layers/templates | only five extension surfaces; readable value types; API examples | simplify, consolidate, document cost; reject speculative framework code |
| R-15 | License/provenance prevents redistribution or publication | Medium | High | missing source license/tensor provenance | manifest provenance, dependency/model license inventory | distribute tooling/metadata only where necessary; document acquisition |
| R-16 | DSpark is mis-modeled as attention and couples policy to kernels | Medium | Medium | proposal/rollback code appears in attention provider | DEC-003 architecture test and S05 ownership | move state machine to DecodeStrategy; expose only needed kernel capability |
| R-17 | Hot-path allocations or sync regressions creep in | High | High | latency spikes, trace detects allocation/device sync | trace regression test required for runtime/provider changes | block merge/promotion and restore preallocated plan behavior |
| R-18 | Second model forces a fourth IR or new extension surface | Medium | High | Gemma work cannot map to existing contracts | phase-specific conformance review; allow generic ops/passes/providers | refactor generic semantics; new surface only via explicit superseding ADR |

## Risk Review Cadence

- Review at each phase start and exit.
- Any Critical-impact risk becoming likely blocks downstream performance claims.
- Add evidence links and status in phase summaries; do not silently remove closed risks.
- New public benchmark claims require explicit review of R-07, R-08, R-09, R-15, and R-17.
