# S04-P8RQ — Native SM120 NVFP4 integration and quality gate: classification B

**Outcome: classification B (native path is performant; the activation-quantisation mechanism
violates the unchanged local quality contract). P7 is retained as production. No promotion.**

## What was built

- Derived census (`tools/qwen38_nvfp4_census.py`): **401** `nvfp4_linear` launches/token across 8
  classes, validated per command (packed = rows·K/2, scales = rows·K/16) and asserted by the Arm bench.
- Experimental provider `backends/sm120/kernels/experimental/native_nvfp4_provider.h`: advertises
  kernel 27 only for sm_120a with `K % 64 == 0`, `rows % 16 == 0`; otherwise delegates to the retained
  P7 `BaselineProvider`. Selection is at specialization time; no model-name or per-token env branch.
- Runtime `nvfp4_activation_quantize_f32` + `nvfp4_linear_mma_f32` + `launch_nvfp4_linear_mma` (id 27)
  using the existing row-major `.sinf` weight layout. Activation scratch lives in the session
  workspace (allocated once); no hot-path allocation. Recipe held at the pinned block-16 `amax/6`
  E2M1+UE4M3 rule.
- Experiment selector `SUPERINFER_QWEN38_NATIVE_NVFP4` wired into the e2e binary and both fixtures.

## Arm accounting (corrected, complete census)

| quantity | ms/token |
|---|---:|
| `mma_ms_per_token` (natural layout) | 15.10 |
| `activation_quant_ms_per_token` | 2.97 |
| `native_unfused_total_ms_per_token` | 18.07 |
| `repack_cost_once_ms` | 1.00 |
| `native_repacked_unfused_ms_per_token` | 15.10 |

Projection-subsystem throughput 55.3 proj-tok/s. PREDICTION-ONLY whole-model bound with the P7
non-NVFP4 residual (44.3 ms/token): **<= 16.0 tok/s** (16.8 repacked). Arm B (N=8) 496–610 proj-tok/s.

## Gates with the native provider actually active

| Gate | P7 control | Native | Threshold | Result |
|---|---|---|---|---|
| layer-3 differential | max_abs 0.00107, mean 3.56e-5 | **max_abs 3.10462, mean 0.14473** | max<=2e-2, mean<=2e-4 | **FAIL (~155x)** |
| GDN layer-0 / 2 segments | max_abs 3.13e-4, mean 3.08e-6 | **max_abs 6.044, mean 0.04585** | local | **FAIL** |
| D-021 model (margin-qualified) | control not evaluable (see below) | verdict **pass**: 240 strict rows greedy-exact, 12 tie rows, bounds clear, 66 listed outliers | D-021 unchanged | pass |
| same-artifact capture contract | control inconclusive (non-repeatable) | `real_superinfer_discrepancy`: 11 greedy flips | contract | fails |
| native kernel determinism | — | identical layer-3 output over 3 runs | — | deterministic |

Fixture wiring was verified faithful: the baseline provider still reproduces 0.00107 / 3.13e-4 exactly.

## Integrated performance (chat-60, 59 continuation steps, wall clock)

| provider | session 1 | session 2 | session 3 |
|---|---:|---:|---:|
| P7 | 35.60 s | 36.94 s | — |
| native | 32.54 s | 32.65 s | 33.38 s |

Native is **~1.09–1.13x** end-to-end faster; startup/artifact materialisation (~25 s) dominates the
absolute wall time. Decode tok/s is not quotable from these wall times; the microbenchmark's 81–98
proj-tok/s is an Arm number, not model throughput.

## Determinism

- Native MMA/quantisation kernels are deterministic (identical fixture output over 3 runs).
- P7 chat-60 was byte-identical across two fresh sessions; native chat-60 diverged at continuation
  token 3 across three fresh sessions. The D-021 invocation showed the opposite pattern (native
  internally repeatable for all 8 cases, P7 not). The pre-existing whole-model session
  non-determinism is therefore **not resolved**, and the native path's larger numerical error makes
  near-tie decisions more sensitive to it.

## Decision

**B.** The native path is materially faster and the model-level D-021 margin contract held, but the
activation-quantisation mechanism breaks the unchanged, binding layer-3 and GDN gates by two orders of
magnitude. Per the phase rule, promotion stops here.

- Retain P7 as production; do not promote kernel 27.
- Do not loosen D-021 or local thresholds.
- Keep the experimental provider behind `SUPERINFER_QWEN38_NATIVE_NVFP4` for the later
  quantisation-research decision.
- Layerwise evidence to carry forward: layer-3 `max_abs=3.10462` / `mean_abs=0.14473`; GDN
  `max_abs=6.044` / `mean_abs=0.04585`; e2e same-artifact 11 greedy flips concentrated in the
  degenerate-repetition `seeded-long-103` case.
- Next quantisation research (separate decision): a two-level activation scale (block-16 plus a
  per-tensor/global scale) or per-32/64-block granularity, judged against the same unchanged gates.
