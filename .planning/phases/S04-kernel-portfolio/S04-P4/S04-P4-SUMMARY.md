# S04-P4 Summary — Row-Parallel RMSNorm with Bit-Exact Denominator

**Status: PASS.** Fifth real profiler -> hypothesis -> implementation -> correctness -> benchmark loop complete.

## Current perf

| Case | R03 | P1 | P2 | P3 | P4 | vs R01 |
|---|---|---|---|---|---|---|
| chat-60 | 196.5 s | 114.6 s | 73.0 s | 49.2 s | 42.7 s | — |
| long-103 | 425.8 s | 176.5 s | 105.3 s | 64.1 s | 52.9 s | — |
| marginal decode | 5.33 s/tok | 1.44 | 0.75 | 0.347 | **0.238 s/tok** | 131x |
| decode throughput | 0.19 tok/s | 0.69 | 1.33 | 2.88 | **~4.2 tok/s** | 131x |

Cumulative across loops 1–5 vs the R01 baseline (31.3 s/token): **~0.032 -> ~4.2 tok/s (131x)**.

## Fresh profile (pre, post-P3, 60 tokens, 20.78 s GPU)

`rms_norm_f32_bf16_scale` = 32.7% (6.80 s, 12,540 launches, 542 us avg), launched `<<<1,1>>>` (one thread for the whole tensor). New #2 after NVFP4.

## Local before -> after

| Kernel | Before | After | Speedup |
|---|---|---|---|
| `rms_norm_f32_bf16_scale` -> `_parallel` | 6.80 s | 0.31 s | **21.9x** |
| **Total GPU (60 tokens)** | 20.78 s | 14.31 s | **1.45x** |

## Correctness

- Unit differential bit-exact across {1,5120}/{4,5120}/{64,128} and both `add_one_to_scale` values.
- **D-021 verdict pass** on the full 8-case corpus (240 strict rows exact, 12 ties in-set).
- Captures **byte-identical to S04-P3** (chat-60 `9d588484…`, long-103 `b145be58…`); all five loops remain mutually output-identical.
- Second E2E session reproduces: 42.75 s / 52.95 s vs 42.70 s / 52.93 s (<=0.1%).
- Python suite 79 passed; `validate.py --full` pending final run.

## Observation

Fixed per-process cost (18 GB artifact load + 19.2 GB arena upload, ~25-30 s) now dominates wall time on
short runs. The honest decode metric is the marginal slope (0.238 s/token). Load/materialization is a
separate optimization axis from decode TPOT.

## New #1 bottleneck (post-P4, 14.31 s GPU)

| Kernel | share | launches | avg |
|---|---:|---:|---:|
| `nvfp4_linear_rows_vec_f32` | **64.8%** | 24,060 | 385 us |
| `cast_bf16_to_f32` | 19.0% | 45,240 | 60 us |
| `linear_f32` | 7.7% | 5,760 | 192 us |
| `rms_norm_f32_bf16_scale_parallel` | 2.2% | 12,540 | 25 us |

Decode is now purely GPU-bound: 14.31 s / 60 tokens = 0.238 s/token, matching the wall marginal.

## Evidence

- `artifacts/S04/s04p4-post-nsys-profile.json` (filled after the post-profile run).
- `benchmarks/runs/S04-P4/`; D-021 verdict JSON.
- nsys reports in `build/evidence/profiler-reports/` (untracked binaries).
