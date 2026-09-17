# S04-P2 Summary — Vectorized, Scale-Hoisted NVFP4 Projection

**Status: PASS.** Third real profiler -> hypothesis -> implementation -> correctness -> benchmark loop complete.

## Current perf

| Case | R03 | S04-P1 | S04-P2 | vs R03 |
|---|---|---|---|---|
| chat-60 | 196.5 s | 114.6 s | 73.0 s | 2.69x |
| long-103 | 425.8 s | 176.5 s | 105.3 s | 4.04x |
| marginal decode (slope) | 5.33 s/tok | 1.44 s/tok | **0.75 s/tok** | 7.1x |
| decode throughput | 0.19 tok/s | 0.69 tok/s | **~1.33 tok/s** | 7.1x |

Cumulative across loops 1–3: **0.032 -> ~1.33 tok/s (~42x)**.

## Fresh profile (pre, post-P1, 60 tokens, 86.45 s GPU)

`nvfp4_linear_rows_f32` = 58.8% (50.83 s). Measured traffic 14.4 GB/token vs roofline 8.0 ms but ~847 ms/token => ~17 GB/s effective, ~106x off peak.

## Why this target

Largest single-kernel share after the attention fix; the gap to roofline is instruction/latency bound
(per-element `ldexpf` FP8 decode; scalar 1-byte packed loads), not roofline bound.

## Local before -> after

| Kernel | Before | After | Speedup |
|---|---|---|---|
| `nvfp4_linear_rows_f32` -> `_vec_f32` | 50.83 s | 9.27 s | **5.48x** |
| **Total GPU (60 tokens)** | 86.45 s | 44.90 s | **1.93x** |

## Correctness

- Unit differential bit-exact: scalar incumbent vs vectorized across aligned + tail shapes (`test_nvfp4_row_parallel_identity`).
- **D-021 verdict pass** on the full 8-case corpus (240 strict rows exact, 12 ties in-set, 66 listed outliers).
- Captures **byte-identical to S04-P1** (chat-60 `9d588484…`, long-103 `b145be58…`), so all three loops are output-identical to each other.
- Second E2E session reproduces: 73.39 s / 105.57 s vs 73.02 s / 105.31 s (<=0.5%).
- Python suite 79 passed; `validate.py --full` pending final run.

## New #1 bottleneck (post-P2 profile, 44.90 s GPU)

| Kernel | share | launches | avg |
|---|---:|---:|---:|
| `gated_delta_attention_f32` | **54.1%** | 2,880 | 8,431 us |
| `nvfp4_linear_rows_vec_f32` | 20.7% | 24,060 | 385 us |
| `rms_norm_f32_bf16_scale` | 15.2% | 12,540 | 544 us |
| `cast_bf16_to_f32` | 6.0% | 45,240 | 60 us |
| `linear_f32` | 2.5% | 5,760 | 192 us |

GDN and RMSNorm are both single-block/single-thread and have large local headroom; NVFP4 still has a
~19x gap to its remaining roofline. Next loop target pending the S04-P3 profile.

## Evidence

- `artifacts/S04/s04p2-post-nsys-profile.json`; nsys report `build/evidence/profiler-reports/s04p2-nsys-chat60.nsys-rep` (untracked binary).
- `benchmarks/runs/S04-P2/`; D-021 verdict JSON.
