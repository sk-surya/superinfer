# S04-P3 Summary — Value-Dimension-Parallel Gated DeltaNet

**Status: PASS.** Fourth real profiler -> hypothesis -> implementation -> correctness -> benchmark loop complete.

## Current perf

| Case | R03 | P1 | P2 | P3 | vs R03 |
|---|---|---|---|---|---|
| chat-60 | 196.5 s | 114.6 s | 73.0 s | 49.2 s | 3.99x |
| long-103 | 425.8 s | 176.5 s | 105.3 s | 64.1 s | 6.64x |
| marginal decode | 5.33 s/tok | 1.44 | 0.75 | **0.347 s/tok** | 15.4x |
| decode throughput | 0.19 tok/s | 0.69 | 1.33 | **~2.9 tok/s** | 15.4x |

## Fresh profile (pre, post-P2, 60 tokens, 44.90 s GPU)

`gated_delta_attention_f32` = 54.1% (24.28 s, 2,880 launches, 8,431 us avg), launched `<<<1,256>>>` with only 48 active threads.

## Local before -> after

| Kernel | Before | After | Speedup |
|---|---|---|---|
| `gated_delta_attention_f32` -> `_parallel_f32` | 24.28 s | 0.19 s | **128x** |
| **Total GPU (60 tokens)** | 44.90 s | 20.78 s | **2.16x** |

## Correctness

- Unit differential bit-exact over state AND output across {4/48/128/128 pos 1}, {.., pos 3}, {2/8/64/96 pos 2}.
- **D-021 verdict pass** on the full 8-case corpus (240 strict rows exact, 12 ties in-set).
- Captures **byte-identical to S04-P2** (chat-60 `9d588484…`, long-103 `b145be58…`).
- Second E2E session reproduces: 49.93 s / 64.93 s vs 49.23 s / 64.09 s (<=1.4%).
- Python suite 79 passed.

## New #1 bottleneck (post-P3 profile, 20.78 s GPU)

| Kernel | share | launches | avg |
|---|---:|---:|---:|
| `nvfp4_linear_rows_vec_f32` | 44.6% | 24,060 | 385 us |
| `rms_norm_f32_bf16_scale` | **32.7%** | 12,540 | 542 us |
| `cast_bf16_to_f32` | 13.1% | 45,240 | 60 us |
| `linear_f32` | 5.3% | 5,760 | 192 us |
| `gated_delta_attention_parallel_f32` | 0.9% | 2,880 | 68 us |

Selected next: `rms_norm_f32_bf16_scale` (S04-P4).
