# S04-P1 Summary — KV-Window Attention Score Caching

**Status: PASS.** Second real profiler -> hypothesis -> implementation -> correctness -> benchmark loop complete.

## Current perf

| Case | R03 (loop 1) | S04-P1 (loop 2) | E2E |
|---|---|---|---|
| chat-60 | 196.5 s | 114.6 s | **1.72×** |
| long-103 | 425.8 s | 176.5 s | **2.41×** |
| decode TPOT (long-context marginal) | ~5.3 s | ~2.4 s | — |

Decode throughput roughly **0.19 -> 0.42 tok/s** on the measured corpus.

## Fresh profile (pre, 60 tokens, 168.5 s GPU)

`grouped_attention_bf16_cache` = 48.8% (82.15 s, 960 launches, 85.6 ms avg).

## Selected target

`grouped_attention_bf16_cache` — `backends/sm120/runtime/cuda_plan_executor.cuh:160`, launched at `:888`.
Chosen because it was both the largest single-kernel share and the largest E2E opportunity; NVFP4
(30.1%) and GDN (14.4%) were explicitly not assumed. See `S04-P1-PROFILE.md`.

## Why this target

Two provable defects: (a) the value pass recomputed the invariant Q·K score inside a per-dimension
loop, making it `O(positions · head_dim²)`; (b) `<<<1,256>>>` used only `query_heads`=24 threads.
(A third defect — `rms_norm_f32_bf16_scale` is `<<<1,1>>>`, single-thread — is real but only 4%.)

## Local before -> after

| Kernel | Before | After | Speedup |
|---|---|---|---|
| `grouped_attention_bf16_cache` / `_cached` | 82.15 s | 0.02 s | **~4,320×** |
| **Total GPU (60 tokens)** | 168.47 s | 86.45 s | **1.95×** |

Post-change ranked table (steady state, 60 tokens): NVFP4 58.8%, GDN 28.1%, RMSNorm 7.9%,
cast_bf16_to_f32 3.1%, `linear_f32` 1.3%, attention 0.0%.

## Qwen tok/s before -> after

~0.19 -> **~0.42 tok/s** (measured corpus; marginal decode ~2.4 s/token at 100-token context).

## Correctness

- Unit differential bit-exact across 5 shapes (`head_dim` 64/128/256; `positions` 1/7/64/257;
  head ratios 24/4, 8/2, 4/1) — `test_grouped_attention_cached_identity`.
- **D-021 verdict pass** on the full 8-case corpus with the optimized binary (240 strict rows
  greedy-exact, 12 ties in-set, 66 listed outliers, bounds clear).
- **Captures byte-identical to R03** for chat-60 and long-103 (`9d588484…`, `b145be58…`), confirming
  the operation-order claim empirically, not just by construction.
- `python tools/validate.py --full` pending final run; Python unit suite 79 passed.

## Design (bit-identical by construction)

New `grouped_attention_bf16_cache_cached`: one block per query head; shared memory holds `positions`
scores + `positions` probabilities; Q·K computed once; max via exact `fmaxf` reduction; softmax
numerators parallel, denominator summed sequentially by thread 0 (preserving summation order); value
accumulation dimension-outer/position-inner exactly as the incumbent. Incumbent retained as fallback
(shared-memory guard falls back when the KV window exceeds the device opt-in limit).

## New #1 bottleneck

`nvfp4_linear_rows_f32` at **58.8%** (50.83 s, 24,060 launches, 2,112 us avg). Estimated ~847 ms/token
for ~16 GB of packed weights is ~90× off the HBM roofline; the decode path calls `ldexpf`-based
`decode_e4m3fn_device` per element and re-decodes the block scale 16× redundantly. This is the next
E2E opportunity, pending the S04-P2 profile.

## Evidence

- `artifacts/S04/s04p1-pre-nsys-profile.json`, `artifacts/S04/s04p1-post-nsys-profile.json`
- nsys reports in `build/evidence/profiler-reports/` (untracked binaries): pre `bbcb686e…`,
  post `17c2262d…`.

## Next action

S04-P2: fresh profile -> confirm NVFP4 is the largest defensible E2E opportunity -> optimize it
(hoist block-scale decode, cheap E4M3/E2M1 decode, vectorized loads) behind a bit-exact differential
and retained fallback. Third real loop.
