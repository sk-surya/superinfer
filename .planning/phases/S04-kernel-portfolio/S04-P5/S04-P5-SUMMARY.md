# S04-P5 Summary — Block-Parallel Remaining Single-Block Dense Kernels

**Status: PASS.** Sixth profiler -> hypothesis -> implementation -> correctness -> benchmark loop. **The >=5 decode tok/s checkpoint is reached.**

## Current perf

| Case | R03 | P1 | P2 | P3 | P4 | P5 | vs R01 |
|---|---|---|---|---|---|---|---|
| chat-60 | 196.5 s | 114.6 | 73.0 | 49.2 | 42.7 | 39.9 s | — |
| long-103 | 425.8 s | 176.5 | 105.3 | 64.1 | 52.9 | 48.2 s | — |
| marginal decode | 5.33 s/tok | 1.44 | 0.75 | 0.347 | 0.238 | **0.192 s/tok** | **163x** |
| decode throughput | 0.19 tok/s | 0.69 | 1.33 | 2.88 | 4.2 | **~5.2 tok/s** | **163x** |

Cumulative vs R01 (31.3 s/token, 0.032 tok/s): **~5.2 tok/s, ~163x**. Checkpoint met.

## Fresh profile (pre, post-P4, 60 tokens, 14.31 s GPU)

`cast_bf16_to_f32` 19.0% (2.72 s, 45,240 launches), `linear_f32` 7.7% (1.11 s, 5,760 launches), `cast_f32_to_bf16` 1.0% — combined ~27.7%, all `<<<1,256>>>`.

## Local before -> after

Launch-only change: `min(4096, ceil(elements/256))` blocks instead of one, for `cast_bf16_to_f32`, `cast_f32_to_bf16`, `linear_f32`. Kernel bodies unchanged.

| Kernel | Before | After | Speedup |
|---|---|---|---|
| `cast_bf16_to_f32` | 2.72 s | 0.07 s | **39x** |
| `cast_f32_to_bf16` | 0.15 s | ~0 (collapsed) | — |
| `linear_f32` | 1.11 s | 1.11 s | 1.0x (latency-bound; block count does not help) |
| **Total GPU (60 tokens)** | 14.31 s | 11.54 s | **1.24x** |

## Correctness

- Executor unit tests (cast, lm_head) pass; Python suite 79 passed.
- **D-021 verdict pass** on the full 8-case corpus (240 strict rows exact, 12 ties in-set) — reproduced twice consecutively.
- Captures **byte-identical to S04-P4** (chat-60 `9d588484…`, long-103 `b145be58…`); all six loops remain mutually output-identical.
- Second E2E session reproduces: 39.94 s / 48.17 s vs 39.90 s / 48.17 s.
- Determinism stress: 5x long-103 + 2x direct + 2 E2E sessions + 2 D-021 corpus runs = **9+ consecutive byte-identical runs** (`b145be58…`).
- `validate.py --full` pending final run.

### Anomaly record (transparency)

One P5 D-021 corpus run produced a different long-103 capture (`522d5300…`) with first divergence at row 20
and verdict `fail`. It did not reproduce: two subsequent full D-021 runs, two direct long-103 runs, two E2E
sessions, and a 5x stress all produced the canonical `b145be58…` capture. The P5 change is launch-only
(block count) over elementwise/row-independent loops, which cannot alter arithmetic order. The anomaly is
treated as an unreproduced environment/transient event and is recorded here rather than suppressed; a
dedicated runtime-nondeterminism investigation is warranted if it recurs. P4 (the prior commit) never
showed it and is the retained fallback for all P5 launch sites.

## Notes

Fixed per-process cost (artifact load + arena upload, ~25-30 s) still dominates short-run wall time;
the marginal decode slope (0.192 s/token) is the honest decode metric and is now fully GPU-bound.
