# R03 Summary — First Reproduced End-to-End Qwen Speed Proof

**Status: PASS. Recovery sprint first stopping condition met.**

## Result

| Case | R01 baseline | R03-A (optimized) | R03-B (fresh repro) | E2E speedup |
|---|---|---|---|---|
| special-3 (3 rows) | 116.3 s | 32.5 s | 33.1 s | 3.6× (load-dominated) |
| chat-60 (60 rows) | 1843.2 s | 196.5 s | 197.1 s | 9.4× |
| long-103 (103 rows) | 3252.5 s | 425.8 s | 425.8 s | 7.6× |

- **Local target speedup:** `nvfp4_linear_f32` 84.62 s → `nvfp4_linear_rows_f32` 2.54 s = **33×** (R02 predicted 15–30×).
- **End-to-end decode speedup:** slope 31.3 s/token → ~5.3 s/token (**~6×** on marginal cost; **7.6–9.4×** wall including fixed load). Decode now 0.19 tok/s, TPOT ~5.3 s. Still ugly — deliberately; the loop is proven, not finished.
- **Amdahl check:** predicted 8–12× E2E from 97.7% share; realized 7.6–9.4×. Within prediction. No stacking indicated yet.
- **Re-rank:** `gated_delta_attention_f32` (1.21 s, 27%) is now the top contributor — the mechanical next R02 candidate if the loop repeats.

## Promotion gates (all required, all met)

1. Correctness green: D-021 verdict **pass** on the optimized binary in both sessions (95 strict exact, 5 ties in-set, 66 listed outliers reported, bounds clear).
2. Region genuinely improves: 33× local, same workload/manifest/GPU.
3. E2E improves: positive on all three cases, both sessions.
4. Second session reproduces: sign, magnitude, and **byte-identical captures** A↔B.
5. No invalidation: same artifact/config/GPU/power envelope; only the NVFP4 launch site changed; no fallback, transfer, memory, or sync delta (peak VRAM unchanged at 18.37 GB class; launch count unchanged at 2,424/token).

## Evidence

- `benchmarks/runs/R03/` — before/after wall logs, D-021 verdicts, SHA256SUMS; nsys before/after reports in `build/evidence/profiler-reports/` (untracked binary blobs: r01 `6dc73cf3…`, r03 `bbe39a69…`).
- Differential test: `tests/gpu/sm120/cuda_plan_executor_test.cu::test_nvfp4_row_parallel_identity` (bit-exact, 3 shapes) — green.
- Implementation: one commit, one launch-site change, incumbent retained, rollback = one revert.

## What this proves

SuperInfer's hardware-specialization thesis now has direct evidence: a profiler-selected, capability-preserving optimization produced a reproduced end-to-end decode speedup without correctness regression. The manual profiler → optimize → correctness → speedup loop works; autoresearch schema (S05) can now be grounded in this proven loop shape instead of an assumed one.
