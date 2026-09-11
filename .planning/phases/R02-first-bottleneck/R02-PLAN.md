# R02 Plan — Multi-Block Row-Parallel `nvfp4_linear_f32`

**Status:** Draft → self-review → autonomous execution (pre-authorized under D-020/D-014).
**Target source:** `backends/sm120/runtime/cuda_plan_executor.cuh:311` (`nvfp4_linear_f32`), launched by `launch_nvfp4_linear` (`cuda_plan_executor.cuh:634`).

## Measured bottleneck share (R01)

`nvfp4_linear_f32`: 84.62 s / 86.63 s GPU kernel time = **97.7%** of steady-state decode; 1,203 launches, 70.3 ms avg. Next contributor 1.4%. Amdahl E2E ceiling ≈ **43×** if eliminated. Nothing else can move E2E.

## Resource/roofline hypothesis

The kernel launches `<<<1, 256>>>`: one thread-block uses ~1/148 SMs. Each thread serially reduces whole output rows (up to 248,320 rows for the LM head) with per-element `ldexpf` E4M3 decode and strided packed loads. Occupancy-starved and latency-bound; arithmetic intensity is unchanged by parallelization, but ~100 idle SMs absorb the row-parallel work and the memory system serves contiguous row chunks instead of one strided stream.

## Change (only this + tests)

Add `nvfp4_linear_rows_f32` beside the incumbent: grid = `clamp(ceil(output_elements/256), 1, 2048)` blocks of 256 threads; thread `(b, t)` reduces exactly the same rows in exactly the same column order as baseline thread `t` did (`row = b*256 + t + k*gridDim*256`, serial column loop with identical `decode_e2m1_device`/`decode_e4m3fn_device`/accumulate sequence). Per-row floating-point operation order is unchanged by construction → **bit-identical outputs** for all shapes, including small-row launches (grid collapses to 1 block = old mapping).

Switch `launch_nvfp4_linear` to the new kernel. Keep `nvfp4_linear_f32` in the file untouched as the fallback.

## Expected speedup

- Local: 15–30× on large-row launches (occupancy 1→40–970 blocks; memory-bound, not perfectly linear). Small-row launches unchanged.
- E2E (Amdahl with 97.7% share): 20× local on ~90% of nvfp4 time ≈ 8–12× per-token (31 s → ~3 s). Verify, do not trust.
- If E2E gain < 3×, stop: re-profile instead of stacking phase-2 (intra-row split) work.

## Correctness differential (D-006)

1. New CUDA differential test (beside the owning module, `tests/gpu/sm120/`): baseline vs row-parallel kernel on shapes {48×5120, 1024×5120, 10240×5120, 2048×2048} with fixed seed → require **bit-exact** equality (same op order ⇒ exact).
2. Full-model gate: D-021 session verdict on the 8-case corpus must remain **pass** with byte-identical captures to the R01 baseline run (bit-identity predicts identical bytes; any difference fails the hypothesis and blocks promotion).
3. Existing CPU CI + `validate.py --full` green.

## Fallback/rollback

- Incumbent kernel stays in the file; promotion = one launch-site change.
- Rollback is one `git revert`: restore `launch_nvfp4_linear` to `nvfp4_linear_f32`.
- No new dependencies, no kernel-ID/schema change, no hot-path branch (direct call swap).

## Benchmark comparison (R03 input)

Same manifest (`benchmarks/manifests/qwen38-results-first-v1.json`): R01 baseline wall times vs candidate on special-3/chat-60/long-103 + nsys kernel table before/after + D-021 verdict. Report local target speedup AND end-to-end decode speedup separately, each reproduced in a second fresh session.

## Self-review

- Scope: one kernel + launch site + tests. No S04 portfolio, no Flash-Next, no autoresearch. ✓
- Profiler chose the target (97.7%, 70× next). Not interest-driven. ✓
- Bit-identity by construction + test + full-model byte check: three independent correctness nets. ✓
- Rollback is atomic revert. ✓
- Risk: LM-head-scale grids (970 blocks) may contend on input-vector broadcast; bounded by measurement (phase-1 result decides whether phase-2 exists). Known and accepted.
