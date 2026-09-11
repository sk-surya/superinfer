# S04-P2 Plan — Vectorized, Scale-Hoisted NVFP4 Projection

**Status:** selected from the S04-P1 post-profile; executed autonomously (D-020/D-014).
**Target:** `nvfp4_linear_rows_f32` (`backends/sm120/runtime/cuda_plan_executor.cuh`), launched from `launch_nvfp4_linear`.
**Loop:** third real profiler -> hypothesis -> implementation -> correctness -> benchmark cycle.

## Measured current share

58.8% of post-loop-2 steady-state decode GPU time (50.83 s / 86.45 s over 60 tokens; 24,060 launches, 2,112 us avg). New #1 after the KV-attention fix. See `S04-P1/S04-P1-SUMMARY.md`.

## Root performance hypothesis

Measured traffic is 14.4 GB/token of NVFP4 weights + FP8 scales (roofline 8.0 ms at 1.79 TB/s), but the
kernel takes ~847 ms/token → **~17 GB/s effective, ~106x off the HBM roofline**. The cause is not the
roofline but instruction/latency overhead:

1. the FP8 block scale is decoded with `ldexpf`-based `decode_e4m3fn_device` **once per element** though
   it is constant for 16 consecutive elements (15/16 redundant);
2. packed weights are read **one byte per two columns** as scalar loads.

## Change (only this kernel + launch site + tests)

Add `nvfp4_linear_rows_vec_f32` beside the incumbent; switch the launch when the row is 32-element
aligned and the packed pointer is 16-byte aligned, else fall back to `nvfp4_linear_rows_f32`.

- Read 16 packed bytes (32 codes) per `uint4` load instead of 32 scalar loads.
- Decode each FP8 block scale once per 16 columns.
- Keep the exact per-row, per-column accumulation order and the exact multiplication order
  (`e2m1 * e4m3 * tensor_scale` then `* input`), so outputs are **bit-identical**.

## Expected speedup

- Local: predicted 3–6x from fewer instructions and better memory-level parallelism.
- Amdahl E2E bound (58.8% share): local 4x ⇒ ~1.44x E2E; local 6x ⇒ ~1.53x E2E. Above the 15% threshold.

## Independent correctness oracle

- Unit differential: scalar incumbent vs vectorized kernel, bit-exact, over aligned and tail shapes
  (`test_nvfp4_row_parallel_identity`).
- Model gate: D-021 verdict on the 8-case corpus **pass**; captures byte-compared to S04-P1.
- `python tools/validate.py --full` green.

## Fallback / rollback

Incumbent retained; promotion is one launch-site change; rollback is one revert.

## Benchmark

Same manifest: chat-60 / long-103 wall times before/after, nsys kernel table, D-021 verdict, second
fresh session.

## Self-review

One kernel, one launch site, one test. Target selected from the fresh profile (58.8%, next 28.1%).
Bit-identity by construction + unit + model gate. Fallback retained. No fusion/GDN scope creep.
