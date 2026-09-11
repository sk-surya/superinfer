# S04-P5 Plan — Block-Parallel Remaining Single-Block Dense Kernels

**Status:** selected from the S04-P4 post-profile; executed autonomously (D-020/D-014).
**Targets:** `cast_bf16_to_f32` (launch `launch_cast_bf16_to_f32`), `cast_f32_to_bf16` (`launch_cast_f32_to_bf16`), `linear_f32` (`launch_lm_head`).
**Loop:** sixth profiler -> hypothesis -> implementation -> correctness -> benchmark cycle.

## Measured current share

Post-loop-5 profile (60 tokens, 14.31 s GPU): `cast_bf16_to_f32` 19.0% (2.72 s, 45,240 launches, 60 us avg), `linear_f32` 7.7% (1.11 s, 5,760 launches, 192 us avg), `cast_f32_to_bf16` 1.0%. Combined **~27.7%**. All three are still launched `<<<1,256>>>`.

## Root performance hypothesis

Each kernel's loop is already grid-stride and elementwise/row-independent, but the launch uses a single
block, so the whole tensor is processed by 256 threads on one SM. Occupancy, not arithmetic, is the cost.

## Change (only these launch sites)

Launch `min(4096, ceil(elements/256))` blocks instead of one. Because the work is elementwise
(`cast_*`) or per-row (`linear_f32`), the result is bit-identical for any block count. No kernel body
changes; only the launch configuration.

## Expected speedup

- Local: 45,240 launches at 60 us and 5,760 at 192 us should collapse toward launch-latency floor.
- Amdahl E2E bound (27.7% share): local 5x => ~1.23x E2E; local 10x => ~1.33x E2E.

## Independent correctness oracle

- Existing executor unit tests (bf16<->f32 cast, lm_head) still pass.
- Model gate: D-021 corpus **pass**; captures byte-compared to S04-P4.
- `python tools/validate.py --full` green.

## Fallback / rollback

One-line-per-site launch change; rollback is one revert. Kernel bodies untouched.

## Benchmark

Same manifest: chat-60 / long-103 wall times, nsys kernel table, D-021 verdict, second fresh session.
