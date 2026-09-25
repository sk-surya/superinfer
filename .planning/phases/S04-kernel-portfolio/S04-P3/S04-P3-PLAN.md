# S04-P3 Plan — Value-Dimension-Parallel Gated DeltaNet

**Status:** selected from the S04-P2 post-profile; executed autonomously (D-020/D-014).
**Target:** `gated_delta_attention_f32` (`backends/sm120/runtime/cuda_plan_executor.cuh`), launched from `launch_gated_delta_attention`.
**Loop:** fourth real profiler -> hypothesis -> implementation -> correctness -> benchmark cycle.

## Measured current share

54.1% of post-loop-3 steady-state decode GPU time (24.28 s / 44.90 s over 60 tokens; 2,880 launches, 8,431 us avg). New #1 after the NVFP4 fix. See `S04-P2/S04-P2-SUMMARY.md`.

## Root performance hypothesis

Launched `<<<1,256>>>` but only `value_heads` (48) threads do work, and each thread serially owns a full
`key_dimension x value_dimension` (128x128) state: the decay pass, the `key_value` reduction, the rank-1
state update and the output reduction are all `O(key_dim * value_dim)` per head per position. One block
on one SM with 1.5 active warps computes ~3.1 MFLOP in 8.4 ms (~0.37 GFLOP/s).

## Change (only this kernel + launch site + tests)

Add `gated_delta_attention_parallel_f32` beside the incumbent; one block per value head, one thread per
value dimension. Each `value_index` is an independent recurrence column, so parallelising over it
preserves every summation/update order: bit-identical output and state. Only the head-level scalars
(query/key norms, scales, decay, beta) are computed once in shared memory to avoid redundant work; the
expressions are unchanged.

## Expected speedup

- Local: ~`value_dimension` (up to 128x) minus occupancy/bandwidth limits; predicted >=10x.
- Amdahl E2E bound (54.1% share): local 10x => ~1.8x E2E; local 20x => ~2.0x E2E. Well above threshold.

## Independent correctness oracle

- Unit differential: sequential incumbent vs parallel kernel, bit-exact over state AND output, shapes
  including `key_heads/value_heads` 4/48 and 2/8, `key_dim` 128/64, `value_dim` 128/96, positions 1/2/3.
- Model gate: D-021 corpus **pass**; captures byte-compared to S04-P2.
- `python tools/validate.py --full` green.

## Fallback / rollback

Incumbent retained; promotion is one launch-site change; rollback is one revert.

## Benchmark

Same manifest: chat-60 / long-103 wall times, nsys kernel table, D-021 verdict, second fresh session.

## Self-review

One kernel, one launch site, one test. Target selected from the fresh profile (54.1%, next 20.7%).
Bit-identity by construction (independent columns) plus unit + model gate. Fallback retained.
