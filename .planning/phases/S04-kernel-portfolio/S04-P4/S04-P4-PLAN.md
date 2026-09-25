# S04-P4 Plan — Row-Parallel RMSNorm with Bit-Exact Denominator

**Status:** selected from the S04-P3 post-profile; executed autonomously (D-020/D-014).
**Target:** `rms_norm_f32_bf16_scale` (`backends/sm120/runtime/cuda_plan_executor.cuh`), launched from `launch_rms_norm_bf16`.
**Loop:** fifth profiler -> hypothesis -> implementation -> correctness -> benchmark cycle.

## Measured current share

32.7% of post-loop-4 steady-state decode GPU time (6.80 s / 20.78 s over 60 tokens; 12,540 launches, 542 us avg). New #2 after NVFP4, and the largest kernel still launched `<<<1,1>>>` (a single thread for the whole tensor).

## Root performance hypothesis

One CUDA thread performs the entire tensor: for each row it (a) sums squares sequentially over
`scale_elements` and (b) writes `scale_elements` outputs, both with global-memory latency and zero
parallelism. `scale_elements` is 5120 for the layer norms, so the sum and the store each cost thousands
of serialised memory round-trips. The reduction order is what forces single-thread execution — a naive
tree reduction would change the denominator bits.

## Change (only this kernel + launch site + tests)

Add `rms_norm_f32_bf16_scale_parallel`: one block per row, 256 threads. The row is staged into static
shared memory with coalesced loads; **thread 0 then accumulates the sum of squares sequentially in the
original element order** (bit-identical denominator); the output write is parallelised. The per-column
scale is read from global as `scale[index]` (it is broadcast across rows, exactly as the incumbent does).
Falls back to the incumbent when `scale_elements > 8192` or the row count is out of range.

## Expected speedup

- Local: the parallel staging and store dominate; predicted >=8x (serial part is only the 5120-add sum
  in thread 0, ~3 us, versus the incumbent's full serial pass).
- Amdahl E2E bound (32.7% share): local 8x => ~1.4x E2E; local 20x => ~1.5x E2E.

## Independent correctness oracle

- Unit differential: `<<<1,1>>>` incumbent vs parallel kernel, bit-exact, shapes {1,5120}/{4,5120}/{64,128}
  and both `add_one_to_scale` values (`test_rms_norm_parallel_identity`).
- Model gate: D-021 corpus **pass**; captures byte-compared to S04-P3.
- `python tools/validate.py --full` green.

## Fallback / rollback

Incumbent retained; promotion is one launch-site change; rollback is one revert.

## Benchmark

Same manifest: chat-60 / long-103 wall times, nsys kernel table, D-021 verdict, second fresh session.

## Self-review

One kernel, one launch site, one test. Target selected from the fresh profile (32.7%). Bit-identity by
construction (thread-0 sequential sum over shared staging) plus unit + model gate. Fallback retained.
Note: at this point the fixed artifact load (~25-30 s/process) is a large fraction of wall time, so the
honest decode metric is the marginal slope, not the amortised wall.
