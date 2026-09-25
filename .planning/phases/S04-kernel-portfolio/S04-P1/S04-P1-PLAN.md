# S04-P1 Plan — KV-Window Attention: Cache Scores, Remove the O(head_dim²) Value Pass

**Status:** selected from fresh profile; execute autonomously (D-020/D-014).
**Target:** `grouped_attention_bf16_cache` (`backends/sm120/runtime/cuda_plan_executor.cuh:160`), launched at `:888` (kernel id 23).

## Measured current share

48.8% of steady-state decode GPU time (82.15 s / 168.5 s over 60 tokens; 960 launches at 85.6 ms avg). Largest single kernel and largest E2E lever. See `S04-P1-PROFILE.md`.

## Root performance hypothesis

Two independent defects, both measurable in the source:

1. **Redundant recomputation / algorithmic blow-up.** The value pass is dimension-outer and recomputes the Q·K score inside the per-dimension loop:
   `for dimension: for position: score = Σ_key_dim q·k; result += prob(score)·v`.
   That is `O(positions · head_dim²)` per query head. Q·K is invariant across dimensions and is computed three times total (max pass, denominator pass, value pass).
2. **Occupancy starvation.** Launched `<<<1,256>>>` with only `query_heads` (24) active threads.

## Change (only this kernel + launch site + tests)

Add `grouped_attention_bf16_cache_cached` beside the incumbent; switch the launch. Keep the incumbent untouched as fallback.

Design (bit-identical by construction):
- Grid = `query_heads` blocks × 256 threads; dynamic shared memory `2·positions·sizeof(float)`.
- Phase 1: compute `score[p]` once (each thread does a full sequential dot for its positions ⇒ identical accumulation order).
- Max: parallel `fmaxf` (max is exact, order-independent).
- Phase 2: `exp[p] = expf(score[p]·scale − max)` in parallel; denominator summed **sequentially by thread 0** so sum order is identical.
- Phase 3: `prob[p] = exp[p] / den`.
- Phase 4: `for dimension = tid; dimension < head_dim; dimension += blockDim` accumulate over positions in order, using cached `prob[p]`.

Every value that entered the incumbent's arithmetic is reproduced with the same operands in the same order, so outputs are bit-identical for all shapes and positions.

## Expected speedup

- Local: removes a `head_dim`=256× factor from the dominant pass plus two redundant score passes. Predicted **≥30×**; realistically 100×+ at the measured 24-thread occupancy.
- Amdahl E2E bound (48.8% share): local 30× ⇒ ~1.45× E2E; local 100× ⇒ ~1.9× E2E. This is well above the 15% threshold.
- Verify, do not trust; if E2E gain < 15%, re-profile instead of stacking.

## Independent correctness oracle

- Unit differential: incumbent vs cached kernel over shapes covering `head_dim∈{64,128,256}`, `positions∈{1,7,64,257}`, `query_heads/kv_heads∈{24/4, 8/2}`, deterministic LCG inputs → require **bit-exact** equality.
- Model gate: D-021 verdict on the 8-case corpus must remain **pass**; capture bytes compared to R03.
- Existing layer differentials unchanged; `python tools/validate.py --full` green.

## Fallback / rollback

Incumbent kernel retained; promotion is one launch-site change; rollback is one revert. No schema/kernel-id change.

## Exact before/after benchmark

Same manifest (`benchmarks/manifests/qwen38-results-first-v1.json`): 3/60/103-row wall times, nsys kernel table before/after, D-021 verdict, second fresh session. Report local target speedup and E2E decode speedup separately.

## Self-review

One kernel, one launch site, one test. Profiler-selected (48.8%, 2× next). Bit-identity by construction plus unit + model gates. Fallback retained. No fusion/GDN/NVFP4 scope creep. Acceptable residual risk: shared-memory size at 4096 positions (32 KB) — within Blackwell limits; guard the launch on it.
