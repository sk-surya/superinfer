# Deferred-RMSNorm + normalized-on-load specialization — MEASURED NEGATIVE RESULT

git_sha: 431d825 (baseline) — implementation reverted, measurement retained
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)

## What was built

* `rms_norm_denominator_bf16_f32` (capability `rms_norm_denominator`, kernel 31): exact-order serial
  FP32 sum over the raw BF16 activation, writes one FP32 scalar, no normalized vector.
* `nvfp4_gemv_rows_rmsnorm_f32` (capability `nvfp4_linear_rmsnorm`, kernel 32): k29 schedule with the
  activation formed on load as `bf16_round_trip(bf16(raw) / denom * (bf16(scale) + offset))`.
* `semantic_lowering` deferred-norm analysis (consumer count == 1, single supported consumer, static
  BF16 shapes, fail-closed) + FFN and attention fused emission.

All of it worked: the plan moved from `754 k16 / 387 k17 / 401 k29 / 209 k12` to
`498 k16 / 259 k17 / 176 k32 / 128 k31 / 225 k29`, i.e. **240 launches/token eliminated**
(2424 -> 2184), and correctness was preserved exactly (layer-3 `max_abs=0.00106812`, GDN
`max_abs=0.000312805`, chat-60 60/60 tokens — all identical to the accepted values).

## Why it was reverted: it is slower

| variant | fused GEMV | unfused GEMV | norms | casts | control | device sum |
|---|---:|---:|---:|---:|---:|---:|
| baseline (no fusion) | — | 13.676 | 4.032 | 1.62 | 0.78 | **22.141** |
| A: per-warp normalize, mid `<8,2,4,4>` | 12.439 | 7.245 | 3.524 | 1.322 | 0.777 | 27.337 |
| B: shared-normalized staging, mid `<8,2,4,4>` | 9.149 | 6.783 | 3.525 | 1.319 | 0.784 | 23.585 |
| C: shared staging, mid `<8,4,4,3>` | 11.090 | ~6.8 | 3.535 | 1.3 | 0.78 | 25.614 |

Best fused variant is **23.585 vs 22.141 baseline** = 1.44 ms/token slower.

## Mechanism

1. **Savings are bounded and small.** The fusion removes the two boundary casts (256 `bf16->f32` +
   128 `f32->bf16` = 384 launches ~ 0.56 ms) and the norm's FP32 output write. Total ~1.1 ms.
2. **The denominator command is not much cheaper than the norm it replaces.** 1.862 ms for the
   deferred boundaries vs ~2.0 ms of the 4.032 ms baseline norm — because the exact-order serial
   FP32 reduction (~11 us of dependent FMA chain for K=5120, a D-021 constraint that cannot be
   parallelized) dominates either way. The materialization was never the bulk of RMSNorm.
3. **The fused projection pays more than it saves.** It must read the raw BF16 activation *and* the
   BF16 norm scale (20 KB per CTA — the same bytes as the f32 normalized vector it replaces), plus
   per-element convert + RNE round-trip, plus a per-CTA shared-normalization staging pass that
   measured ~24% of CTA work. Fused shapes ran at **825 GB/s vs 1011 GB/s** for the unfused kernel.
4. Enlarging the CTA to amortize (variant C) reduced the CTA count and lost more.

## Consequence for the ≤18 ms target

The recoverable ceiling on this path is ~1.1 ms, not the 3.5-4.0 ms estimated, because the
*denominator reduction itself is the RMSNorm cost* and it is a hard D-021 constraint. Reaching
<=18 ms requires attacking the projection kernel bandwidth (13.68 ms, 62% of device time) or launch
overhead (2424 launches, 5% idle), not the norm materialization.
