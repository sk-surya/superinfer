# BF16-native k29 (BF16 activation input, elided widening cast) — MEASURED, REVERTED

git_sha: 579f012 baseline — implementation reverted, measurement retained
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)

## What was built

* `nvfp4_gemv_rows_bf16_f32` (kernel 36) — textually identical to k29 except the activation is read as
  packed BF16 (`uint2`, 16 values/lane/phase -> `bf16x2_bits_to_float2_device`, exact
  `__uint_as_float(half << 16)` widening matching `cast_bf16_to_f32` and `bf16_to_float_device`).
  Same code loads, scale decoding, tensor scale, accumulator chains, FMA order, warp reduction, FP32
  output.
* `semantic_lowering`: narrow compile-time pattern — when a `gated_dense_ffn`'s leading activation is
  an authored BF16 tensor that the op would only widen, keep it BF16, skip the emitted
  `cast_bf16_to_f32`, and emit the BF16-native projection.
* `e2a_gemm_provider` routes `nvfp4_linear_bf16` -> kernel 36 behind a compile-time flag; k29 retained.

## Result: correct and bit-identical, but too small to keep

FFN family (64 layers, gate+up), 60-token chat:

| | ms/token |
|---|---:|
| incumbent k29 (128 launches, gridX=1088) | 5.878 |
| BF16-native k36 (128 launches, gridX=1088) | 5.792 |

Whole model:

| | baseline | BF16-native FFN |
|---|---:|---:|
| kernel sum | **22.138** | **21.981** (−0.157) |
| device span | 23.34 | 23.12 (−0.22) |
| launches/token | 2424 | 2360 (−64) |
| clean wall (back-to-back, 3 runs each) | 29.27 s | 29.29 s (no measurable change) |

Correctness: **bit-identical** — first-token greedy, logit 11.6875, checksum -791994 all unchanged;
chat-60 **60/60** vs baseline in the same session, and baseline itself matches the reference 60/60.

## Why it was not kept

1. The whole-model gain is **0.157 ms/token (0.7%)**, below the sprint's own "useful >= 0.75 ms" bar,
   and the clean wall number is indistinguishable from baseline. The kernel is memory-latency bound,
   so halving the activation bytes changes almost nothing (the activation is L2-resident and small
   relative to the streamed weights).
2. **The sibling attention conversion is not sound.** Extending the identical mechanism to
   `gated_grouped_query_attention` (q/k/v, 48 projections, 16 fewer casts) produced the same expected
   plan changes and the same identical first-token checksum, but **diverged from baseline at
   continuation token 1 (42/60 same-session)**. By construction an exact-widening rewrite should be
   bit-identical there too, so this indicates a latent fragility in the activation-buffer/lifetime
   handling rather than a numerical difference. I could not explain it within this budget.
3. Landing a 0.7% gain while an unexplained correctness anomaly sits in the same mechanism is a bad
   trade.

Reverted entirely; `artifact`/`k29` production path unchanged.

## Open item for a future sprint

Explain the attention divergence before reusing this mechanism. The first thing to check is whether a
projection input buffer is aliased by another live tensor once the intervening cast command is
removed (changing arena lifetimes), since the kernel itself is provably an exact widening.
