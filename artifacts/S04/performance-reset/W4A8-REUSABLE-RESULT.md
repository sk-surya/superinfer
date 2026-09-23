# Reusable W4A8 DP4A (one quantization per logical activation) — MEASURED REJECTION

git_sha: eab9228 baseline — implementation reverted, measurement retained
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)
donor: gittensor-ai-lab/sparkinfer (MIT) `si_nvfp4_i8x8`, `si_nvfp4_quant_x_kernel`,
       `gemv_nvfp4_rows_dp4a_kernel`, `launch_gemv_nvfp4_quant_x`.

## This run implemented the ACTUAL reusable architecture

The prior experiment quantized the activation inside every projection CTA (1088 redundant
quantizations per 17408-row matrix) and therefore did not test SparkInfer's design. This run fixed
that:

* `nvfp4_quantize_i8_group16` (kernel 34) — standalone reusable quantizer, one thread per 16-element
  group, `amax/127` + `__float2int_rn` RNE, emitting `int8[K]` + `FP32[K/16]` into static arena
  scratch.
* `nvfp4_gemv_dp4a_pre_f32` (kernel 35) — DP4A projection with ZERO activation quantization; it
  consumes the prequantized `int8* xq` / `float* xscale` directly.
* `semantic_lowering` emits ONE quantize command per logical activation, then the projections that
  share it consume the int8 form.

Verified from the produced Physical Plan: **64 quantizations/token and 128 DP4A projections/token**
for the 64-layer FFN family — one quantization per layer, shared by gate and up. No per-CTA and no
per-projection quantization remains.

## Result: the FFN family is NOT faster once the quantizer is paid for

| FFN gate+up family | ms/token |
|---|---:|
| incumbent: 128 x k29 | **5.878** |
| reusable W4A8: 128 x DP4A (kernel 35) | 5.788 |
| + 64 activation quantizations (kernel 34) | 0.157 |
| **reusable W4A8 combined** | **5.945** (1.1% slower) |

Whole-model:

| | incumbent | reusable W4A8 |
|---|---:|---:|
| kernel sum | **22.138** | **22.209** |
| device span | 23.34 | 23.40 |
| launches/token | 2424 | 2488 |

Correctness: first-token greedy identical (`49276`); **chat-60 diverges — 40/60 tokens**.

## Diagnosis (the one obvious implementation question, answered)

The DP4A inner loop is only **1.5%** faster than k29 despite replacing 16 FP32 FMAs with 4 `dp4a`
and a hardware-free nibble decode — i.e. removing most of the arithmetic changes nothing. There is no
implementation defect to fix: the projection kernel is memory/latency-bound, not arithmetic-bound, so
removing arithmetic cannot pay for an extra quantization pass. This is the same conclusion the P7
decomposition reached (`B3` input-load/multiply bound, decode only ~0.14 ms of 4.06 ms), and the
second independent arithmetic rewrite to confirm it.

Rejected. k29 (W4A16) remains the production projection path.
