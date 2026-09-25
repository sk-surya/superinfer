# W4A8 DP4A NVFP4 projection — MEASURED REJECTION

git_sha: 9b2b8ee (baseline) — implementation reverted, measurement retained
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)
donor: gittensor-ai-lab/sparkinfer (MIT), kernels/csrc/cuda/gemm/gemv.cu `si_nvfp4_i8x8`,
       `si_nvfp4_quant_x_kernel`, `gemv_nvfp4_rows_dp4a_kernel`.

## What was built (all correct, all measured)

* `e2m1_to_i8x8_device` — exact E2M1 nibble -> signed int8 magnitude decode via two `__byte_perm`
  byte-LUTs plus `__vsub4` sign apply, reproducing SparkInfer `si_nvfp4_i8x8`. Mapping is exactly
  `2 * e2m1_value` = `{0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}` (E2M1 magnitudes are exact
  half-integers under PRMT + bytewise subtract, so the map is exact; code 8 / negative zero also maps
  to 0).
* `nvfp4_gemv_dp4a_f32` (kernel 33) — per-CTA group-16 int8 activation quantization (`amax/127`,
  `__float2int_rn`, RNE) into dynamic shared memory, then the DP4A group dot with
  `contribution = (e4m3_group * tensor_scale * activation_scale * 0.5) * sum int_weight*q`.
  Same accumulation/warp-reduction structure as k29; k29 retained as fallback; selected by a
  specialization-time provider flag, all 401 projections routed to kernel 33 for the measurement.

## Result: slower

| | baseline k29 | DP4A |
|---|---:|---:|
| projection subtotal ms/token | **13.676** | **16.509** (16.628 before the dynamic-shared fix) |
| total kernel sum ms/token | **22.141** | **24.921** |
| device span ms/token | 23.34 | 26.08 |
| launches/token | 2424 | 2424 |

Correctness at every step: greedy token unchanged, chat-60 **60/60** vs P7; no NaN/Inf.

The dynamic-shared change (sizing the quantization scratch per shape instead of a worst-case static
21.8 KB) recovered only 0.12 ms, so occupancy was not the cause.

## Why it lost

The DP4A inner loop is not cheaper than the hardware-decoded FP32 loop in this regime:

* k29 already decodes E2M1 with the hardware `__nv_fp4x2_e2m1` conversion (measured bit-identical to
  the software table in P7), so its per-group cost is already low.
* The DP4A path adds a full activation round trip through shared memory (write K int8 + K/16 scales,
  then re-read K bytes per warp) plus the per-group int8 staging, in exchange for replacing 16 FMAs
  with 4 `dp4a`. Net instruction count is roughly a wash, and the added shared-memory traffic and
  latency dominates.
* The earlier P7 decomposition already showed this kernel is not FP32-FMA-bound; the DP4A route
  removes arithmetic that was not the bottleneck and adds memory traffic that is.

Rejection rule from the sprint brief: "If integrated DP4A projection cost remains >=13 ms: reject it."
It is 16.5 ms. No further DP4A tuning performed.
