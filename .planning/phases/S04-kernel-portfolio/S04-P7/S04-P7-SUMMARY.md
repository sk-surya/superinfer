# S04-P7 Summary — NVFP4 Decode Decomposition; Hardware E2M1 and Scale Predecode Rejected

**Status: NO PROMOTION (below threshold), plus a shape-adaptive dispatch cleanup.** Eighth loop.

## Method

Extended `tests/gpu/sm120/nvfp4_decode_bench.cu` with discriminating microbenchmarks (Nsight Compute
counters remain unavailable: `RmProfilingAdminOnly: 1`). Real shapes, CUDA-event timing, weighted by the
artifact's per-token multiplicities.

## Decode cost decomposition (ms/launch)

| Stage | lm_head 248320×5120 | mlp 17408 | gdn 10240 | attn 6144 | down 5120×17408 |
|---|---:|---:|---:|---:|---:|
| A raw packed+scale stream | 0.639 | 0.016 | 0.011 | 0.007 | 0.016 |
| B software E2M1 decode only | 0.780 | 0.058 | 0.036 | 0.024 | 0.063 |
| B2 decode + scale + tensor mul, **no input load** | 0.874 | 0.065 | 0.041 | 0.027 | 0.070 |
| B3 decode + **input load** + one mul | **4.002** | 0.287 | 0.177 | 0.111 | 0.300 |
| C full software (no cross-lane reduce) | 4.057 | 0.292 | 0.181 | 0.115 | 0.311 |
| D P6 warp-per-row | 4.061 | 0.292 | 0.181 | 0.114 | 0.312 |

**Finding:** memory floor (`A`) is 0.639 ms; E2M1 decode adds only ~0.14 ms and scale decode ~0.09 ms.
The cost appears at `B3` — the **per-element `input` load + multiply** (4.00 vs 0.87 ms). The kernel is
bound by the input-operand access and the FP32 multiply stream, not FP4 decode, scale decode, or the
reduction. The memory-only lower bound (~8 ms/token) is not reachable by this software-decode structure.

## P7-2 — hardware E2M1 (`cvt.rn.f16x2.e2m1x2`)

- Exhaustive proof over all 256 packed bytes: **EXACT** vs the software magnitude table (max_abs = 0).
- The combined hardware-decode + predecoded-FP16-scale kernel is **bit-identical to P6** (D-vs-E
  max_abs = 0 on every shape).
- Performance: **1.01×** weighted (76.0 vs 76.5 ms/token). Decode was not the bottleneck.

## P7-3 — predecoded FP16 scales

- Exhaustive round-trip: every valid E4M3 scale code → FP16 → FP32 equals the incumbent value exactly.
- Removes `ldexpf` from the inner loop, but yields **no measurable gain** (scale decode is 0.09 ms of
  4.06 ms). FP32-predecoded scales were therefore **not** pursued (would add ~6.4 GB/token for nothing).

## P7-4 — limited warp-group / mapping sweep

| Variant | weighted ms/token | vs D |
|---|---:|---:|
| D P6 warp-per-row (contiguous chunks) | 76.5 | 1.00× |
| E hardware E2M1 + FP16 scales | 76.0 | 1.01× |
| F lane-interleaved (coalesced) hardware | 74.7 | 1.02× |
| 2 warps/row | no material change | ~1.00× |
| 4 warps/row | no material change | ~1.00× |

Coalescing (F is bit-identical to P6 within ~2e-8) and warp-group splitting do not move the needle —
consistent with the input-load/multiply-bound diagnosis.

## P7 decision

**No candidate reached the promotion threshold** (≥1.5× weighted local or ≥15% E2E). P7 is rejected; the
P6 warp kernel remains. The measured hardware limit is instruction/latency-bound per-element input access
and FP32 multiply, which the memory-only 8 ms/token figure does not describe. Escaping it requires
raising arithmetic intensity — i.e. native block-scaled NVFP4 **tensor-core MMA** — which is P8.

## P7-0 shape-adaptive cleanup (promoted)

P6's warp kernel regresses two large shapes (LM head 4.06 vs 1.35 ms; MLP 17408 0.292 vs 0.253 ms).
Dispatch now selects the bit-exact row-per-thread vector kernel for `output_elements >= 16384` and the
warp kernel otherwise — a deterministic function of the compile-time output size (buffer metadata), not a
model-name or environment decision. Both kernels remain as fallbacks.
