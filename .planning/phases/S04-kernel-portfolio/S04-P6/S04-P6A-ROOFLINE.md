# S04-P6A — NVFP4 Roofline and Candidate Measurement

**Status:** measured (Nsight Compute unavailable on this host; direct event-timed microbenchmark used).

## Counter availability

`ncu` is blocked: `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1` and the account has no
passwordless sudo. Hardware performance counters (L2 hit rate, sector efficiency, issue stalls,
occupancy) are therefore **not available**. Achieved DRAM bandwidth is measured directly instead:
the kernels are timed with CUDA events and bytes/time is computed against the 1.79 TB/s RTX 5090
DRAM roofline. Coalescing is inferred from layout + differential results, and the candidate that
reduces the true bottleneck wins by measurement, not by counter.

Harness: `tests/gpu/sm120/nvfp4_gemv_bench.cu` (registered as `superinfer.sm120.nvfp4_gemv_bench`).

## Checked roofline (per token)

NVFP4 weight traffic per token: **12.80 GB packed + 1.60 GB FP8 scales = 14.40 GB**.
Roofline at 1.79 TB/s = **8.0 ms/token**. Measured pre-P6 NVFP4 time ≈ **144.8 ms/token** (sum over the
401 launches/token using the per-shape incumbent times below) → **~5.5% of roofline, ~18× off**.

## Shape classes (artifact-derived, 401 NVFP4 launches/token)

| Shape (rows×in) | count | packed MB |
|---|---:|---:|
| 248320×5120 (LM head) | 1 | 635.7 |
| 5120×17408 (down) | 64 | 44.6 |
| 17408×5120 (mlp) | 128 | 44.6 |
| 12288×5120 (qkv) | 16 | 31.5 |
| 10240×5120 (gdn) | 48 | 26.2 |
| 6144×5120 (attn) | 48 | 15.7 |
| 5120×6144 (down) | 64 | 15.7 |
| 1024×5120 (small) | 32 | 2.6 |

## Measured kernel times (ms/launch, CUDA events)

| Shape | incumbent (row/thread, row-major) | A interleaved | B warp/row |
|---|---:|---:|---:|
| lm_head 248320×5120 | 1.347 | 1.030 | 4.060 |
| mlp 17408×5120 | 0.253 | 0.235 | 0.292 |
| qkv 12288×5120 | 0.253 | 0.235 | 0.224 |
| gdn 10240×5120 | 0.253 | 0.236 | 0.181 |
| attn 6144×5120 | 0.253 | 0.235 | 0.115 |
| down 5120×17408 | 0.848 | 0.797 | 0.312 |
| down 5120×6144 | 0.319 | 0.282 | 0.110 |
| small 1024×5120 | 0.251 | 0.234 | 0.025 |

Correctness: **A is bit-exact** vs incumbent on all shapes. **B max_abs ≈ 1e-7** (≈1 ULP; FP32
reduction-order only), rmse ≈ 1e-8.

## Diagnosis (replaces the unverified "coalescing" hypothesis)

The incumbent's per-launch time is **nearly constant (~0.25 ms) across shapes from 2.6 MB to 44.6 MB**.
That is not bandwidth behaviour — it is a **fixed per-row decode-latency floor**: one thread serially
decodes a full row (`decode_e4m3fn_device` via `ldexpf` per 16 columns + `decode_e2m1_device` table
lookup per element) with little ILP to hide it. Interleaving the layout (A) raises achieved bandwidth
only ~1.3× because bandwidth was never the binding constraint for these shapes.

Weighted per-token using the real multiplicities:

| Kernel | ms/token |
|---|---:|
| incumbent | **144.8** |
| A interleaved | 134.0 (1.08×) |
| B warp/row | **87.0 (1.66×)** |

B's only regressions are the single LM-head launch (4.06 vs 1.35 ms) and the 128 mlp launches
(0.292 vs 0.253); it wins 10× on the 32 small and 2–3× on the down projections.

## Decision

A preserves bit-exactness but yields only ~1.08× — below the promotion threshold and not "comparable
performance". B's candidate-level E2E gain is ~1.66× on NVFP4 (≈1.45× E2E at the 80% share), with a
~1e-7 numerical difference. Proceed to integrate B into the executor and measure the real model (P6-B/C).
Prefer a shape-adaptive selection if B regresses the LM-head launch in the real run.
