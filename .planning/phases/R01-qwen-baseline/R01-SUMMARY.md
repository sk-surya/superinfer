# R01 Summary — First Qwen Baseline and Critical-Path Profile

**Status:** Complete. Exactly one R02 target selected from profiler evidence. No optimization performed in R01.

## Baseline (ugly, as-found, commit `5af6bc6`, artifact `e25022c8…dc65d5`, GPU1)

| Metric | Current SuperInfer |
|---|---|
| Decode | 0.032 tok/s |
| TPOT | ~31,300 ms (31.3 s/token; 30.7 s at 60 rows → 31.6 s at 103 rows: superlinear) |
| Prefill (3-token cold, incl. load) | 115 s wall; compute-only prefill not yet isolated |
| TTFT (cold, incl. 18 GB load + 19.2 GB upload + compile) | ~115 s |
| Peak VRAM | 18.37 GB device (19,191,265,792-byte arena); host RSS 18.5 GB |
| Kernel launches | 7,272 per 3-token run = 2,424/token; all `<<<1, 256>>>` single-block |
| Host syncs | 1 `synchronize_for_test` per token position; single stream, no D2D transfers |
| One-time H2D | 19.19 GB (arena upload); steady-state decode does no bulk transfer |

Decode fit over (3, 60, 103)-row cases: slope ≈ 31.3 s/token. Short cases are load-dominated; the slope is the honest decode number.

## Ranked decode critical-path contributors (nsys, 3-token run, 86.63 s GPU kernel time)

| # | Region/kernel | GPU time | Share | Launches | Avg/launch | Actionable now? |
|---|---|---|---|---|---|---|
| 1 | `nvfp4_linear_f32` (decode GEMV/GEMM over NVFP4) | 84.62 s | **97.7%** | 1,203 | 70.3 ms | **YES — R02 target** |
| 2 | `gated_delta_attention_f32` | 1.21 s | 1.4% | 144 | 8.4 ms | Later |
| 3 | `rms_norm_f32_bf16_scale` | 0.34 s | 0.4% | 627 | 0.54 ms | Later |
| 4 | `grouped_attention_bf16_cache` | 0.24 s | 0.3% | 48 | 5.0 ms | Later |
| 5 | `cast_bf16_to_f32` + rest | 0.22 s | 0.3% | 5,250 | ≤0.2 ms | Later |

## R02 target (exactly one)

**`nvfp4_linear_f32`** in `backends/sm120/runtime/cuda_plan_executor.cuh:311`, launched from `launch_nvfp4_linear` (`cuda_plan_executor.cuh:634`).

- **Measured share:** 97.7% of steady-state decode GPU time → Amdahl E2E ceiling ≈ 43× if eliminated.
- **Resource hypothesis:** single-block launch (`<<<1, 256>>>`) uses ~1/148 SMs; each thread serially reduces whole rows with per-element `ldexpf` E4M3 decode and uncoalesced packed loads. The kernel is occupancy-starved, not ALU-bound.
- **Why nothing else:** the next contributor is 70× smaller; optimizing anything but #1 cannot move E2E (Amdahl ceiling 1.4% even if free).

## Evidence

- Manifest (draft): `benchmarks/manifests/qwen38-results-first-v1.json`
- Wall-clock logs: `/tmp/opencode/r01-{special-3,chat-60,long-103,prefill}.log` (raw; hashes below on commit)
- VRAM sampler: `/tmp/opencode/r01-vram.csv` (2 s cadence, 3,822 samples)
- nsys report: `build/evidence/profiler-reports/r01-nsys-special3.nsys-rep` (untracked; sha256 `6dc73cf3…`, binary profiler blob excluded from git per push-protection) + sqlite tables summarized inline above
- Correctness gate: D-021 session-2 verdict **pass** on the measured build; R01/R03 comparisons reuse it.

## Non-goals respected

No kernel changed, no Flash-Next work, no autoresearch, no public comparative claim. R02 may change only the selected bottleneck and its immediate launch path.
