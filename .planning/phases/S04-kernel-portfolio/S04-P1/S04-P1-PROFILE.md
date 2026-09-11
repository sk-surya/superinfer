# S04-P1 — Fresh R03 Profile and Next Target Selection

**Status:** profile complete; target selected from evidence. No optimization performed in this step.

## Method

Profile the current optimized R03 binary (row-parallel NVFP4) on the steady-state decode corpus:
60-token chat-template continuation, GPU1 (RTX 5090), same artifact/config as R03. Nsight Systems
full capture (`--stats=true`); Nsight Compute selectively on the leading kernel with counters.
Raw: `build/evidence/profiler-reports/s04p1-nsys-chat60.nsys-rep`; summary `/tmp/opencode/s04p1-profile.json`.

Steady state matters: a 3-token profile under-samples context-growing kernels and mis-ranks the targets.

## Fresh ranked table (60 tokens, 145,440 launches, 168.6 s GPU span)

| # | Kernel | Launches | GPU total | Share | Avg/launch |
|---|---|---:|---:|---:|---:|
| 1 | `grouped_attention_bf16_cache` | 960 | 82.15 s | **48.8%** | 85,573 us |
| 2 | `nvfp4_linear_rows_f32` | 24,060 | 50.77 s | 30.1% | 2,110 us |
| 3 | `gated_delta_attention_f32` | 2,880 | 24.23 s | 14.4% | 8,413 us |
| 4 | `rms_norm_f32_bf16_scale` | 12,540 | 6.80 s | 4.0% | 543 us |
| 5 | `cast_bf16_to_f32` | 45,240 | 2.72 s | 1.6% | 60 us |
| 6 | `linear_f32` | 5,760 | 1.10 s | 0.7% | 192 us |
| — | all remaining (rope/silu/residual/split/cast_f32_to_bf16/…) | ~47k | ~0.7 s | 0.3% | ≤77 us |

Device idle **0.10%** of the 168.6 s span; 120 synchronizations; 19.2 GB memcpy is the one-time arena
upload. Launch topology: 2,424 commands/token, overwhelmingly single-block.

## Nsight Compute status (hardware counter access)

`ncu` could not collect counters on this host: `ERR_NVGPUCTRPERM — the user does not have permission
to access NVIDIA GPU Performance Counters on the target device 0` (driver restricts profiling to
admin users). Occupancy/SM-utilization/achieved-bandwidth counters are therefore unavailable without
a host permission change. The target selection does not depend on them: the algorithmic defect is
provable directly from source (three Q·K passes; `O(positions · head_dim²)` value pass) and the
launch topology is analytic (incumbent `<<<1,256>>>` with only `query_heads`=24 active threads ⇒ 1 warp
on 1 of 148 SMs). The fix and its correctness argument are structural, not counter-derived.

## Explicit answers

1. **Largest single-kernel opportunity:** `grouped_attention_bf16_cache` at 48.8%. It is launched
   `<<<1,256>>>` but only `query_heads` (24) threads do work; it performs three full passes over the
   KV window, and its value pass is `O(positions · head_dim²)` because it recomputes the Q·K score
   inside the per-dimension loop.
2. **Largest E2E opportunity:** the same kernel. Its share is dominant and its algorithmic headroom is
   ~`head_dim` (256×) plus two redundant score passes.
3. **Same?** Yes. This is the rare case where the largest kernel is also the largest E2E lever.
4. **Has launch overhead / fragmentation become first-order?** **No.** Device idle is 0.10%;
   145k kernels overlap CPU enqueue because each kernel is long. Launch-count reduction (fusion of
   the ~2% elementwise tail) cannot reach the >15% E2E threshold. Not selected.
5. **Would fusion/persistent execution beat optimizing GDN in isolation?** Neither is the best move.
   GDN is now 14.4%, not the ~27% seen in the 3-token profile; and the fused elementwise tail is ~0.3%.
   The dominant, context-growing kernel is grouped attention.

## Selected target

**`grouped_attention_bf16_cache`** (`backends/sm120/runtime/cuda_plan_executor.cuh:160`, launched at
`:888`). Optimize exactly this. Do not touch NVFP4, GDN, or fusion in this step.

See `.planning/phases/S04-kernel-portfolio/S04-P1-PLAN.md`.
