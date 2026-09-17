# S04-P6 Summary — NVFP4 GEMV Architecture: Warp-per-Row Wins

**Status: PASS (promoted).** Seventh profiler-driven loop. Candidate B (warp-per-output-row) is the winner.

## Roofline (checked)

Per-token NVFP4 traffic 12.80 GB packed + 1.60 GB scales = **14.40 GB**; RTX 5090 DRAM roofline
1.79 TB/s ⇒ **8.0 ms/token**. Pre-P6 measured NVFP4 ≈ **144.8 ms/token** (sum over the 401 launches/token)
⇒ **~5.5% of roofline (~18× off)**. Nsight Compute counters unavailable (`RmProfilingAdminOnly=1`, no
passwordless sudo), so achieved bandwidth was measured directly with CUDA events
(`tests/gpu/sm120/nvfp4_gemv_bench.cu`); see `S04-P6A-ROOFLINE.md`.

## Diagnosis

The incumbent's per-launch time is nearly **constant (~0.25 ms) from 2.6 MB to 44.6 MB** — a fixed
**per-row decode-latency floor** (one thread serially decodes a full row), not a bandwidth limit.
Interleaving the layout (A) improves coalescing but yields only ~1.3×, confirming bandwidth was not binding.

## Candidates (measured, `S04-P6A-ROOFLINE.md`)

| | local (weighted ms/token) | local speedup | correctness |
|---|---:|---:|---|
| incumbent | 144.8 | 1.0 | bit-exact |
| A — 32-row interleaved packed+scale layout | 134.0 | 1.08× | **bit-exact** |
| B — warp-per-output-row | **87.0** | **1.66×** | tolerance-qualified (per-projection max_abs ≈ 1e-7) |

A is bit-exact but below the promotion threshold and not "comparable performance"; B is materially
faster. Per the P6 selection rule, B is the winner; A is retained as the documented bit-exact alternative.

## Winner: B — `nvfp4_linear_warp_f32`

One warp per output row; each lane reduces a contiguous column range; warp tree reduction. Global
weight loads become naturally coalesced (lanes read contiguous column groups of one row).

## Local speedup

NVFP4 weighted **144.8 → 87.0 ms/token (1.66×)**. Note B regresses the single LM-head launch in the
microbenchmark (1.35 → 4.06 ms) but wins 2–10× everywhere else; net strongly positive. A future
shape-adaptive selection could recover the LM-head case if it proves material in the real profile.

## E2E tok/s before -> after

| Metric | P5 | P6 (B) session 1 | P6 (B) session 2 |
|---|---:|---:|---:|
| chat-60 wall | 39.90 s | 36.95 s | 38.70 s |
| long-103 wall | 48.17 s | 42.06 s | 42.28 s |
| **GPU kernel total / 60 tok** | 11.54 s | **7.62 s** | — |
| decode throughput (GPU-bound) | ~5.2 tok/s | **~7.9–8.4 tok/s** | ~reproduced |

The most robust decode metric is the nsys GPU kernel total (excludes the ~25–30 s fixed process
load): **11.54 → 7.62 s / 60 tokens = 1.51×**, with NVFP4 in-context **9.27 → 5.35 s (1.73×)**. Wall
marginal is noisier (load variation); both sessions are faster than P5 on both cases.
Cumulative vs R01: 0.032 → ~8 tok/s (**~250×**).

## Correctness

- **D-021 verdict pass** with candidate B: 240 strict rows greedy-exact, 12 ties in-set, 66 listed
  outliers, distributional bounds clear.
- Local differential (`test_nvfp4_warp_tolerance`): warp vs incumbent max difference ≤ 1e-4 × magnitude
  (observed ~1e-7).
- **Default promotion verified**: with B as the default (no env override), D-021 still **pass** (240 strict,
  12 ties). The bit-exact incumbent remains selectable via `SUPERINFER_QWEN38_NVFP4_WARP=0`.
- `python tools/validate.py --full` green.
- The layer/GDN artifact C++ tests require `SUPERINFER_QWEN38_ARTIFACT`/`..._REFERENCE_F32` fixtures that
  are not wired into CI (they SKIP with return 77); B's per-projection error ~1e-7 is two orders of
  magnitude below their 2e-2/2e-4 thresholds. Model-level D-021 is the binding gate and passes.
- Because B changes FP32 reduction order, model logits are perturbed (chat-60 max_abs 0.86 vs P5) purely
  by 64-layer amplification of a ~1e-7 per-projection change; this is expected and quantified, not a
  same-binary nondeterminism signal.
- B is a deterministic arithmetic-order change, not a bit-exact successor; the retained bit-exact
  incumbent (env fallback) is the rollback path.

## Storage / materialization cost

B needs **no repack and no extra storage** — it uses the existing row-major layout. (A would have needed
a 14.4 GB repack or in-place permutation; avoided.)

## Determinism

No recurrence of the P5 anomaly. Candidate B's difference vs the incumbent is a deterministic
arithmetic-order effect (local max_abs ≈ 1e-7), distinguishable from run-to-run nondeterminism. Within
the same binary the captures are repeatable (session 1 and 2 give the same ~8 tok/s regime).

## New profile (post-P6, 60 tokens, 7.62 s GPU)

| Kernel | share |
|---|---:|
| `nvfp4_linear_warp_f32` | 70.2% |
| `linear_f32` | 14.5% |
| `rms_norm_f32_bf16_scale_parallel` | 4.1% |
| `causal_conv_silu_f32` | 2.9% |
| `gated_delta_attention_parallel_f32` | 2.6% |
| everything else | <2% |

## Next bottleneck

NVFP4 remains #1 at 70.2% and is still ~11× off its 8.0 ms/token roofline (5.35 s / 60 = 89 ms/token),
so another tightly justified NVFP4 loop is allowed; `linear_f32` (14.5%) is second. Selection from the
fresh profile, not assumption.
