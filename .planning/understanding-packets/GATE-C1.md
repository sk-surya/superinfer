# Gate C.1 — Dense NVFP4: CUDA cores vs SM120 Tensor Cores

## 1. What changed

SuperInfer's decode path for dense NVFP4 projections moved from a CUDA-core GEMV (per-element E2M1
decode, block-scale multiply, FP32 FMA) to the **SM120 warp-level block-scaled tensor-core MMA**
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`.
The hardware contract is now pinned end to end: fragment layouts, scale ownership, E2M1 codes, and the
UE4M3 scale encoding. A synthetic differential passes exactly, and a full warp-level GEMV on the real
Qwen3.8 shapes measures the E2E cost.

## 2. Why it exists

The prior classification-D claimed SM120 has no block-scaled NVFP4 `mma.sync`. That was a **probe bug**:
the type string omitted the mandatory trailing scale type (`.ue4m3`), so `ptxas` rejected the whole
`.block_scale`/`.kind::mxf4nvf4`/`.scale_vec::4X` modifier set with misleading "illegal modifier"
errors. The false negative mattered: it closed the only path from an instruction-issue-bound GEMV
(~3 instructions/weight) to the memory floor.

## 3. One execution path

For one `m16n8k64` tile of a projection `[rows, K]`:

weights `packed[rows][K/2]` + `scales[rows][K/16]` (E4M3FN) + activation quantised on device
→ per-lane A fragments (4×`.b32`), SFA (lane `4g`→row `g`, lane `4g+1`→row `g+8`), B fragments
(lane `(g,q)` owns column `g`, K `8q..` and `32+8q..`), SFB (lane `4n`→column `n`)
→ K-loop of MMAs accumulating `D` (`c` must be the running accumulator, not zero)
→ `SM80_16x8_Row` epilogue: `c0=(g,2q)`, `c2=(g+8,2q)`, × tensor scale.

## 4. Important data structures

- Weight `packed`/`scales`: SuperInfer's existing row-major NVFP4 layout; the `scales` bytes are used
  **directly** as the `.ue4m3` operand because both are IEEE E4M3 with bias 7.
- A/B fragments: 4 and 2 `.b32` registers; C/D: 4 `.f32`.
- `{byte-id, thread-id}` selectors must be `{0,0}` for `scale_vec::4X` (PTX 9.7.16.3, Table 46).
- Activation quantisation: block-16 `amax/6` → UE4M3 scale + E2M1 codes (a **new numerical mechanism**).

## 5. Core invariants

- The MMA is natively M16N8K64: N=1 and N=8 issue the **identical** instruction, so batching/speculative
  decoding is ~8× free; N=1 only wastes output columns, not throughput.
- Correctness of the fragment/scale contract is proven by exact (`rel=0`) differentials, not by tolerance.
- The activation-quantisation error is a separate, measured mechanism and must be judged by D-021.
- The native path is a spike: no production promotion without a provider/layout architecture decision.

## 6. Performance model

Weight streaming dominates. CUDA-core GEMV is instruction-issue bound (~83 ms/token at ~3 inst/weight);
the tensor-core path issues ~1 MMA per 512 weight bytes and reaches memory bandwidth. Measured
per-token (multiplicity-weighted over the **derived 401-launch census**): Arm A N=1 MMA path
**15.10 ms/token**, activation quantisation **2.97 ms/token**, `native_unfused_total` **18.07 ms/token**
(repacked 15.10). This is a projection-subsystem figure, not model throughput; the PREDICTION-ONLY
whole-model bound with the P7 residual is ~16 tok/s. (An earlier draft of this packet quoted partial
census figures of 81-98 tok/s as model throughput; that was a census/accounting error, corrected here.)
Small-M shapes are parallelism-limited, not bandwidth-limited.

## 7. Likely failure modes

- `c` hard-wired to zero: the K-loop overwrites instead of accumulates (silent, shape-dependent).
- Packing the B fragment with N-contiguous instead of K-contiguous elements (8 columns instead of 8 K).
- Wrong scale type (missing `.ue4m3`) → the misleading `ptxas` "illegal modifier" cascade.
- Wrong scale supplier lane / byte order → plausible-looking but wrong results.
- Activation quantisation error judged only on synthetic data instead of D-021.

## 8. Exactly three files to read

1. `tests/gpu/sm120/nvfp4_mma_bench.cu`
2. `backends/sm120/runtime/cuda_plan_executor.cuh`
3. `.planning/phases/S04-kernel-portfolio/S04-P8/S04-P8R-CONTRACT.md`

## 9. Hands-on experiment

In `nvfp4_mma_bench.cu`, change the MMA helper's `c` operands back to `0.f` and run the `small_1024x5120`
shape. Prediction: the Arm A output collapses toward a single K-block's contribution (`rel` vs the FP32
reference becomes large, ~1.0) while the kernel still runs and produces finite numbers. Restore the
accumulator feedback after observing it.

## 10. Five questions

1. Why is the same instruction correct for both N=1 and N=8, and what exactly is wasted at N=1?
2. Why must the B fragment's eight elements span K rather than N for this instruction shape?
3. Why does the missing `.ue4m3` produce errors about `.block_scale` rather than about the type list?
4. Why can a 635 MB projection run at ~1.1 TB/s with a repacked layout but only ~0.5 TB/s without?
5. Which single measured quantity decides classification A versus B, and where is its evidence stored?
