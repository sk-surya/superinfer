# S04-P8R — Hardware-Contract Correction (supersedes the P8 classification-D closure)

**Status:** P8 reopened. The earlier classification **D** is **invalidated**: it was based on a malformed
PTX probe, not on the actual `sm_120a` contract.

## P8-R0 — false-negative root cause

Two probe sources are preserved under `artifacts/S04/p8r/`.

### What P8 actually assembled (wrong)

`p8r0_false_negative_old_probe.ptx`:

```ptx
.version 8.8
.target sm_120a
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32
    {c0..c7}, {a0..a7}, {b0..b3}, {c0..c7};
```

ptxas errors (verbatim):

```
error : Illegal modifier '.block_scale' for instruction 'mma'
error : Illegal modifier '.kind::mxf4nvf4' for instruction 'mma'
error : Incorrect instruction type specified for mma with shape '.m16n8k64'
```

A separate probe of the non-block-scaled fp4 path used `.kind::f8f6f4` and produced
`.kind::f8f6f4 modifier required for instruction 'mma'`. That message was about a different (non
block-scaled) instruction and was mis-generalised into "block-scale is tcgen05-only". The
`tcgen05.alloc` probe only showed that the 5th-gen path is absent, which is true but irrelevant: the
warp-level `mma.sync` path is the relevant one on `sm_120a`.

### The correct documented form

`p8r1_mma.ptx`:

```ptx
.version 9.1
.target sm_120a
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.
    f32.e2m1.e2m1.f32.ue4m3
    {%Rd0,%Rd1,%Rd2,%Rd3}, {%Ra0,%Ra1,%Ra2,%Ra3}, {%Rb0,%Rb1}, {%Rc0,%Rc1,%Rc2,%Rc3},
    %scaleAData, {%bidA, %tidA}, %scaleBData, {%bidB, %tidB};
```

### Character-for-character differences (the actual causes)

1. **Missing `.ue4m3` scale type on the type string.** The old form ended `.f32`; the block-scaled
   type string requires a trailing `.{stype}` (`.ue4m3` for NVFP4). With an incomplete type string,
   ptxas rejected `.block_scale` and `.kind::mxf4nvf4` as *illegal modifiers* rather than as a
   missing-operand error — which is what made the failure look like a hardware-contract absence.
2. **Missing `scaleA, {byte-id-A, thread-id-A}, scaleB, {byte-id-B, thread-id-B}` operands.** Once the
   type string is complete, omitting these yields `Arguments mismatch`.
3. **Wrong operand counts** for `.e2m1`: m16n8k64 e2m1 requires A = 4×`.b32`, B = 2×`.b32`,
   C/D = 4×`.f32` (not 8/4/8).
4. PTX version was **not** the cause: `.version 8.8` also assembles the corrected form
   (`p8r1_v88.ptx`). Version 8.7+ supports `.block_scale`.

## P8-R1 — ptxas + GPU execution proof

- `ptxas --version`: CUDA 13.1, V13.1.115 (`artifacts/S04/p8r/ptxas-version.txt`).
- Command: `/usr/local/cuda/bin/ptxas -arch=sm_120a artifacts/S04/p8r/p8r1_mma.ptx -o artifacts/S04/p8r/p8r1_mma.cubin`
- Exit status: **0**; cubin sha256 `65e3ddf7737d3f88c3e2d1b60048bbd49c02345f4c46a3febbe36064fa2f29d1`.
- PTX sha256 `58ebc0e7bcf64cafbe73f535abca9eef2ef16cea7ce1af9f63c68da1fb208f1d`.
- **GPU execution** (`p8r1_exec.cu`): the instruction ran on the physical RTX 5090 (`sm_120`).
  With A = B = all E2M1 1.5 (`0x33` bytes) and four UE4M3 1.0 scales, lane-0 `D = {144,144,144,144}`,
  exactly `64 × 1.5 × 1.5 = 144`. Non-finite values: none.

So: `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
**assembles and executes correctly on `sm_120a`**, with E2M1×E2M1 operands, UE4M3 block scales
(vector size 16), and FP32 accumulate.

## P8-R2 — synthetic fragment differential (IN PROGRESS)

`artifacts/S04/p8r/p8r2_diff.cu` runs one randomized m16n8k64 blockscaled MMA and compares it to an
independent FP32 reconstruction. The uniform case is exact (`p8r1_gpu_execution.txt`: A=B=1.5, scales=1.0
→ `D=144 = 64×1.5×1.5`), which proves operands, scales and accumulate are wired correctly.

The **randomized** differential currently fails (`p8r2_differential_initial_result.txt`:
`max_abs=22.45`, `max_mag=14.16`, `rel=1.585`). A `rel≈1.6` error (not garbage) indicates the A-fragment
row/K mapping is broadly right but at least one of the B-fragment `(K,N)` mapping or the SFA/SFB
thread/byte ownership is wrong. The exact TV layouts are documented in
`cute/atom/mma_traits_sm120.hpp` (`((2,2,8),(16,4)):((32,0,4),(0,1))` for SFA and
`((4,8),(16,4)):((0,4),(0,1))` for SFB) and in the PTX warp-level-matrix-fragment section.

**Consequence:** Arms A/B/C and the final A/B/C/D classification are **not yet evaluated** — they require
the verified fragment/scale mapping first. No performance numbers are reported for them.

## P8-R2 calibration findings (empirical, on the RTX 5090)

`artifacts/S04/p8r/p8r2_calibration.txt` (from `p8r2_calib.cu`, `p8r2_scale.cu`):

1. **E2M1 codes are the standard NVFP4 values** `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` — verified by the
   code-to-code ratios (e.g. D(7)/D(1) = 12, D(3)/D(1) = 3).
2. **The UE4M3 block scale does NOT follow IEEE E4M3 bias 7.** Measured: `s = 2^(e-6)·(1+m/8)`, i.e.
   exactly **2× the bias-7 value** (0x38 → 2.0, not 1.0). Using the bias-7 host decode was a second, real
   source of differential error.
3. **A-fragment row/K mapping corrected** to the PTX spec (section 9.7.16.5.11): `a0`→(row g, k 8q..8q+7),
   `a1`→(row g+8, same k), `a2`→(row g, k 32+8q..), `a3`→(row g+8, same k). B is `b0`→(k 8q.., n g),
   `b1`→(k 32+8q.., n g); C/D is `c0`→(g,2q), `c1`→(g,2q+1), `c2`→(g+8,2q), `c3`→(g+8,2q+1).

Remaining before the randomized differential passes: the **SFA/SFB thread/byte ownership** for
`scale_vec::4X` (which lane/byte supplies which (row, k-block) scale; PTX selectors `{byte-id, thread-id}`
plus the quad-broadcast layout from CUTLASS `mma_traits_sm120.hpp`). My current single-b32-per-lane
construction is not yet the hardware layout. Arms A/B/C and the final classification remain gated on this.

## P8-R2 status — SFA supplier mapping unresolved (hard blocker)

Authoritative layouts obtained from CUTLASS `mma_traits_sm120.hpp` (`SM120_16x8x64_TN_VS`):
`ALayout ((4,8),(8,2,2)):((128,1),(16,8,512))`, `BLayout ((4,8),(8,2)):((64,1),(8,256))`,
`SFALayout ((2,2,8),64):((8,0,1),16)`, `SFBLayout ((4,8),64):((0,1),8)`, `CLayout SM80_16x8_Row`.

A/B/C mappings are now settled and match the PTX fragment spec. E2M1 codes are standard; the UE4M3
scale uses `s = 2^(e-6)·(1+m/8)` (bias 6, 2× IEEE bias-7).

`p8r2_sfa_probe.txt` shows the remaining inconsistency: with `sfa byte0 = 0x38|(lane&7)`, `sb=0x38`,
and a single nonzero A at `(m=0,k=0)`, the measured SFA values are `m0→2.0 (byte 0x38)`,
`m1→2.5 (byte 0x3A)`, `m2→2.0 (byte 0x38)`. That does **not** match the derived supplier
`m = 8*(L%2) + L/4` predicted by SFALayout, so either the `{byte-id, thread-id}` selection or the
byte-within-`b32` convention (which of the 4 UE4M3 bytes is K-block 0) still differs. Until this is
resolved the randomized P8-R2 differential fails (`rel≈1.08`) and **Arms A/B/C remain unrun**.

Next step for the next session: determine the byte/selector convention empirically with a probe that
varies `{byte-id, thread-id}` and the byte index independently, then re-run the differential.

---

# P8-R0 / R1 / R2 RESOLVED

## P8-R0 — false-negative root cause (reproduced, `artifacts/S04/p8r/p8r2_rootcause.txt`)

Running the exact old probe through the same `ptxas` reproduces the error:

```
ptxas p8r0_false_negative_old_probe.ptx, line 4; error : Illegal modifier '.block_scale' for instruction 'mma'
ptxas ... error : Illegal modifier '.kind::mxf4nvf4' for instruction 'mma'
ptxas ... error : Incorrect instruction type specified for mma with shape '.m16n8k64'
ptxas ... error : Illegal modifier '.scale_vec::4X' for instruction 'mma'
ptxas fatal : Ptx assembly aborted due to errors            (exit 255)
```

Character-for-character, the old probe's type string was:

```
...m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32
```

The documented form appends the **scale operand type** (`.stype`):

```
...m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
```

Findings, in order of causality:

1. **PRIMARY — wrong/missing scale type.** The `.stype` qualifier (`.ue4m3`) was absent. Without it the
   trailing type sequence `.f32.e2m1.e2m1.f32` matches no legal block-scaled `mma` combination, so `ptxas`
   cannot resolve the variant and emits the misleading cascade of "illegal modifier" errors for
   `.block_scale`, `.kind::mxf4nvf4`, and `.scale_vec::4X`. `.kind` and `.scale_vec::4X` were in fact correct.
2. **SECONDARY — wrong operand list.** The old probe declared A as 8 `.b32` registers and B as 4, and omitted
   the `scale-a-data, {byte-id-a,thread-id-a}, scale-b-data, {byte-id-b,thread-id-b}` operands entirely. The
   correct fragment sizes are A = 4 `.b32`, B = 2 `.b32`, C/D = 4 `.f32`.
3. **NOT a cause — PTX version.** `.version 8.8` also assembles (`p8r1_v88.ptx`, exit 0). The toolchain is
   CUDA 13.1 / ptxas V13.1.115, PTX ISA 9.4 documented.

## P8-R1 — minimal ptxas + GPU proof (existing evidence, unchanged)

`p8r1_mma.ptx` (`.version 9.1`, `.target sm_120a`) assembles exit 0; cubin sha256
`65e3ddf7...f29d1`; executed on the RTX 5090 giving lane-0 `D={144,144,144,144}` for A=B=1.5 over k=64.

## P8-R2 — synthetic differential now passes exactly

`artifacts/S04/p8r/p8r2_diff.cu` (v3) compares one `m16n8k64` block-scaled MMA against an independent FP32
reference `sum_k (e2m1_A * scale_A[m,k/16]) * (e2m1_B * scale_B[k/16,n])`, over random E2M1 data:

```
P8R2 v3 (uniform scales): max_abs=0 max_mag=148      rel=0  PASS
P8R2 v3 (random  scales): max_abs=0 max_mag=252.967  rel=0  PASS
```

### Pinned hardware contract (all verified on-device)

- **A** `(T32,V32)->(M16,K64)`: reg `r=(v/8)`, row `g+8*(r&1)`, `k = 8*q + (v%8) + 32*(r/2)`.
- **B** `(T32,V16)->(K64,N8)`: reg `r=(v/8)`, `k = 8*q + (v%8) + 32*r`, col `n = g`. Elements span **K**
  (stride 8 in a `[k*8+n]` array) — packing consecutive N values is wrong.
- **SFA** `(T32,V64)->(M16,K64)`: lane `4g` supplies row `g`; lane `4g+1` supplies row `g+8`; the four bytes
  of the lane's `.b32` are K-blocks 0..3.
- **SFB** `(T32,V64)->(N8,K64)`: lane `4n` supplies column `n`; four bytes are K-blocks 0..3.
- **C/D** `SM80_16x8_Row`: `c0=(g,2q)`, `c1=(g,2q+1)`, `c2=(g+8,2q)`, `c3=(g+8,2q+1)`.
- Selectors `{byte-id,thread-id}` must be `{0,0}` for `.scale_vec::4X` (PTX 9.7.16.3, Table 46).
- **E2M1** codes are the standard NVFP4 magnitudes `{0,0.5,1,1.5,2,3,4,6}` with the sign bit at 0x8.
- **UE4M3** is IEEE E4M3, **bias 7** (`0x38 = 1.0`) — the earlier "bias 6" note was wrong; it came from a
  buggy calibration probe. `p8r2_sfprobe2`, `p8r2_map`, `p8r2_kmap` confirm the supplier lanes and the K map.

The fragment/B-packing discovery (`p8r2_kmap.cu`) is the reason the earlier differential failed; the SFA
"supplier mystery" was a red herring caused by a probe that stimulated every lane.

---

# P8-R3 / R4 / R5 — Arm A (N=1), Arm B (N=8), Arm C (minimal warp MMA)

Evidence: `tests/gpu/sm120/nvfp4_mma_bench.cu`, `artifacts/S04/p8r/p8r3_arm_abc_result.txt`.
Every number below is a full warp-level path on the real projection shapes and multiplicities
(`1,128,48,48,64,32` for `lm_head, mlp, gdn, attn, down, small`), not MMA-only timing.

## Correctness first

Arm A output equals an independent FP32 reconstruction over the *same quantised operands* exactly
(`max_abs = 0`, `rel = 0.0`) on all six shapes. (Fixing the earlier bench required feeding the running
accumulator back into the `c` operand — with `c = 0` the K-loop overwrote instead of accumulated.)

## Measured cost (ms per shape; weighted = ms per decoded token over multiplicities)

| shape | wt MB | quant | stream | ArmA N=1 | ArmB /8 | A2 N=1 | A2 N=8 /8 |
|---|---|---|---|---|---|---|---|
| lm_head_248320x5120 | 635.7 | 0.007 | 0.004 | 1.229 | 0.155 | 0.576 | 0.073 |
| mlp_17408x5120 | 44.6 | 0.007 | 0.002 | 0.027 | 0.004 | 0.022 | 0.003 |
| gdn_10240x5120 | 26.2 | 0.007 | 0.002 | 0.023 | 0.003 | 0.021 | 0.003 |
| attn_6144x5120 | 15.7 | 0.007 | 0.002 | 0.023 | 0.003 | 0.020 | 0.003 |
| down_5120x17408 | 44.6 | 0.009 | 0.002 | 0.073 | 0.010 | 0.066 | 0.009 |
| small_1024x5120 | 2.6 | 0.007 | 0.001 | 0.023 | 0.003 | 0.020 | 0.003 |
| **weighted / token** | | **2.36** | | **12.27** | **1.69** | **10.24** | **1.37** |

- **Arm A (N=1, no weight repack): 12.27 ms/token ≈ 81 tok/s** — ~10x the promoted software path.
- **Arm A2 (N=1, one-time repack to an MMA-native weight layout): 10.24 ms/token ≈ 98 tok/s**;
  `lm_head` reaches ~1.10 TB/s, i.e. memory-bound.
- **Arm B (N=8 genuine activations, same single MMA per K-step): 1.69 ms/token ≈ 591 tok/s**
  (1.37 ms ≈ 728 tok/s repacked). Because the instruction is natively M16N8K64, N=1 and N=8 issue the
  identical instruction; batching/speculative decoding therefore gets ~8x almost for free.
- Arm C is the bare loop: the only cost Arm A adds over Arm C is the activation quantisation
  (`quant`, 0.007 ms per 5120-wide projection — launch-bound at 321 calls/token = 2.36 ms; fusable).
- The small-M shapes (`down` 320 warps, `small` 64 warps) are parallelism-limited, not bandwidth-limited;
  splitting K across warps is the obvious follow-up.

## Activation quantisation (the decisive quality risk)

`amax/6` block-16 E2M1 + UE4M3, applied to the activation, changes each projection output by
**rel-L2 ≈ 0.10** on synthetic uniform activations (10.06%, 10.08%, 9.94%, 10.24%, 10.15%, 9.98%).
The prior real-hidden-vector experiment (`artifacts/S04/p8-activation-quantization.json`) measured
2.44-9.30% rel-L2 for the activation itself. This is the new numerical mechanism and the one thing that
can turn a performance win into classification B. It is **not** a stop condition; it must be measured
against D-021, and D-021 must not be loosened.

## Classification

- Performance axis: **A** — N=1 native NVFP4 is performant (~81-98 tok/s single-stream, ~10-12x the
  current 8.3 tok/s), and N=8 lifts it to ~591-728 tok/s, so the path is *not* merely a batching
  (C) or non-competitive (D) result.
- Quality axis: **B risk open**. The activation-quantisation contract has not yet been validated against
  D-021 on the real model. Classification A is provisional on that gate.
- The earlier classification D is withdrawn (invalidated by the corrected hardware contract).

## Next action (requires a separate provider/layout architecture decision — do NOT auto-promote)

1. Add the native MMA path behind an experimental selector in `cuda_plan_executor.cuh`, retaining the P7
   software path as the fallback oracle.
2. Run the projection-level native-vs-P7 differential, the layer-3 differential, the GDN differential,
   then the full D-021 corpus, long-103, same-binary repeatability, and a second fresh session.
3. Decide A vs B from that evidence, then the provider/layout architecture decision.
