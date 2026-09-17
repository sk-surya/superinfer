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
