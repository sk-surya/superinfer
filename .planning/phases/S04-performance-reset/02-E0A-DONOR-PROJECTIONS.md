# E0a — Donor Projection Backend

## Purpose

Replace the 101+ ms/token projection subsystem with mature decode kernels while preserving Physical Plan command topology.

This is the highest-value production implementation step. It is not a benchmark study.

## Scope

### Replace

- all 401 packed NVFP4 projection launches/token;
- the 96 GDN control projections currently accounted under linear_f32;
- per-shape dispatch/layout preparation needed by those replacements.

### Preserve

- operation ordering;
- current attention;
- current GDN recurrence;
- current executor;
- current residual/SwiGLU command boundaries;
- current token feedback harness for the first integrated timing;
- P7 fallback/oracle.

## Major principle: adapt donor kernels to current activation ownership

Current SuperInfer command boundaries expose FP32 activation buffers.

Do NOT implement:

    FP32 activation
       -> standalone BF16 cast kernel
       -> donor GEMV

401 times/token.

Instead adapt the donor path so the hot GEMV loads FP32 activation values and converts to BF16/float pairs in registers/shared memory as required by the arithmetic contract. If a donor schedule fundamentally requires a compact BF16 activation tile, materialize it within the same kernel/CTA launch.

The activation bytes are small relative to streamed weights. Launch proliferation is not acceptable.

## Donor baseline

Primary donor:

NInfer at the pinned source used in the master review.

Study first:

- src/ops/linear/nvfp4/nvfp4_gemv.cuh
- src/ops/linear/nvfp4/nvfp4_config.h
- geometry-specific shape files under src/ops/linear/nvfp4/shapes/
- src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
- src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu

Important donor properties already established:

- compile-time Geometry/Schedule;
- multiple rows per warp;
- multiple accumulator chains;
- vector code loads;
- ld.global.cg option;
- staged/swizzled scale access;
- exact 128-row parent layout;
- BF16 activation input;
- FP32 accumulation.

Secondary independent comparator:

FlashInfer SM12x BF16×FP4 GEMV / CuTe DSL weight preparation.

Do not spend time embedding two donor stacks. Benchmark/select quickly, then integrate one.

## Weight/layout strategy

Current .sinf stores source/checkpoint layout. Do not force the runtime kernel to compensate indefinitely.

Preferred sequence:

1. Define one prepared-layout descriptor behind StoragePolicy.
2. Repack once during conversion or artifact/load preparation.
3. Preserve original source tensors/checksums/provenance.
4. Store or cache prepared bytes deterministically.
5. Bind prepared pointers in the Physical Plan before generation.
6. No per-token repack.

E0a may add a layout section/version if the existing artifact mechanism already supports an additive prepared representation cleanly. Do not redesign .sinf globally.

## Shape census

Use tools/qwen38_nvfp4_census.py as the source of truth for the 401 launches and exact geometries.

The agent must produce a table:

    role
    N
    K
    launches/token
    bytes/launch
    existing kernel
    donor schedule
    donor/prepared layout
    integrated ms/token contribution

Do not assume the six-class historical table is complete.

## GDN control projections

The current 96 linear_f32 launches are a major E0 target.

E0a constraint: no semantic fusion yet.

Implement a high-occupancy specialized small-output projection that consumes the current normalized activation boundary and current control weights.

Allowed:

- new kernel ID/provider;
- prepared weight layout;
- vectorized loads;
- warp/CTA specialization for exact N/K;
- compile-time shape constants.

Not allowed in E0a:

- folding the RMSNorm command into this kernel;
- changing gate semantics;
- removing/reordering commands.

The fused NInfer gdn_norm_gating_27 implementation is the blueprint for E0b, not a reason to skip E0a attribution.

## Correctness

E0a changes implementation, not intended model recipe.

Therefore use same-recipe differentials.

For each shape family:

- compare donor result to retained P7/reference at representative real activation captures;
- report max_abs, mean_abs, rel-L2 and any downstream layer delta;
- do not demand bit identity if reduction order changes;
- do not weaken a threshold to make a bad kernel pass;
- if a discrepancy is caused only by deliberate BF16 activation arithmetic, explicitly classify it as a recipe boundary and compare against an independent CPU/GPU reference of that exact recipe before integration.

Full-model D-021 smoke after all shapes integrate.

## Performance measurement

Microbenchmark only to choose/validate donor integration.

Required anti-cache-hot rule:

- measure representative single-shape cold/streamed behavior;
- AND measure the complete 401-projection traversal or whole-model path.

Never project model tok/s from a matrix that fits in L2.

Report:

- projection subtotal ms/token;
- effective bytes/s;
- percentage of theoretical DRAM bandwidth using documented byte accounting;
- E2E device span;
- donor ratio.

## Fast failure handling

If donor integration is slow because of layout mismatch:

- fix layout;
- do not tune arithmetic.

If a custom adaptation is >10% slower than donor code on the same prepared bytes after one serious pass:

- port the donor more literally;
- preserve required notices/licenses.

If one rare shape resists integration:

- use fallback temporarily;
- land the 80/20 path;
- keep the shape visible in the scoreboard.

Do not block 350+ good projection launches on one awkward class.

## E0a completion

E0a is complete when:

- major projection shapes are donor-backed;
- small control linears are no longer the 18.5 ms/token implementation;
- full-model run succeeds;
- correctness smoke passes;
- integrated timing exists.

Target:

- projection subsystem <=20 ms/token;
- full model <=55 ms/token.

Immediately proceed to E0b.
