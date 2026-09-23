# E0b — Fused Decode Roles

## Purpose

After E0a proves mature projection arithmetic inside SuperInfer, remove command/materialization waste using fusion patterns already demonstrated by mature engines.

This is not a speculative megakernel effort.

## Rule

Only implement fusion with a known semantics-preserving donor pattern or a trivial pointwise epilogue.

No whole-layer persistent kernel in E0b.

## Priority 1 — gate/up/SwiGLU

Donor:

- NInfer src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu

Target transformation:

    normed_x
      -> gate projection
      -> up projection
      -> silu_mul
      -> hidden

becomes a fused projection with paired/interleaved gate/up rows and SwiGLU epilogue.

Requirements:

- offline interleaved row layout;
- one command replacing the three-command role where lowering permits;
- same visible BF16/FP32 boundary expected by down projection;
- differential against unfused exact-recipe output.

## Priority 2 — down + residual

Donor:

- NInfer src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu

Target transformation:

    hidden
      -> down projection
      -> residual add

becomes projection with residual epilogue and in-place/owned residual update where legal.

Requirements:

- explicit alias/ownership proof;
- no hidden synchronization;
- retained fallback;
- differential before promotion.

## Priority 3 — GDN norm + control projection + gate parameters

Donor blueprint:

- NInfer src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu

Current waste includes RMSNorm plus two tiny control linears plus separate gate parameter work.

The donor establishes a powerful algebraic fact for this model: the normalization scalar can be applied after the control dot products while the norm is accumulated concurrently, and normalized h can be emitted without a separate norm pass.

Build a dedicated fused physical command through the existing lowering surfaces.

Required outputs:

- normalized h at the exact boundary required by the main GDN projection;
- log-decay/g;
- beta;
- any control intermediates only if downstream semantics require them.

Do not keep dead materializations for historical graph shape.

## Priority 4 — projection output routing

Attention and GDN input projections often immediately split/route concatenated outputs.

Use donor epilogues to write directly to final q/k/v/z/gate buffers when this is purely an output-layout transformation.

Do not introduce extra memcpy/split kernels for a representation the projection can emit directly.

## Physical Plan policy

E0b is allowed to add fused command types and lowering rules.

It must still preserve:

- model-agnostic executor;
- compile-time specialization;
- no runtime model-name branch;
- static workspace ownership;
- deterministic plan validation.

The executor should receive pre-bound launch records; it should not discover fusion.

## Launch-count objective

Current: approximately 2,424 launches/token.

E0b target is not an arbitrary magic count, but the fused role set should eliminate hundreds of cast/elementwise/split launches.

Record before/after launch count.

A rough medium-term target is <=500 captured launches/token before speculation work. Do not delay E0b merely to hit exactly 500.

## Performance gate

After E0b:

- **survival gate:** <=40 ms/token device span;
- **target:** 30–35 ms/token.

If 40–49 ms:

- one 12-hour diagnosis window;
- identify one measured cause with enough magnitude to close the gate;
- implement that fix immediately.

If >=50 ms with valid donor kernels:

- do not start another kernel research loop;
- prepare runtime/data-plane pivot.

## Correctness

Fusion should preserve the chosen arithmetic recipe.

Required:

- unfused-vs-fused local differential;
- layer/GDN fixture;
- D-021 smoke;
- fresh-process repeatability for the integrated result.

Do not use model-level acceptance to hide an incorrect fused implementation.

## Completion output

Return:

    BASE_SHA
    FINAL_SHA
    e0a_ms_per_token
    e0b_ms_per_token
    e0b_tok_per_s
    launches_per_token_before
    launches_per_token_after
    projection_subtotal_ms
    top_10_kernels
    correctness_matrix
    fallbacks_remaining
    blocker_or_next_bottleneck
