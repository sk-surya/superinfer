---
phase: S03-qwen38-e2e
plan: S03-02
status: complete
completed: 2026-08-27
---

# S03-02 — Qwen Lowering, Baseline Coverage, and `.sinf` Integration

## Delivered

- Completed generic lowering/provider coverage for the pinned 64-layer Qwen3.8 graph, including
  BF16 activation conversion, NVFP4 projections with explicit scale sidecars, full-attention gate
  composition, BF16 KV cache, and Gated-DeltaNet convolution/recurrent state.
- Added typed artifact-to-Physical-Plan binding with authenticated payload ranges and fail-closed
  descriptor checks.
- Added liveness-aware activation allocation. Weights and state are persistent; physical buffers
  carry half-open lifetimes so reusable ranges are legal only across non-overlapping commands.
- Added explicit KV-capacity specialization. The first 32-GiB deployment plan uses 4,096 positions;
  the authored 262,144-token maximum is retained in Semantic IR but is not claimed to fit in V0.

## Evidence

- [Full-attention layer differential](../../../artifacts/S03/qwen38-layer3-artifact-differential.json):
  real `.sinf`, RTX 5090 GPU 0, max abs `1.98513e-4`.
- [Gated-DeltaNet continuation differential](../../../artifacts/S03/qwen38-gdn-layer0-artifact-differential.json):
  real `.sinf`, RTX 5090 GPU 0, two segments, max abs `0.0357285`.
- [Complete artifact plan compile](../../../artifacts/S03/qwen38-artifact-plan-compile.json):
  4,823 lowered tensors, 2,421 physical commands, 4,695 physical buffers, 2,001 payload views,
  and 19,190,769,152 device-arena bytes.

## Verification

- Focused CPU specializer, Qwen frontend, and artifact-plan compile tests pass.
- Real artifact integrity, frontend inventory authentication, full specialization, Physical Plan
  verification, and artifact payload binding pass offline.
- Real layer CUDA differentials pass on RTX 5090 GPU 0; GPU 1 was not used.
- Full repository validation is the final pre-S03-03 gate for this plan.

## Boundary

S03-02 proves compilation and layer correctness, not model-token execution. S03-03 owns token input,
prefill/decode state continuation, logits, and fixed greedy-sequence differential evidence.
