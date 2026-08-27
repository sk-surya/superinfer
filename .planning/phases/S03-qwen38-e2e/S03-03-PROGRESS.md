# S03-03 Progress — Qwen3.8 End-to-End Acceptance

## Status

In progress. The first real `.sinf` full-graph token has been executed on GPU 0 and now selects the
same greedy token as the independent Transformers reference. Multi-token state continuation,
repeatability, failure behavior, and the final machine-readable acceptance report remain open.

## Evidence so far

- Artifact: `build/evidence/qwen38-payload-v1-final-a.sinf`
- Artifact SHA-256: `e25022c8592875449968b9d0b1f56e6800971e0ba04d8a43eec980fe60dc65d5`
- Target: GPU 0, NVIDIA GeForce RTX 5090, `sm_120a`; GPU 1 was not used.
- Full graph: 64 layers, 48 Gated-DeltaNet and 16 full-attention layers.
- Physical plan: 2,424 commands, 19,190,769,152-byte device arena, 128 explicit state buffers.
- Reference: streamed `Qwen3_5TextConfig`/`Qwen3_5DecoderLayer`, Transformers 5.14.1 available in
  the local offline environment; the pinned source/model identities remain recorded by the artifact.
- Token 0: SuperInfer greedy `49276`, logit `11.6875`, checksum `-792475`; reference greedy `49276`,
  checksum `-792161.3125`.
- Two-token continuation: SuperInfer greedy sequence `49276, 2349`; reference sequence `49276, 2349`.
- Full FP32 logits and per-step max/mean/RMSE are recorded in
  `artifacts/S03/qwen38-e2e-two-token-differential.json`. Two fresh GPU sessions produced byte-identical
  step captures.
- Failure matrix passes artifact corruption/truncation, target mismatch, injected CUDA fault poisoning,
  and over-capacity continuation rejection; see `artifacts/S03/qwen38-e2e-failure-matrix.json`.

## Debugging record

The initial full-graph differential diverged at layer 3, the first full-attention layer. The
Qwen `q_proj` layout is `[query(256), gate(256)]` per attention head. The lowering had flattened
the split as `[all_query, all_gate]`, causing every other QNorm row to consume gate data. The
specializer now emits the authored `24 x 256 x 256` split for real shapes, retaining a one-row
fallback only for tiny structural fixtures. A missing final RMSNorm and missing Q/K
`one_plus_weight` convention were also corrected in this round.

## Next boundary

The two-token continuation now executes successfully on GPU 0. The test-only position replay path
reuses the validated 2,424-command schedule, mutates only cache append/RoPE/full-attention position
fields, and completes with 4,848 command launches. The greedy sequence is `49276, 2349`, matching
the shared-cache Transformers reference. Full-vector drift is recorded per step in
`artifacts/S03/qwen38-e2e-two-token-differential.json`; it is quantized-logit drift, not a token
divergence. The local reference environment is Transformers 5.14.1 rather than the planned 5.12.1.

The remaining acceptance work is the final reviewed corpus/report packaging and any additional prompt
fixtures required by the plan (chat/Unicode/near-boundary coverage). S03F-02 remains blocked until
that S03 acceptance is explicitly closed.
