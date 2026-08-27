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

## Debugging record

The initial full-graph differential diverged at layer 3, the first full-attention layer. The
Qwen `q_proj` layout is `[query(256), gate(256)]` per attention head. The lowering had flattened
the split as `[all_query, all_gate]`, causing every other QNorm row to consume gate data. The
specializer now emits the authored `24 x 256 x 256` split for real shapes, retaining a one-row
fallback only for tiny structural fixtures. A missing final RMSNorm and missing Q/K
`one_plus_weight` convention were also corrected in this round.

## Next boundary

Add decode-position/state-continuation execution for at least two token segments, compare selected
intermediates and final logits against the external reference, then add deterministic repeatability,
capacity rejection, corruption/failure, and acceptance-report evidence.
