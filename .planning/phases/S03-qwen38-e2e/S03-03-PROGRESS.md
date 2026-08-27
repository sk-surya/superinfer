# S03-03 Progress — Qwen3.8 End-to-End Acceptance

## Status

In progress. The first real `.sinf` full-graph token and a statically unrolled three-token prefill
have been executed on GPU 0 and select the same greedy tokens as the independent Transformers
reference. Layer evidence, state continuation, repeatability, and failure behavior are recorded;
broader corpus coverage and final acceptance closure remain open.

## Evidence so far

- Artifact: `build/evidence/qwen38-payload-v1-final-a.sinf`
- Artifact SHA-256: `e25022c8592875449968b9d0b1f56e6800971e0ba04d8a43eec980fe60dc65d5`
- Target: GPU 0, NVIDIA GeForce RTX 5090, `sm_120a`; GPU 1 was not used.
- Full graph: 64 layers, 48 Gated-DeltaNet and 16 full-attention layers.
- Physical plan: 2,424 commands, 19,190,769,152-byte device arena, 128 explicit state buffers.
- Reference: streamed `Qwen3_5TextConfig`/`Qwen3_5DecoderLayer`, Transformers 5.12.1 with
  torch 2.13.0+cu130 in the local offline environment; the pinned source/model identities remain
  recorded by the artifact and the corpus-reference evidence.
- Token 0: SuperInfer greedy `49276`, logit `11.6875`, checksum `-792475`; reference greedy `49276`,
  checksum `-792161.3125`.
- Two-token continuation: SuperInfer greedy sequence `49276, 2349`; reference sequence `49276, 2349`.
- Full FP32 logits and per-step max/mean/RMSE are recorded in
  `artifacts/S03/qwen38-e2e-two-token-differential.json`. Two fresh GPU sessions produced byte-identical
  step captures.
- A three-token shared-state chain `49276, 2349, 1074` also matches the external reference; its
  per-step vector metrics are recorded in `artifacts/S03/qwen38-e2e-three-token-differential.json`.
- A three-token static `prefill` entry point from the real `.sinf` artifact produces all three logit
  rows in one execution, matches the reference greedy sequence, and has byte-identical fresh-session
  captures; metrics and capture hashes are recorded in
  `artifacts/S03/qwen38-e2e-three-token-prefill-differential.json`.
- The corpus `special-tokens` fixture (`248045,846,248046`) also passes the same static-prefill
  differential and fresh-session repeatability check; see
  `artifacts/S03/qwen38-e2e-special-token-prefill-differential.json`.
- The corpus `plain-short` fixture passes six decode-replay positions with matching greedy output
  and byte-identical fresh-session captures; per-position metrics are recorded in
  `artifacts/S03/qwen38-e2e-plain-short-decode-differential.json`.
- The real-artifact capacity guard rejects position `4096` before launch for the configured
  `kv_capacity=4096`; this remains a boundary-rejection check rather than a full near-boundary run.
- A legal position `4095` execution now completes on the same real-artifact plan with finite output,
  128 stable state buffers, and no device-allocation growth; the paired `4096` rejection is recorded
  in `artifacts/S03/qwen38-e2e-kv-boundary.json`.
- The independent tokenizer contract passes the pinned plain, Unicode, special-token, varied-length,
  and rendered chat-template cases; exact IDs, token hashes, tokenizer file hashes, and Transformers
  5.12.1 identity are recorded in `artifacts/S03/qwen38-tokenizer-contract.json`.
- The layer-streamed corpus oracle amortizes checkpoint loading across all five cases and records
  the exact reference environment in `build/evidence/reference-batched-corpus/`. The paired target
  run is intentionally retained as a failed acceptance artifact in
  `artifacts/S03/qwen38-e2e-corpus-acceptance-failure.json`: plain, Unicode, and special-token cases
  pass, while varied-length and chat-template replay are non-repeatable across fresh sessions and
  diverge at long continuation positions.
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
divergence. The qualified corpus reference environment is Transformers 5.12.1 with torch
2.13.0+cu130.

The remaining acceptance work is to resolve the long-replay reproducibility failure, then produce
the reviewed passing report and second complete target-session corpus run. The legal boundary proof
is complete for positions 4095/4096, but the numerical model contract is not yet closed for all
declared corpus cases. S03F-02 remains blocked until that S03 acceptance is explicitly closed.
