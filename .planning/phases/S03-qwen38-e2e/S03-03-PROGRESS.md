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

The first nondeterminism-localization round did not reproduce the historical failure in shorter
prefixes: two fresh 13-token chat sessions and two fresh 20-token chat sessions were byte-identical.
The 13-token prefix was also byte-identical after filling the full device arena with both `0x00` and
`0xA5` before artifact uploads, and after compiling with activation reuse disabled. These controls
therefore do not yet distinguish the long-replay failure; the historical 60-token/81-token evidence
remains valid and S03 remains open. The diagnostic hooks are opt-in only:
`SUPERINFER_QWEN38_ARENA_POISON` fills the device arena before uploads, and
`SUPERINFER_QWEN38_DISABLE_ACTIVATION_REUSE` compiles a no-alias control plan. Neither changes
normal acceptance behavior. Compute Sanitizer is not installed on the qualified host, so
initcheck/memcheck evidence is unavailable until that tooling is provisioned.

A current full 60-token one-token replay completed in 20 minutes with capture hash
`9d588484faff80567136daf86cf31118fa54a2e2da07c9b25e6da1318ff3a99`. An independently instrumented
60-token replay produced the identical logits hash and 7,680 state fingerprints (128 buffers at
each step 0..59); every state buffer changed at step 1. This establishes repeatability for the
current pair, but does not explain the historical captures, which differ from the current run at
rows 12 and 43. The machine-readable localization record is
`artifacts/S03/qwen38-long-replay-localization.json`.

Full-length controls now also match the same hash: arena poison `0xA5` and activation-reuse-disabled
physical planning both produce `9d588484...` with no differing row. Initial arena contents and
activation liveness reuse are therefore ruled out for the current replay path. The remaining
diagnostic boundary is replay/state-command localization, followed by a fresh standard acceptance
rerun; the historical non-repeatability and Transformers divergence still prevent S03 closure.

A second fully standard fresh 60-token session also produced `9d588484...` with all rows equal to
the first standard session. The current replay is therefore reproducible across two standard
sessions and the two diagnostic controls; the historical failure is not presently reproducible.
The current capture matches all 60 reference greedy tokens, but four rows exceed the pinned `max_abs`
logit contract (first at row 23, maximum `0.6116333`), so this is not acceptance closure.

An explicit reference diagnostic now models the target's BF16 KV storage by quantizing every
Transformers DynamicCache update to BF16 and reading it back as FP32. Against the stable current
target capture, the original FP32-KV oracle fails rows 23, 28, 29, and 35; the BF16-KV oracle still
fails rows 23, 28, and 29. This changes but does not explain the remaining numerical drift, so no
tolerance or acceptance criterion was changed. The machine-readable result is
`artifacts/S03/qwen38-reference-bf16-kv-diagnostic.json`.

The selected-hidden diagnostic then ran the first 30 chat tokens and captured the final normalized
hidden row at step 29 before LM-head projection. Target logits for the 30-token prefix are
byte-identical to the first 30 rows of the stable 60-token replay. The target hidden row differs
from the FP32-KV oracle by max `0.2152066` / RMSE `0.0627003`, and from the BF16-KV oracle by max
`0.2200685` / RMSE `0.0617750`. This localizes the row-29 discrepancy before LM-head projection;
the next diagnostic boundary is per-layer output/state comparison. Evidence is recorded in
`artifacts/S03/qwen38-hidden-row29-localization.json`.

The per-layer diagnostic captured the post-MLP residual after every one of the 64 decoder layers
at the same row-29 continuation step. Error grows gradually from layer 0 RMSE `0.0003813` to
layer 63 RMSE `0.1032561`; attention-layer jumps occur, but no single layer is catastrophic. The
first BF16 linear-state diagnostic missed the single-token in-place convolution update; after
correcting that hook, the BF16 KV plus linear-state reference reaches final RMSE `0.1028437` and
max error `0.3636072`, still without removing the drift. These storage choices are therefore not
the sole explanation. The target and reference boundary hashes plus selected layer metrics are
recorded in `artifacts/S03/qwen38-layer-boundary-localization.json`.

The post-token-mixer boundary was then captured after each decoder layer with a matching
Transformers token-mixer hook. At layer 0 the target/reference RMSE is `0.0002270`; at layer 63 it
is `0.0885137`, with no layer exceeding max error `0.5`. The post-MLP layer-63 RMSE is `0.1028437`,
so the MLP/residual stage adds only a bounded increment to an already accumulated token-mixer
drift. This capture used physical GPU 1 because GPU 0 was occupied by an unrelated 30-GiB VLLM
process; both are RTX 5090 `sm_120a`. The corrected comparison is included in the same localization
artifact.

Repeating the post-token-mixer capture against a CUDA Transformers oracle produced the same
boundary shape: layer-63 RMSE `0.0876815` and max error `0.3370228`, with no traced boundary over
`0.5`. This is effectively the same result as the CPU oracle and confirms that the remaining
acceptance outliers are not caused by host/device placement of the independent reference. The
CUDA capture identity is recorded in `artifacts/S03/qwen38-layer-boundary-localization.json`.

An independent Transformers oracle was also run on physical GPU 1 with the same BF16 KV and
linear-attention state emulation. Its greedy sequence is byte-for-byte token-equivalent to the
current SuperInfer replay, while logit comparison still has rows 23 and 29 over the unchanged
`max_abs=0.5` threshold (`max_abs=0.5977492`, RMSE `0.0518031`). This rules out the CPU oracle's
host-vs-device math as the primary explanation. The CUDA-oracle hash and environment are recorded
in `artifacts/S03/qwen38-layer-boundary-localization.json`.

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

The remaining acceptance work is to localize the residual numerical drift and historical replay
discrepancy, then produce
the reviewed passing report and second complete target-session corpus run. The legal boundary proof
is complete for positions 4095/4096, but the numerical model contract is not yet closed for all
declared corpus cases. S03F-02 remains blocked until that S03 acceptance is explicitly closed.

An operation-level GDN diagnostic then compared the first packed-NVFP4 projection and convolution
against a two-segment Transformers oracle. The artifact-bound `in_proj_qkv` output matched at
max `1.38283e-5`, and post-convolution output matched at max `0.00198197`; existing attention,
state, and final layer checks also passed. This rules out the first GDN projection and convolution
as the immediate source of the long-replay drift. The diagnostic is opt-in through
`SUPERINFER_QWEN38_GDN_QKV_F32` and `SUPERINFER_QWEN38_GDN_CONV_F32`, with durable evidence in
`artifacts/S03/qwen38-gdn-operation-localization.json`. No acceptance tolerance changed.
