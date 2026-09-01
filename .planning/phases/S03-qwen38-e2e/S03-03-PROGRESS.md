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

The current acceptance review is recorded in `S03-03-REVIEW-LATEST.md`. The
deployment-v8 report has two fresh, byte-identical target sessions: all five
corpus cases select the reference greedy tokens, while `chat-template` and
`varied-length` fail the unchanged FP32-logit contract. Existing probes rule
out current nondeterminism, activation aliasing, isolated GDN recurrence, and
standalone long-context full-attention/cache defects. This is an evidence-
bounded accumulated-drift blocker, not a reason to loosen tolerances or close
S03 on token agreement alone.

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

The next diagnostic round corrected the state probe to select model layer 42 rather than the 42nd
GDN operation. The corrected layer-42 recurrent state matches the independent deployment-matched
reference at max absolute error `0.01169449`, and the BF16 convolution state at `0.25`; the
correlations are `0.9998574` and `0.9995375`. A no-alias activation plan produces a byte-identical
post-token-mixer trace and the same greedy sequence as the normal liveness plan, ruling out
activation reuse as the immediate source. A temporary query-scale ordering change was tested
against the 37-token replay, worsened row 34 from max `0.59628` to `1.04679`, and was reverted.
The diagnostic selector and durable results are recorded in
`artifacts/S03/qwen38-long-replay-diagnostic-round2.json`; the numerical acceptance blocker remains
open and no tolerance was changed.

To separate implementation error from error amplification, model-layer-42 command outputs were
captured at position 36 while the independent Transformers helper was driven with the target's
layer-41 hidden row and layer-42 state from position 35. QKV, causal convolution, and recurrent
core outputs matched at max errors `0.01714611`, `0.00386417`, and `0.00003777`; the complete
layer output matched at max `0.12633133`. This proves the layer-42 state transition and kernel
composition under the target input. The full-model layer-42 jump is therefore upstream numerical
error amplification, not a layer-42 state corruption. Evidence is recorded in
`artifacts/S03/qwen38-layer42-input-amplification.json`; S03 remains open because the unchanged
full-model logit contract is still exceeded.

An operation-level GDN diagnostic then compared the first packed-NVFP4 projection and convolution
against a two-segment Transformers oracle. The artifact-bound `in_proj_qkv` output matched at
max `1.38283e-5`, and post-convolution output matched at max `0.00198197`; existing attention,
state, and final layer checks also passed. This rules out the first GDN projection and convolution
as the immediate source of the long-replay drift. The diagnostic is opt-in through
`SUPERINFER_QWEN38_GDN_QKV_F32` and `SUPERINFER_QWEN38_GDN_CONV_F32`, with durable evidence in
`artifacts/S03/qwen38-gdn-operation-localization.json`. No acceptance tolerance changed.

The same diagnostic was extended through the GDN recurrent core, capturing both the chunk and
cached single-token reference paths. The artifact-bound core output matched the external oracle at
max `3.6478e-5` over both segments. The first GDN projection, convolution, and recurrent core are
therefore not the immediate source of the long-replay drift; acceptance remains open and tolerances
remain unchanged. The operation-localization artifact records the core capture hash and result.

The diagnostic was completed through the normalized/gated GDN core output, which matched at max
`0.00436258`. Combined with the existing output-projection, recurrent-state, and final layer
checks, the complete layer-0 GDN token-mixer path passes its staged external differential across
two state-continuing segments. The remaining S03 failure is accumulated cross-layer behavior or a
later operation, not an isolated layer-0 GDN boundary.

The complete-layer trace was then reduced across all 64 layers. The first material post-token-mixer
jump is at layer 3 (`max_abs=0.070608`, `RMSE=0.001046`), which is the first full-attention layer;
later full-attention boundaries show the same jump pattern while preceding GDN drift is already
nonzero. This localizes the next investigation to long-context full-attention/cache behavior or
amplification of prior input error, without claiming a standalone full-attention defect. The
machine-readable reduction is `artifacts/S03/qwen38-full-attention-jump-localization.json`.

The bounded long-context full-attention experiment initially exposed an oracle-contract error:
rounding KV after the reference attention call produced a false step-0 discrepancy. After moving
BF16 rounding into `DynamicCache.update`, before attention reads K/V, the 30-step layer-3 artifact
differential passed with final hidden max `0.00111389` / mean `0.0000233764` and attention-output
max `0.00135803`. This rules out a standalone long-context full-attention/cache defect and leaves
the full-model outlier as accumulated cross-stack numerical drift. Evidence is recorded in
`artifacts/S03/qwen38-layer3-long-context-differential.json`; no tolerance or production kernel
changed.

A reversible whole-model probe changed only `nvfp4_linear_f32` accumulation from FP32 to FP64 for
the 30-token chat prefix. Its stored logits capture was byte-identical to the baseline
(`367b47fc...`), with the same outlier rows 23 and 29 and max `0.59774923`; the change was reverted.
This rejects GEMV accumulator precision as an observable fix for the current BF16 logits contract.
The failed hypothesis is retained in
`artifacts/S03/qwen38-nvfp4-double-accumulation-probe.json`.

An output-contract check rounded the FP32 reference logits to BF16 before comparison. It removed
row 23 from the `max_abs=0.5` failures but left row 29 at `0.609375`, confirming that output dtype
does not explain the hidden-state drift; evidence is in
`artifacts/S03/qwen38-logit-dtype-diagnostic.json`. A separate FP64 RMSNorm reduction probe worsened
the staged layer-3 differential and was reverted; its result is in
`artifacts/S03/qwen38-rmsnorm-precision-probe.json`.

The GDN staged differential was extended from two to 30 state-continuing segments. The complete
artifact-bound layer remained within contract at aggregate max `0.035728455`, mean `0.0002591842`,
and RMSE `0.00039070655`; the row-29-equivalent segment was max `0.00368` / RMSE `0.00020`.
Projection, convolution, recurrent-core, gated-output, attention-output, and final-layer checks
all remained passing. This rules out GDN recurrence as the immediate source of the whole-model
outlier. Evidence is recorded in `artifacts/S03/qwen38-gdn-long-context-differential.json`.

The deployment-matched corpus oracle was corrected to use the actual qualified environment,
Transformers `5.12.1` with torch `2.13.0+cu130`, and to follow the target's cached single-token
GDN path from position zero. It rounds the current GDN QKV row before causal convolution, rounds
the BF16 embedding output and final RMSNorm output at their explicit lowering boundaries, and
retains FP32 recurrent state plus BF16 KV/convolution storage. The broad per-module BF16 hook was
tested and rejected because it increased 30-token error. The corrected 60-token comparison is
repeatable and has max/mean/RMSE `0.5842886/0.0388823/0.0520733`; the narrower 30-token storage
probe improves to `0.5856843/0.0430106/0.0568142` but still leaves rows 23 and 29 above the
unchanged max-abs `0.5` contract. Evidence is recorded in
`artifacts/S03/qwen38-reference-deployment-storage-probe.json`.

The reviewer-requested minimized fresh-process replay is now repeatable: the first 13 chat tokens
produce byte-identical captures across five independent SuperInfer processes on GPU 0. This rules
out current-session nondeterminism for that prefix but does not close the historical long-replay
discrepancy; evidence is recorded in `artifacts/S03/qwen38-chat-prefix13-repeatability.json`.

The corpus reference identity discrepancy is resolved in the checked-in acceptance fixture: the
qualified local and planned Transformers version are both `5.12.1`. S03 remains open because the
full numerical contract still fails on accumulated long-context logit outliers; no tolerance was
changed and no production kernel was modified by these diagnostics.

## Deployment-contract acceptance state

The corrected deployment oracle comparison is now bounded and deterministic. It uses the
Transformers `5.12.1` / torch `2.13.0+cu130` CUDA capture at
`build/evidence/reference-chat-deployment-v7/chat-template.f32` and models cached decode from
position zero, BF16 embedding I/O, BF16 GDN QKV/linear state, BF16 KV, and BF16 final-norm I/O.
The stable 60-token SuperInfer replay matches all 60 reference greedy tokens, but the unchanged
row contract still fails absolute-logit limits on rows 23, 29, and 30. Failing row maxima are
`0.5208769`, `0.5856843`, and `0.5124722`; mean/RMSE limits pass. The full comparison, artifact
and capture hashes, unchanged numerical contract, and validation results are recorded in
`artifacts/S03/qwen38-s03-deployment-contract-acceptance-state.json`.

Repository CPU gates pass: `uv run --extra dev pytest -q` reports 60 tests and 9 subtests passed,
and `python tools/validate.py --full` passes all configure/build/test/install/sanitize/wheel steps.
Fresh full-artifact target execution is currently externally blocked because both RTX 5090s are
occupied by user-owned services using 20.75 GiB and 24.57 GiB respectively; the 19.19 GiB
SuperInfer arena cannot safely fit alongside either. This occupancy is recorded in the acceptance
state evidence and is not a SuperInfer failure.

## Final acceptance refresh

With GPU 0 available again, the canonical batched acceptance harness was rerun for two independent
target sessions against the pinned artifact and Transformers 5.12.1 / torch 2.13.0+cu130 reference.
Plain-short, Unicode, and special-token cases passed token, numerical, and repeatability checks.
Chat-template produced 60/60 greedy-token matches and repeatable captures but failed 8 numerical
rows, with max absolute error `0.6116333008`. Varied-length produced 81/81 greedy-token matches
and repeatable captures but failed 46 numerical rows, with max absolute error `4.6154732704`, mean
absolute error `0.6096376059`, and RMSE `0.7747322253`. The exact result is recorded in
`artifacts/S03/qwen38-s03-final-acceptance-refresh.json`, with raw evidence under
`build/evidence/qwen38-s03-acceptance-final/`.

This refresh confirms the existing diagnosis: deterministic long-context numerical drift remains;
no tolerance changed, no production kernel changed, and S03 is not closed. S03F-02 remains
engineering-blocked.

The next precision-contract probe rounded each Transformers decoder-layer output through BF16
between layers while retaining the qualified BF16 embedding, GDN state, KV, and final-norm
boundaries. It worsened the chat-template comparison to global max `1.0134909153`, mean
`0.0439874046`, and RMSE `0.0602338836`, versus the deployment-matched baseline max `0.6116333008`.
The per-layer BF16 materialization hypothesis is rejected and no production change was made.
Evidence is recorded in `artifacts/S03/qwen38-layer-output-rounding-probe.json`.

## Deployment-v8 acceptance and post-attention localization

The canonical two-session acceptance rerun completed against the deployment-matched CUDA
Transformers 5.12.1 oracle. Plain-short, Unicode, and special-token cases pass numerical,
greedy-token, and repeatability checks. Chat-template matches all 60 greedy tokens and is
repeatable, but fails rows 23, 29, and 30 with max/mean/RMSE `0.5856843/0.0390874/0.0522504`.
Varied-length matches all 81 greedy tokens and is repeatable, but fails rows
`36,43,46,52,54,56,62,64,65,66,67,70,71,72,73,74,75,76,77,78,79` with
max/mean/RMSE `4.5100713/0.0807379/0.1720535`. Both fresh target sessions have identical
capture hashes. The tracked summary is `artifacts/S03/qwen38-s03-deployment-v8-acceptance.json`
and raw evidence is under `build/evidence/qwen38-s03-acceptance-deployment-v8/`.

A focused varied-length step-36 run captured all 64 post-token-mixer residuals and the model
layer-42 state on GPU 0. Compared with a fresh CUDA Transformers oracle using the same BF16
embedding/KV/convolution/final-norm contract, post-attention drift is already present at layer 0
(`max_abs=0.0158510`, `RMSE=0.0002234`) and increases through layer 41. It jumps at layer 42
(`max_abs=0.6800499`, `RMSE=0.0288765`) and reaches layer-63
`max_abs=2.0009232`, `RMSE=0.2335027`. The layer-42 recurrent state remains close
(`max_abs=0.0118160`, `RMSE=0.0000958`); the BF16 convolution-state difference is
`max_abs=0.2109375`, `RMSE=0.0342077`, consistent with the already qualified storage contract.
The state capture is byte-identical to the prior target diagnostic. This localizes the remaining
failure to deterministic accumulated cross-layer numerical amplification; it does not identify a
reproducible uninitialized read, activation-aliasing defect, or isolated layer-42 state corruption.
Evidence is recorded in `artifacts/S03/qwen38-post-attention-state-localization-v8.json`.

S03 remains open: the numerical contract and tolerances are unchanged, no production kernel was
modified, and S03F-02 remains engineering-blocked until the contract is resolved or formally
superseded by an approved evidence-backed decision.

A compiler arithmetic probe built the CUDA target with `--fmad=false` and replayed the first 30
chat-template tokens. The capture changed materially from the incumbent (`max_abs=0.984375`),
and its reference comparison worsened to max/RMSE `0.5925694/0.0570751` with failing rows
`21,24,28,29`; the incumbent has max/RMSE `0.5856843/0.0568142` with failing rows `23,29`.
FMA contraction is therefore rejected as the primary correction. The isolated build was not
promoted and no production kernel changed. Evidence is recorded in
`artifacts/S03/qwen38-fmad-off-probe.json`.

A compensated-summation probe then rebuilt the CUDA target with a temporary Kahan-style FP32
accumulator inside `nvfp4_linear_f32` and replayed the first 30 chat-template tokens. The probe
capture differed from the incumbent, but its reference comparison worsened from
`0.5856843/0.0430106/0.0568142` to `0.8269062/0.0435945/0.0570161` for max/mean/RMSE, with
four failing rows rather than two. The isolated build was discarded and the production kernel was
restored unchanged. Evidence is recorded in `artifacts/S03/qwen38-kahan-probe.json`.

An opt-in oracle probe then rounded the exact semantic operation boundaries that the lowering
materializes through BF16: embedding, input norm, token mixer, attention residual, post norm, MLP,
layer residual, and final norm. Against the same 60-row target capture, this produced
`0.8422465/0.0466893/0.0633060` max/mean/RMSE with 14 failing rows, compared with the incumbent
`0.5856843/0.0430106/0.0568142` and two failing rows. The semantic-boundary storage hypothesis
is rejected; the opt-in reference flag is retained as a reusable diagnostic and no acceptance
contract or production runtime changed. Evidence is recorded in
`artifacts/S03/qwen38-semantic-boundary-rounding-probe.json`.

An isolated NVFP4 GEMV probe then used eight interleaved FP32 partial sums followed by a pairwise
tree reduction. It changed the 30-token capture and slightly improved mean/RMSE, but worsened the
max error from `0.5856843` to `0.7053003` and increased failing rows from two to four. The candidate
is rejected by the max-abs correctness gate, the isolated build was discarded, and the production
kernel was restored unchanged. Evidence is recorded in
`artifacts/S03/qwen38-nvfp4-pairwise-probe.json`.

The oracle was also run with every non-quantized BF16 checkpoint weight explicitly round-tripped
through BF16 before FP32 computation. This produced the same greedy sequence and the same
`0.5856843/0.0390874/0.0522504` max/mean/RMSE and failing rows `23,29,30` as deployment-v8.
Because the source values are already BF16-representable, weight storage is ruled out as the
remaining drift source. The opt-in flag is retained for contract diagnostics; no acceptance or
production change was made. Evidence is recorded in
`artifacts/S03/qwen38-bf16-weight-roundtrip-probe.json`.

## Diagnostic trace output contract

The test-only command trace API previously copied the final declared command operand, which is
not always the result buffer: RMSNorm places its scale after the output, and KV append places the
cache operands after the input rows. The API now accepts explicit `(command_id, buffer_id)` trace
requests while retaining the legacy final-operand overload for compatibility. A BF16 KV append
regression captures the position-zero key-cache row and passes on GPU 0 RTX 5090. This changes no
production execution, numerical contract, tolerance, or plan representation. Evidence is recorded
in `artifacts/S03/qwen38-command-trace-output-contract.json`.

The existing real-state layer-3 differential remains the strongest isolated attention evidence:
with target layer-3 input and prior KV state supplied at position 28, the artifact layer matches
the independent quantized Transformers oracle at max absolute error `3.8147e-6`. The whole-model
acceptance failure therefore remains upstream deterministic accumulation/amplification; S03 is
still open.

## Layer-42 post-path localization

The test-only command trace was extended to accept explicit `command_id:buffer_id` requests so
multi-operand commands can be captured at their authored result buffer. A minimized varied-length
run captured layer 42 at continuation step 36 on GPU 0, including GDN RMSNorm, gated output,
output projection, token-mixer residual, post-attention RMSNorm, MLP projections/product, and
layer residual. The matching independent Transformers trace uses the deployment-v8 storage
contract and the pinned Transformers 5.12.1 / torch 2.13.0+cu130 environment.

The operation-level comparison shows that the recurrent GDN core is not the isolated source of
the apparent layer-42 cliff: the prior core trace was max `0.0005254` / RMSE `0.0000190`, while
the post-core path progresses through max errors `0.4607` (GDN RMSNorm), `1.3200` (SiLU gate),
`0.6370` (output projection), `0.9009` (token-mixer residual), and `1.6383` (layer residual).
The target residual command independently satisfies its captured-input sum within max
`0.0198853` / RMSE `0.0011967`, consistent with its explicit BF16/F32 materialization path.
This is deterministic upstream drift amplified by the nonlinear gated path, not evidence of an
uninitialized read, activation-aliasing defect, or corrupted recurrent state. No production
kernel, tolerance, or acceptance criterion changed. Full evidence and hashes are recorded in
`artifacts/S03/qwen38-layer42-post-path-localization-v9.json`; raw captures remain under
`build/evidence/trace-var36-layer42-ops/` and
`build/evidence/reference-var36-layer42-post/`.

S03 remains open because the unchanged numerical contract still fails on the long-context
acceptance cases. The next closure boundary is a reviewed root-cause correction or a separately
reviewed, independently justified superseding numerical contract, followed by the final
acceptance bundle and fresh re-review.

## First full-attention jump localization

The next planned diagnostic captured the first full-attention layer at chat-template step 29,
where all-layer tracing previously showed the first material boundary jump. The explicit physical
trace covers input RMSNorm, q/gate projection, K/V projections, Q/K norms, partial RoPE,
cache-backed grouped attention, sigmoid output gate, output projection, token-mixer residual,
post-attention RMSNorm, all MLP stages, and the layer residual. The independent Transformers
trace uses the same deployment-v8 storage contract; split, SiLU-product, and final-residual
values are derived only from captured reference tensors.

Every full-attention operation remains bounded: input RMSNorm max `0.0361`, Q/gate projection
`0.0335`, attended output `0.0106`, output projection `0.0171`, token-mixer residual `0.0723`,
and final layer residual `0.0699`. The target's gated MLP product self-check is max `0.00000006`;
its residual self-check is max `0.005619`. This rules out a standalone KV, RoPE, GQA, output
gate, full-attention, or MLP implementation defect at the first jump. The jump is inherited
activation error entering the block. Evidence is recorded in
`artifacts/S03/qwen38-full-attention-operation-localization-v10.json`, with raw captures under
`build/evidence/trace-chat29-full-attention/` and
`build/evidence/reference-chat29-full-attention/`.

S03 remains open under the unchanged numerical contract. The remaining technical question is
the accumulated pre-layer-3 activation drift, not full-attention correctness.

## GDN z-projection localization

To test whether the layer-42 GDN `in_proj_z` path was responsible for the varied-length cliff, a
diagnostic reference run replayed the exact target hidden input and recurrent state at position 36
through the independent Transformers layer. The physical trace and reference captured the QKV,
causal-convolution, z-projection, and recurrent-core outputs independently. The z projection
comparison was max `0.0132127`, mean `0.00201811`, RMSE `0.00264875`; QKV was max `0.0171461`,
convolution max `0.00386417`, and the recurrent core max `0.00003777`. This rejects the standalone
z-projection hypothesis and confirms that the apparent full-model layer-42 cliff is inherited and
amplified upstream. The diagnostic-only reference tool now records z captures for future GDN
localization; no production kernel, tolerance, or acceptance contract changed. Evidence and input
hashes are recorded in `artifacts/S03/qwen38-gdn-z-projection-localization-v11.json`.

S03 remains open because the unchanged numerical contract still fails on long replay. The next
closure boundary is root-cause correction or a separately reviewed, independently justified
quantized-reference contract, followed by final acceptance packaging and fresh review.

## Activation-reuse localization

The diagnostic tree next disabled lifetime-based activation reuse for a 30-token chat-template
replay while keeping the artifact, command schedule, GPU, and state initialization unchanged. The
diagnostic capture was byte-identical to the incumbent capture (`byte_difference_max_abs=0.0`), and
both plans had the same oracle comparison: max `0.5856843`, mean `0.0430106`, RMSE `0.0568142`,
with greedy tokens matching and rows `23,29` outside the unchanged max-absolute contract. This
rejects activation lifetime aliasing as the source of the long-context drift. No production plan,
memory policy, tolerance, or acceptance criterion changed. Evidence is recorded in
`artifacts/S03/qwen38-activation-reuse-localization-v12.json`; raw diagnostic output remains under
`build/evidence/no-alias-chat30/`.

S03 remains open under the unchanged numerical contract. The next boundary is continued
localization of accumulated whole-model arithmetic drift or a separately reviewed, independently
justified quantized-reference contract.
