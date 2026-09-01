---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: S03
status: autonomous_execution
last_updated: "2026-09-01T00:00:00Z"
progress:
  total_phases: 10
  completed_phases: 3
  total_plans: 31
completed_plans: 10
current_phase_name: qwen38-e2e
parallel_research_phase: S03F-01
s03f_01_status: research_complete_capacity_quality_blocked
---

# Project State

**Project:** SuperInfer
**Milestone:** V0 — Qwen proof, Flash-Next architecture proof, research loop, model-family validation
**Status:** Autonomous execution enabled under D-014; understanding packets retained as study checkpoints
**Primary implementation phase:** S03 — Qwen3.8 end-to-end
**Permitted parallel research:** S03F-01 — Flash-Next model contract/capacity proof only
**Branch intent:** `work/ultraplan-v0`

## Progress

| Phase | Status | Plans | Evidence |
|---|---|---:|---|
| S00 | Complete | 2 | [S00-01](phases/S00-foundation/S00-01-SUMMARY.md), [S00-02](phases/S00-foundation/S00-02-SUMMARY.md) |
| S01 | Complete — Gate A reached | 3 | [S01-01](phases/S01-artifact-ir/S01-01-SUMMARY.md), [S01-02](phases/S01-artifact-ir/S01-02-SUMMARY.md), [S01-03](phases/S01-artifact-ir/S01-03-SUMMARY.md) |
| S02 | Complete — Gate B reached | 3 | [S02-03](phases/S02-sm120-baseline/S02-03-SUMMARY.md) |
| S03 | In progress | 3 | S03-01 and S03-02 complete; S03-03 acceptance is blocked by accumulated numerical drift |
| S03F | Planned; S03F-01 may run research-only in parallel | 6 | [design](FLASH-NEXT-DESIGN.md); official quality decision pending; local packed candidate inventoried |
| S04 | Planned; blocked on S03F correctness | 3 | Pending |
| S05 | Planned | 3 | Pending |
| S06 | Planned | 2 | Pending |
| S07 | Planned | 3 | Pending |
| S08 | Planned | 3 | Pending |

## Current Focus

S03 remains the primary implementation lane. The corrected deployment oracle now bounds the long-context chat result: the deterministic 60-token SuperInfer replay matches all 60 Transformers greedy tokens but fails the unchanged 0.5 max-abs logit contract on rows 23, 29, and 30 while passing mean/RMSE limits. Full CPU repository validation passes. GPU 0 is available for controlled diagnostics; GPU 1 remains occupied by the user-owned NInfer service. The latest state/liveness probes are recorded in `artifacts/S03/qwen38-long-replay-diagnostic-round2.json`.

The approved S03F amendment adds Flash-Next after S03 and before S04. **Only S03F-01 may begin before S03 closes**, and it is research-only: pin reference/model revisions, inventory exact packed tensors, produce a capacity ledger, and evaluate quantization/residency recipes. S03F-01 has pinned official model/source identity and metadata, records the incomplete RadixArk NVFP4 candidate, and now inventories a complete local AtomicChat GGUF conversion with exact shard hashes and packed tensor bytes. The local conversion fits a two-device capacity projection with host-mmap PLE and 4 GiB headroom per GPU, but it is not the official checkpoint and has no SuperInfer quality evidence. Official capacity/quality selection remains blocked, and no upstream serving metric substitutes for SuperInfer qualification. The research tooling did not modify Physical Plan, MemoryPlanner, runtime or kernels.

The current S03 reference is pinned to the actually qualified Transformers `5.12.1` / torch
`2.13.0+cu130` environment. Its deployment-storage correction models cached GDN decode from
position zero, current-row BF16 convolution rounding, and BF16 embedding/final-norm I/O. Five
fresh 13-token chat-prefix target processes are byte-identical, but the corrected full-model
oracle still has accumulated max-abs logit outliers above the unchanged `0.5` contract. See
`artifacts/S03/qwen38-reference-deployment-storage-probe.json` and
`artifacts/S03/qwen38-chat-prefix13-repeatability.json`; S03-03 remains open.

The deployment-v8 acceptance rerun completed two fresh target sessions on GPU 0. Plain-short,
Unicode, and special-token cases pass numerical/token/repeatability checks; chat-template and
varied-length match all greedy tokens and are repeatable but fail the unchanged numerical
contract. Exact results are in `artifacts/S03/qwen38-s03-deployment-v8-acceptance.json`.
Post-attention/state localization at varied-length step 36 shows deterministic accumulated drift
before the MLP, a layer-42 amplification, and no isolated recurrent-state corruption; see
`artifacts/S03/qwen38-post-attention-state-localization-v8.json`. S03 remains open and S03F-02
is still blocked.
An explicit layer-42 physical-output trace now covers the GDN gated-normalization/output path,
token-mixer residual, post-attention norm, MLP, and final residual. The recurrent core remains
close to the independent oracle, while the deterministic upstream difference is amplified
through the gated path; the target residual self-check passes within max `0.0198853`. No
uninitialized read, aliasing defect, or state corruption is evidenced. See
`artifacts/S03/qwen38-layer42-post-path-localization-v9.json`; the strict numerical contract
still fails and S03 remains open.
The current acceptance review is recorded in
`.planning/phases/S03-qwen38-e2e/S03-03-REVIEW-LATEST.md`: token agreement and replay
repeatability are not being substituted for the unchanged numerical contract. Closure requires a
root-cause fix or a separately reviewed, independently justified quantized numerical contract.

After S03 closes, S03F proceeds through multi-device placement, host-resident PLE, MoE, QSA/gated residual and final dual-5090 text correctness. Broad kernel optimization remains S04 work.

## Understanding Gate State

| Field | Current value |
|---|---|
| Current historical gates | Gate A and Gate B reached; neither user-passed |
| User status | Packets retained for later study under D-014 |
| Primary implementation phase | S03 — Qwen3.8 end-to-end |
| Parallel research | S03F-01 model contract/capacity only |
| S03F understanding status | L2 architecture packet not yet reached; design approved |
| Highest passed L2 gate | None |
| Debt policy | D-014 autonomous override active; no gate is marked passed on user's behalf |
| Allowed autonomous work now | Complete S03; independently execute S03F-01 research-only tasks |
| Blocked boundary | S03F-02 multi-device runtime implementation until S03 acceptance closes |
| Next optional user action | Study any retained understanding packet when convenient |

Canonical protocol: [`.planning/UNDERSTANDING-GATES.md`](UNDERSTANDING-GATES.md). Durable user ledger: [`.planning/UNDERSTANDING.md`](UNDERSTANDING.md).

## Next Commands

**Primary lane:** continue `.planning/phases/S03-qwen38-e2e/S03-03-PLAN.md` for broader prefill/decode corpus, near-boundary checks, and final acceptance report.

**Parallel research lane:** execute `.planning/phases/S03F-flash-next/S03F-01-PLAN.md` only. Do not begin S03F-02 runtime changes until S03 is complete.

## Known Blockers / Decision Boundaries

- S03 requires real RTX 5090 model-level differential and end-to-end evidence; primitive-only tests are insufficient.
- A BF16-KV reference diagnostic changes but does not explain the remaining long-replay logit outliers;
  selected-hidden tracing localizes row 29's first proven mismatch before LM-head projection. No
  numerical tolerance was loosened and S03 remains open. Per-layer post-MLP tracing now shows
  gradual accumulated drift through the decoder rather than one catastrophic layer; BF16 KV plus
  BF16 convolution-state emulation does not remove it. Post-token-mixer tracing narrows the first
  layer boundary to RMSE `0.0002270` and the final traced layer to `0.0885137`, still below the
  per-boundary `0.5` diagnostic ceiling but insufficient to close the logit contract. See
  `artifacts/S03/qwen38-layer-boundary-localization.json`. A CUDA Transformers oracle with the
  same storage emulation matches greedy tokens but retains the logit outliers, so the remaining
  cause is not simply CPU-vs-CUDA reference math; its post-token-mixer layer-63 RMSE is `0.0876815`.
  An operation-level GDN diagnostic matches layer-0 packed-NVFP4 qkv projection at `1.38283e-5`
  max error and post-convolution output at `0.00198197`; see
  `artifacts/S03/qwen38-gdn-operation-localization.json`. The first GDN projection/convolution is
  not the immediate source of the remaining drift. The same diagnostic now matches the recurrent
  core at `3.6478e-5` max error, further narrowing the source to later GDN normalization/output or
  accumulated cross-layer behavior. The normalized/gated core output also matches at `0.00436258`
  max error, so the complete layer-0 GDN token-mixer path passes staged external differentials.
  All-layer boundary reduction shows the first material post-token-mixer jump at layer 3, the first
  full-attention block (`max_abs=0.070608`, `RMSE=0.001046`), while prior GDN drift is nonzero. The
  next S03 diagnostic is long-context full-attention/cache behavior versus input-error amplification;
  see `artifacts/S03/qwen38-full-attention-jump-localization.json`.
  The corrected standalone layer-3 experiment now passes 30-step decode with deployment-matched
  BF16 KV rounding (`final_hidden max_abs=0.00111389`, attention-output `0.00135803`), ruling out
  a standalone long-context full-attention/cache defect. See
  `artifacts/S03/qwen38-layer3-long-context-differential.json`; the remaining blocker is accumulated
  full-stack numerical drift. A temporary FP64 NVFP4 GEMV accumulator probe produced a byte-identical
  30-token logits capture and was reverted; see
  `artifacts/S03/qwen38-nvfp4-double-accumulation-probe.json`.
  BF16-rounding the reference removes one but not all logit outliers, and an FP64 RMSNorm reduction
  probe worsens the staged layer differential; both hypotheses are rejected. See
  `artifacts/S03/qwen38-logit-dtype-diagnostic.json` and
  `artifacts/S03/qwen38-rmsnorm-precision-probe.json`.
  The layer-0 GDN differential also passes 30 state-continuing segments (`max_abs=0.035728455`,
  `RMSE=0.00039070655`), including segment 29 at max `0.00368`; see
  `artifacts/S03/qwen38-gdn-long-context-differential.json`. The remaining blocker is whole-model
  accumulated drift rather than isolated GDN recurrence.
  A target-input replay of model layer 42 now matches the independent Transformers layer at QKV
  max `0.01714611`, convolution max `0.00386417`, recurrent-core max `0.00003777`, and complete
  layer-output max `0.12633133`. This confirms the layer-42 implementation and state transition;
  the full-model cliff is upstream hidden-state error amplification. See
  `artifacts/S03/qwen38-layer42-input-amplification.json`.
  A follow-up exact-target-input diagnostic captures layer-42 GDN `in_proj_z` and matches the
  independent oracle at max `0.01321268` / RMSE `0.00264875`; QKV, convolution, and recurrent-core
  comparisons remain bounded. The standalone z-projection hypothesis is rejected and no
  production change was made. See `artifacts/S03/qwen38-gdn-z-projection-localization-v11.json`.
  A 30-token chat replay with activation lifetime reuse disabled is byte-identical to the
  incumbent plan and retains the same oracle error (`max_abs=0.5856843`, `RMSE=0.0568142`,
  failing rows 23 and 29). Activation aliasing is rejected as the drift source; no production
  memory policy changed. See `artifacts/S03/qwen38-activation-reuse-localization-v12.json`.
- The first real deployment plan specializes KV capacity to 4,096 positions; the authored 262,144-token
  capacity does not fit alongside the full Qwen payload in a 32-GiB RTX 5090 envelope.
- S03F-01 has pinned immutable Flash-Next model/reference revisions, exact formula-only state/workspace estimates, and a complete authenticated local GGUF conversion inventory. Its capacity-only projection fits with host-mmap PLE and 4 GiB per-GPU headroom, but the conversion is not the pinned official artifact and has no executable-reference quality evidence; no expert residency implementation may assume support until that decision is resolved.
- The final two-session S03 acceptance refresh is recorded in `artifacts/S03/qwen38-s03-final-acceptance-refresh.json`: all corpus greedy tokens and captures are repeatable, but chat-template and varied-length still fail the unchanged numerical contract.
- The authoritative deployment-v8 two-session acceptance is recorded in `artifacts/S03/qwen38-s03-deployment-v8-acceptance.json`: plain/Unicode/special pass; chat-template and varied-length are greedy-correct and repeatable but numerically fail. Post-attention/state localization is recorded in `artifacts/S03/qwen38-post-attention-state-localization-v8.json`.
- A reference-only per-layer BF16 output-rounding probe worsened the chat max error to `1.0134909153` and is rejected; no production change was made.
- An isolated CUDA `--fmad=false` build materially changed the 30-token capture and worsened the reference comparison; FMA contraction is rejected as the primary drift correction. Evidence: `artifacts/S03/qwen38-fmad-off-probe.json`.
- A temporary Kahan-style FP32 accumulation probe inside `nvfp4_linear_f32` materially changed the first 30-token capture and worsened the reference comparison (`max/mean/RMSE=0.8269062/0.0435945/0.0570161` versus incumbent `0.5856843/0.0430106/0.0568142`); the hypothesis is rejected and the production kernel was restored. Evidence: `artifacts/S03/qwen38-kahan-probe.json`.
- An opt-in reference probe rounded the exact semantic BF16 operation boundaries and worsened the 60-row chat comparison (`max/mean/RMSE=0.8422465/0.0466893/0.0633060`, 14 failing rows) versus the incumbent (`0.5856843/0.0430106/0.0568142`, 2 failing rows). The hypothesis is rejected; the diagnostic flag is retained and no acceptance or production contract changed. Evidence: `artifacts/S03/qwen38-semantic-boundary-rounding-probe.json`.
- An isolated NVFP4 pairwise-reduction probe slightly improved mean/RMSE but worsened max error to `0.7053003` and increased failures to four rows; it is rejected by the unchanged max-abs gate and the production kernel was restored. Evidence: `artifacts/S03/qwen38-nvfp4-pairwise-probe.json`.
- An oracle run explicitly round-tripped all non-quantized BF16 checkpoint weights before FP32 computation and reproduced deployment-v8 exactly (`max/mean/RMSE=0.5856843/0.0390874/0.0522504`, failing rows `23,29,30`); weight storage is ruled out as the remaining drift source. Evidence: `artifacts/S03/qwen38-bf16-weight-roundtrip-probe.json`.
- The diagnostic command-trace API now accepts explicit `(command_id, buffer_id)` result requests; a BF16 KV-append regression proves that tracing no longer silently captures a trailing operand as the result. Production execution and the numerical contract are unchanged. Evidence: `artifacts/S03/qwen38-command-trace-output-contract.json`.
- If acceptable full expert residency across two 5090s is not feasible, S03F-04 may not invent silent expert paging. Record a capacity/residency ADR first.
- Dual-GPU runtime work must validate actual peer-access topology and retain a pinned-host staged fallback; peer access is not assumed from GPU model alone.
- Flash-Next vision and MTP are explicitly outside S03F.

## Planning Notes

- `FLASH-NEXT-DESIGN.md` is the canonical architecture amendment for S03F.
- S03 acceptance criteria were intentionally not modified by the amendment.
- S03F is an architecture/correctness phase, not a performance phase; optimization hypotheses belong in S04/S05 after correctness.
- Decisions are captured in `.planning/DECISIONS.md`; changes require a superseding entry.
