---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: R03-complete
status: autonomous_execution
last_updated: "2026-09-11T00:00:00Z"
progress:
  total_phases: 14
  completed_phases: 7
  total_plans: 35
completed_plans: 14
current_phase_name: first-speed-proof
parallel_research_phase: none
s03f_01_status: research_complete_capacity_quality_blocked
---

# Project State

**Project:** SuperInfer
**Milestone:** V0 — Qwen proof, results-first performance loop, Flash-Next architecture proof, research loop, model-family validation
**Status:** RECOVERY SPRINT COMPLETE (success path). S03 closed under D-021; R01 baseline captured; R02 optimized one profiler-selected bottleneck with retained fallback; R03 reproduced positive end-to-end decode improvement in a second fresh session. Autonomous execution under D-014 + D-020 continues.
**Primary implementation phase:** Next lane decision from fresh profiler evidence (GDN attention now leads at 27%): repeat the proven loop on the next bottleneck, ground minimal autoresearch scaffolding in it, or resume S03F when D-019 evidence is available. No mechanical default to old phase order.
**Next result gate:** None pending; recovery stop condition met.
**Branch intent:** `sol/results-first-recovery`

## Progress

| Phase | Status | Plans | Evidence |
|---|---|---:|---|
| S00 | Complete | 2 | [S00-01](phases/S00-foundation/S00-01-SUMMARY.md), [S00-02](phases/S00-foundation/S00-02-SUMMARY.md) |
| S01 | Complete — Gate A reached | 3 | [S01-01](phases/S01-artifact-ir/S01-01-SUMMARY.md), [S01-02](phases/S01-artifact-ir/S01-02-SUMMARY.md), [S01-03](phases/S01-artifact-ir/S01-03-SUMMARY.md) |
| S02 | Complete — Gate B reached | 3 | [S02-03](phases/S02-sm120-baseline/S02-03-SUMMARY.md) |
| S03 | Complete (S03-01/02 complete; S03-R Outcome A, closed under D-021) | 4 | Deployment-v8 history + `S03-R-SUMMARY.md`, D-021 contract + session-1/2 evidence |
| S03-R | Complete — Outcome A | 1 | Plan `.planning/phases/S03-qwen38-e2e/S03-R-PLAN.md`; evidence `artifacts/S03R/` |
| R01 | Complete | 1 | `R01-SUMMARY.md`; baseline 0.032 tok/s, NVFP4 97.7% profile; evidence `benchmarks/runs/R01-baseline/` |
| R02 | Complete — one profiler-selected target, bit-identical differential | 1 | `R02-PLAN.md`; row-parallel NVFP4, 33× local |
| R03 | **Complete — PASS, recovery stop condition met** | 1 | `R03-SUMMARY.md`; 7.6–9.4× E2E reproduced; evidence `benchmarks/runs/R03/` |
| R02 | Planned; target selected from R01 only | 0 | Placeholder `.planning/phases/R02-first-bottleneck/README.md` |
| R03 | Planned; blocked until R02 | 1 | Plan `.planning/phases/R03-first-speed-proof/R03-PLAN.md` |
| S03F | S03F-01 retained; S03F-02+ postponed until R03 per D-020 | 6 | [design](FLASH-NEXT-DESIGN.md); official quality decision pending; local packed candidate inventoried |
| S04 | Planned; deferred until R03 per D-020 | 3 | Pending |
| S05 | Planned | 3 | Pending |
| S06 | Planned | 2 | Pending |
| S07 | Planned | 3 | Pending |
| S08 | Planned | 3 | Pending |

## Current Focus

S03 is closed: S03-R Outcome A passed session-2 under D-021 (240 strict rows greedy-exact across two byte-identical sessions, 12 near-tie rows in-set, distributional bounds clear, 66 listed long-103 outlier rows reported; all kernel/layer gates unchanged). R01 is the active lane: measure the current unoptimized runtime (decode tok/s + TPOT, prefill tok/s + TTFT, peak VRAM, kernel/region breakdown) before any optimization. No S03 numerical work remains open.

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

After S03-R closure via D-021, R01 captures the reproducible baseline/profile, R02 optimizes exactly one profiler-selected bottleneck, and R03 reproduces the end-to-end speedup. S03F-02+ and broad S04 work remain deferred until R03 per D-020. Broad kernel optimization remains post-R03 work, except the single R02 target and the queued repetition-robustness research note.

## Understanding Gate State

| Field | Current value |
|---|---|
| Current historical gates | Gate A and Gate B reached; neither user-passed |
| User status | Packets retained for later study under D-014 |
| Primary implementation phase | S03-R — Qwen decisive closure (L1) |
| Parallel research | None; S03F-01 retained, S03F-02+ postponed until R03 per D-020 |
| S03F understanding status | L2 architecture packet not yet reached; design approved |
| Highest passed L2 gate | None |
| Debt policy | D-014 autonomous override active; no gate is marked passed on user's behalf |
| Allowed autonomous work now | Execute R01 baseline/profile; then R02 target from R01 only; R03 same-manifest proof |
| Blocked boundary | S03F-02+ and broad S04 until R03 (D-020); full autoresearch until post-R03; S03 numerical work is closed |
| Next optional user action | Study any retained understanding packet when convenient |

Canonical protocol: [`.planning/UNDERSTANDING-GATES.md`](UNDERSTANDING-GATES.md). Durable user ledger: [`.planning/UNDERSTANDING.md`](UNDERSTANDING.md).

## Next Commands

**Recovery sprint complete.** Next lane is an evidence-driven choice, not a default: (a) repeat the proven loop on `gated_delta_attention_f32` (now 27% of decode), (b) ground minimal autoresearch scaffolding in the proven loop shape, or (c) resume S03F when D-019 evidence is available.

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
  An exact-target-hidden LM-head replay using independent chunked NVFP4 dequantization differs by
  max `0.0312481` before BF16 output storage and `0.03125` after storage, ruling out the final
  projection as the source of the full-model outlier. See
  `artifacts/S03/qwen38-lm-head-localization-v13.json`.
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
- A 48-pair explicit provider-result/BF16-destination trace at chat continuation step 29 matched independent RN-even materialization exactly across layers 0--5, 42, and 63; physical buffer binding/materialization is rejected as the drift source. Evidence: `artifacts/S03/qwen38-materialization-localization-v14.json`.
- An isolated kernel-12 `rsqrtf`/multiply RMSNorm probe worsened the first-30 chat comparison to max/mean/RMSE `0.6775799/0.0430284/0.0570885` versus incumbent `0.5856843/0.0430106/0.0568142`; production `sqrtf`/division was restored. Evidence: `artifacts/S03/qwen38-rmsnorm-rsqrt-probe-v15.json`.
- A CUDA Transformers oracle rerun with TF32 explicitly disabled was byte-identical to deployment-v8 and retained target max/mean/RMSE `0.5856843/0.0430106/0.0568142`; reference TF32 policy is rejected. Evidence: `artifacts/S03/qwen38-reference-tf32-probe-v16.json`.
- The diagnostic command-trace API now accepts explicit `(command_id, buffer_id)` result requests; a BF16 KV-append regression proves that tracing no longer silently captures a trailing operand as the result. Production execution and the numerical contract are unchanged. Evidence: `artifacts/S03/qwen38-command-trace-output-contract.json`.
- If acceptable full expert residency across two 5090s is not feasible, S03F-04 may not invent silent expert paging. Record a capacity/residency ADR first.
- Dual-GPU runtime work must validate actual peer-access topology and retain a pinned-host staged fallback; peer access is not assumed from GPU model alone.
- Flash-Next vision and MTP are explicitly outside S03F.

## Planning Notes

- `FLASH-NEXT-DESIGN.md` is the canonical architecture amendment for S03F.
- S03 acceptance criteria were intentionally not modified by the amendment; S03-R may supersede only the historical model-level gate via evidence-derived D-021, never local kernel/layer gates.
- S03F is an architecture/correctness phase, not a performance phase; optimization hypotheses belong post-R03 per D-020.
- Decisions are captured in `.planning/DECISIONS.md`; changes require a superseding entry.
- Recovery sprint order per D-020: S03-R -> R01 -> R02 -> R03, then broader kernels/autoresearch/Flash-Next.
