---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: S04-P8R
status: autonomous_execution
last_updated: "2026-09-11T00:00:00Z"
progress:
  total_phases: 14
  completed_phases: 9
  total_plans: 35
completed_plans: 33
current_phase_name: results-first-performance-ladder
parallel_research_phase: none
s03f_01_status: research_complete_capacity_quality_blocked
---

# Project State

**Project:** SuperInfer
**Milestone:** V0 — Qwen proof, results-first performance ladder, Flash-Next architecture proof, research loop, model-family validation
**Status:** RECOVERY SPRINT COMPLETE (success path). S03/S03-R complete under D-021; R01 baseline captured; R02 one profiler-selected optimization with retained fallback; R03 reproduced positive end-to-end decode gain in a second fresh session.
**Current lane:** S04-P1 — profiler-driven performance ladder (iterate fresh profile -> one target -> correctness -> reproduced benchmark). Fleet target: **>= 5 decode tok/s**.
**Branch:** `sol/results-first-recovery` (draft PR #1)
**Performance headline:** decode **0.032 -> ~8.3 tok/s** cumulative across 8 profiler-selected loops (~260x). GPU kernel time 11.54 -> 7.25 s / 60 tokens. Loops: NVFP4 row-parallel (33x), KV-attention caching (4,320x), NVFP4 vectorization (5.48x), GDN parallel (128x), RMSNorm parallel (21.9x), elementwise/cast block-parallel (39x), NVFP4 warp-per-row GEMV (1.66x local), NVFP4 decode decomposition + shape-adaptive dispatch (1.05x). Loop 7 warp-per-row and the shape-adaptive warp branch are tolerance-qualified and D-021-passing; loops 1-6 and the row-per-thread branch are bit-exact. **>=5 tok/s checkpoint MET; decode now GPU-bound.**

## Operational Truth

| Workstream | Status | Key evidence |
|---|---|---|
| S00 foundation | Complete | S00-01/02 summaries |
| S01 artifact + IR | Complete; Gate A reached (not user-passed) | S01-01/02/03 summaries |
| S02 sm120 baseline | Complete; Gate B reached (not user-passed) | S02-03 summary |
| S03 Qwen E2E | **Complete** | `S03-R-SUMMARY.md`; D-021 contract + session-1/2 evidence |
| S03-R decisive closure | **Complete — Outcome A** | `artifacts/S03R/`; D-021 |
| R01 baseline + profile | **Complete** | `R01-SUMMARY.md`; `benchmarks/runs/R01-baseline/`; 0.032 tok/s, NVFP4 97.7% |
| R02 first bottleneck | **Complete** | `R02-PLAN.md`; row-parallel NVFP4, 33x local, bit-identical |
| R03 first speed proof | **Complete — PASS** | `R03-SUMMARY.md`; `benchmarks/runs/R03/`; 7.6–9.4x E2E reproduced |
| Recovery sprint | **Complete** | stop condition met |
| S04 performance ladder | **P8-R RESOLVED: native sm_120a block-scaled NVFP4 MMA measured; integration + D-021 pending** | `S04-P1/`..`S04-P8/`; 8 optimize loops + reopened spike, 0.032->~8.3 tok/s, D-021 pass |
| S03F Flash-Next | S03F-01 retained; S03F-02+ deferred (D-019 binding, D-020 ordering) | `FLASH-NEXT-DESIGN.md`; capacity/quality blocked |
| S05 autoresearch | Minimum runner implemented (`tools/autoresearch_runner.py`, `S04-AUTORESEARCH-RUNNER.md`); use it for experiment mechanics; do not expand scope | design grounded in the 8 proven loops |
| S06/S07/S08 | Planned | Pending |

## Current Focus

S03 correctness is closed and the first optimization loop is proven. The active lane is the profiler-driven S04 performance ladder: repeatedly (1) take a fresh profile of the current binary, (2) select the change with the largest defensible E2E opportunity, (3) optimize exactly that behind a retained fallback and an independent correctness oracle, (4) reproduce the benchmark in a second fresh session. Do not optimize by name or roadmap order. The minimum autoresearch runner is implemented (`tools/autoresearch_runner.py`, `S04-AUTORESEARCH-RUNNER.md`); use it for experiment mechanics and do not expand its scope. Loops 1-8 are complete; the >=5 decode tok/s checkpoint is met (~8.3 tok/s). **P8-R is RESOLVED**: the earlier classification-D was a false negative caused by a malformed PTX probe (missing trailing scale type `.ue4m3`). `sm_120a` **does** support warp-level block-scaled NVFP4 `mma.sync` (`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`), proven to assemble (ptxas 13.1.115) and execute on the RTX 5090. The P8-R2 synthetic differential now passes **exactly (rel=0)** with random scales; the full fragment/scale contract is pinned in `S04-P8R-CONTRACT.md`. Arm A (N=1) measured at **~81 tok/s** (98 with an MMA-native repacked layout) vs the promoted 8.3 tok/s software path; Arm B (N=8, batched/speculative) at **~591-728 tok/s**. The activation-quantization step adds a ~10% per-projection rel-L2 error on synthetic data (2.4-9.3% on real hidden vectors), so the decisive open item is the quality contract: wire the native path behind an experimental selector and run the layer-3/GDN differentials and full D-021. Per the promotion rule this requires a **separate provider/layout architecture decision**; even a performance-passing Arm A is not auto-promoted. See `tests/gpu/sm120/nvfp4_mma_bench.cu` and `artifacts/S04/p8r/p8r3_arm_abc_result.txt`. **Do not begin startup/TTFT or linear_f32 work until the P8-R quality gate is decided.** **Layer/GDN fixture debt is FIXED** (`tools/run_qwen38_layer_gdn_fixtures.py`; both now run and pass). The minimum autoresearch runner is implemented at `tools/autoresearch_runner.py`.

Reference: `.planning/phases/R03-first-speed-proof/R03-SUMMARY.md`, `.planning/phases/R01-qwen-baseline/R01-SUMMARY.md`.

## Understanding Gate State

| Field | Current value |
|---|---|
| Current historical gates | Gate A, Gate B, and L2 Gate C.1 (dense NVFP4 tensor cores) reached; none user-passed |
| User status | Packets retained for later study under D-014 |
| Highest passed L2 gate | None |
| Debt policy | D-014 autonomous override active; no gate is marked passed on user's behalf |
| Next understanding event | L2 Gate C.2 (attention/KV/QSA) or C.3 (fusion/persistent/MoE) at the next mechanism transition |
| Blocked boundary | S03F-02+ until a lane decision + D-019 evidence |
| Next optional user action | Study `.planning/understanding-packets/GATE-C1.md` and answer its five questions |

Canonical protocol: [`.planning/UNDERSTANDING-GATES.md`](UNDERSTANDING-GATES.md). Durable user ledger: [`.planning/UNDERSTANDING.md`](UNDERSTANDING.md).

## Next Commands

**Primary lane:** fresh profile of the current optimized binary, then one R02-style target per `.planning/RESULTS-FIRST-RECOVERY-DESIGN.md` and the S04-P1 plan. Fleet target >= 5 decode tok/s.

**Deferred:** S03F-02+ runtime work (D-019 evidence required); full autoresearch (after 3 manual loops).

## Known Blockers / Decision Boundaries

- Current headroom: runtime still ~2,424 launches/token, single-block kernels throughout; launch overhead and unfused elementwise kernels are open opportunities to evaluate from fresh profile evidence, not assumed targets.
- `gated_delta_attention_f32` measured ~27% in the first R03 profile; this is an input to the fresh profile, not a preselected target.
- A 103-token repeat prompt is a degenerate-repetition stress case (bilateral activation explosion); repetition robustness is S04+ research, not an S03 defect.
- If acceptable full Flash-Next expert residency across two 5090s is not feasible, S03F-04 may not invent silent expert paging. Record a capacity/residency ADR first (D-019).
- Dual-GPU runtime work must validate actual peer-access topology and retain a pinned-host staged fallback.
- Flash-Next vision and MTP are explicitly outside S03F.
- Historical S03-03 numerical-archaeology notes are archived at [`.planning/archive/S03-03-HISTORICAL-BLOCKERS.md`](archive/S03-03-HISTORICAL-BLOCKERS.md) and are superseded by D-021.

## Planning Notes

- `FLASH-NEXT-DESIGN.md` is the canonical architecture amendment for S03F.
- S03-R superseded only the historical model-level `max_abs <= 0.5` gate via D-021; local kernel/layer gates are unchanged and binding.
- Optimization hypotheses are now in scope (results-first lane); performance work never bypasses the D-021 correctness gate.
- Decisions are captured in `.planning/DECISIONS.md`; changes require a superseding entry.
- Order per D-020/D-021: S03-R -> R01 -> R02 -> R03 (done), then profiler-driven S04 ladder, then autoresearch/Flash-Next by evidence.
