---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: S04-P1
status: autonomous_execution
last_updated: "2026-09-11T00:00:00Z"
progress:
  total_phases: 14
  completed_phases: 9
  total_plans: 35
completed_plans: 17
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
**Recovery headline:** decode **0.032 -> 0.19 tok/s** (TPOT 31 s -> 5.3 s); R02 NVFP4 row-parallel 33x local; 7.6–9.4x E2E reproduced.

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
| S04 performance ladder | **Active (S04-P1)** | fresh profile -> one target -> correctness -> reproduced benchmark |
| S03F Flash-Next | S03F-01 retained; S03F-02+ deferred (D-019 binding, D-020 ordering) | `FLASH-NEXT-DESIGN.md`; capacity/quality blocked |
| S05 autoresearch | Deferred until 3 manual loops exist | design from proven loop, not generic |
| S06/S07/S08 | Planned | Pending |

## Current Focus

S03 correctness is closed and the first optimization loop is proven. The active lane is the profiler-driven S04 performance ladder: repeatedly (1) take a fresh profile of the current binary, (2) select the change with the largest defensible E2E opportunity, (3) optimize exactly that behind a retained fallback and an independent correctness oracle, (4) reproduce the benchmark in a second fresh session. Do not optimize by name or roadmap order. Full autoresearch scaffolding is deferred until three real manual loops exist.

Reference: `.planning/phases/R03-first-speed-proof/R03-SUMMARY.md`, `.planning/phases/R01-qwen-baseline/R01-SUMMARY.md`.

## Understanding Gate State

| Field | Current value |
|---|---|
| Current historical gates | Gate A and Gate B reached; neither user-passed |
| User status | Packets retained for later study under D-014 |
| Highest passed L2 gate | None |
| Debt policy | D-014 autonomous override active; no gate is marked passed on user's behalf |
| Next understanding event | L2 Gate C (S04 mechanisms) when a mechanism transition is reached |
| Blocked boundary | S03F-02+ until a lane decision + D-019 evidence; full autoresearch until 3 manual loops |
| Next optional user action | Study any retained understanding packet when convenient |

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
