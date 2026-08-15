# Project State

**Project:** SuperInfer
**Milestone:** V0 — Qwen proof, research loop, second-model validation
**Status:** Planning packet complete; implementation not started
**Current phase:** S00 — Foundation and Contracts
**Branch intent:** `planning/ultraplan-v0`

## Progress

| Phase | Status | Plans | Evidence |
|---|---|---:|---|
| S00 | Ready to execute | 2 | Pending |
| S01 | Planned | 3 | Pending |
| S02 | Planned | 3 | Pending |
| S03 | Planned | 3 | Pending |
| S04 | Planned | 3 | Pending |
| S05 | Planned | 3 | Pending |
| S06 | Planned | 2 | Pending |
| S07 | Planned | 3 | Pending |
| S08 | Planned | 3 | Pending |

## Current Focus

Start S00 only. Do not begin model- or kernel-specific work until repository boundaries, interfaces, reference testing, and CI exist. Protect the S00–S06 critical path: correct Qwen3.8 from `.sinf` and the first reproducible RTX 5090 graph.

## Understanding Gate State

| Field | Current value |
|---|---|
| Current gate | S00 CUDA execution model — L1, non-blocking |
| User status | Not started |
| Implementation phase | S00 — Foundation and Contracts |
| Highest passed L2 gate | None |
| Debt distance | 0 |
| Allowed next autonomous work | Execute S00 and prepare S01 |
| Blocked boundary | None now; Gate B becomes blocked if Gate A remains unpassed |
| Next required user action | Explain the asynchronous launch/timing issue; then run the three-way timing experiment in `.planning/UNDERSTANDING.md` |

Canonical protocol: [`.planning/UNDERSTANDING-GATES.md`](UNDERSTANDING-GATES.md). Durable user ledger: [`.planning/UNDERSTANDING.md`](UNDERSTANDING.md).

## Next Command

Review `.planning/phases/S00-foundation/S00-CONTEXT.md`, then execute `S00-01-PLAN.md` followed by `S00-02-PLAN.md`.

## Known Blockers

- Implementation requires access to an RTX 5090 for GPU acceptance and benchmark evidence. CPU-only phases can proceed without it.
- Exact Qwen3.8-27B and Gemma 4 26B-A4B upstream repository IDs/revisions must be pinned during their phases; marketing names in this packet are not a substitute for immutable model identity.
- Final CUDA/toolchain feature choices must be verified against the installed `sm_120a`-capable toolchain before S02.

## Planning Notes

- The packet was authored from the approved architecture for a greenfield repository.
- Cloud ultraplan handoff is not required to execute these files; each plan is self-contained and evidence-gated.
- Decisions are captured in `.planning/DECISIONS.md`; changes require a superseding entry.
