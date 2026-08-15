# S06: Reproducible Performance Proof — Context

**Status:** Planned
**Depends on:** S05
**Critical-path role:** Produces the first credible, reproducible RTX 5090 Qwen3.8-27B performance graph.

<domain>
## Phase Boundary

Build controlled benchmark manifests/runner, baseline adapters, environment validity, raw evidence schemas, report/graph generation, reproduction workflow, and claim audit. Publish only the first narrow Qwen graph. Do not expand model coverage or tune specifically to a new unreviewed workload during measurement.
</domain>

<decisions>
## Locked Decisions

- [D-010] Raw data and environment/correctness evidence are product artifacts.
- Correctness gate and matched semantics are prerequisites for comparison.
- Prefill and decode are reported separately; single-request latency and aggregate throughput are never conflated.
- Graphs are generated from immutable raw data and accompanied by tables/manifests.
- Invalid thermals/power/clocks/samples are rejected, not averaged away.
- Any baseline mismatch is labeled or split into a separate panel, never normalized into a misleading rank.

### Executor Discretion

- Exact first graph context/prompt grid after considering runtime cost and clarity.
- Baseline engines supported, provided versions/configurations are pinned and legally usable.
- Statistical presentation consistent with `.planning/BENCHMARKS.md`.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/BENCHMARKS.md` — mandatory protocol and evidence layout.
- `.planning/PROJECT.md` — first public proof success criterion.
- `.planning/REQUIREMENTS.md` — BEN-001–005 and QUA-002/003/005.
- `.planning/RISKS.md` — R-07, R-08, R-09, R-15, R-17.
</canonical_refs>

<deferred>
## Deferred

Broad model leaderboards, multi-GPU/server concurrency claims, marketing extrapolations, unattended publication, and performance claims for Gemma/DSpark unless separately evidence-gated.
</deferred>
