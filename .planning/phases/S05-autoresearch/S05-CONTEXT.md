# S05: Autoresearch and Decode Experiments — Context

**Status:** Planned
**Depends on:** S04
**Critical-path role:** Makes optimization repeatable and safe, then provides the machinery for controlled speculative-decoding research.

<domain>
## Phase Boundary

Implement declarative experiments, isolated patch execution, immutable workload capture, correctness/determinism/environment/statistical gates, replay, promotion/rollback, and DecodeStrategy conformance. Add a speculative state-machine substrate and a DSpark research track only after its primary semantics are pinned.
</domain>

<decisions>
## Locked Decisions

- [D-006] Correctness gates always precede performance ranking.
- [D-008] DSpark is speculative decoding behind `DecodeStrategy`, not attention.
- [D-010] Experiments and promotion decisions produce immutable, auditable evidence.
- Search budgets and tunable domains are bounded; candidates cannot edit tests, tolerances, benchmark semantics, or evidence validators.
- Thermal/power invalidity, nondeterminism, insufficient samples, or correctness failure blocks promotion.
- Promotion is a reviewable patch/commit with one-step rollback.

### Executor Discretion

- Exact isolation mechanism and queue implementation.
- Statistical test/robust summary after simulation against false-positive/negative fixtures.
- Exact DSpark proposal/verifier mechanics only after pinned primary evidence; classification is locked, details are not.
</decisions>

<understanding>
## Understanding Gate

**Level:** L1 for individual mutations; conditional L2 for experimental methodology.

The user owns why an experiment is valid: noise, warmup, environment rejection, sampling, uncertainty, practical significance, and workload scope. A material change to those decision rules triggers a methodology mini-gate; routine candidate mutations do not. `S05-03` also creates Gate D's packet on vanilla speculative decoding economics for presentation at the S05→S06 transition.
</understanding>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — autoresearch pipeline and DecodeStrategy ownership.
- `.planning/RESEARCH.md` — candidate template, search axes and DSpark caution.
- `.planning/BENCHMARKS.md` — environment/correctness/run validity.
- `.planning/REQUIREMENTS.md` — RES-001–005 and DEC-001–004.
- `.planning/RISKS.md` — R-07, R-08, R-16, R-17.
- `.planning/UNDERSTANDING-GATES.md` — conditional methodology gate and Gate D handoff.
</canonical_refs>

<deferred>
## Deferred

Unattended merging to main, distributed experiment farms, training/search over model weights, public DSpark performance claims, server scheduling, and arbitrary code-generation agents.
</deferred>
