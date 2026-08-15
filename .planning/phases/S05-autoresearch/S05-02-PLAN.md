---
phase: "S05-autoresearch"
plan: "S05-02"
type: "feature"
wave: 2
depends_on: [S05-01]
files_modified:
  - python/superinfer/research/{gates,statistics,promotion,rollback}.py
  - schemas/research/promotion.schema.json
  - tests/{unit,integration}/research/{gates,statistics,promotion}/**
  - artifacts/S05/fixtures/**
autonomous: true
requirements_addressed: [RES-002, RES-003, RES-004, KER-004, BEN-001]
must_haves:
  truths:
    - "Candidates are ranked only after every correctness/safety/environment gate passes."
    - "Statistical and practical significance rules resist noise fixtures."
    - "Promotion and rollback are auditable, atomic and reviewable."
  artifacts:
    - "Ordered gate engine and outcome schema"
    - "Validated performance decision/statistics module"
    - "Promotion record/commit workflow and rollback rehearsal"
---

# S05-02 — Correctness-First Gates and Promotion

## Objective

Implement the fail-closed decision pipeline that determines whether an isolated research candidate is eligible, faster, and safe to promote.

## Tasks

1. **Implement ordered gate orchestration**
   - Enforce schema/budget -> build/static -> unit/property -> differential/model correctness -> sanitizer -> deterministic replay -> environment validation -> benchmark sampling -> statistical/practical decision.
   - Each gate consumes/produces typed evidence and cannot be skipped except by a manifest policy that is itself disallowed for required gates.
   - Stop performance work immediately on earlier failure and record the precise terminal outcome.

2. **Implement correctness and determinism gates**
   - Reuse registered S04 kernel/full-model suites and immutable numerical contracts.
   - Compare tokens/state hashes and selected intermediates; reject flakiness across repetitions.
   - Require sanitizer/bounds lane based on candidate path/capability metadata.

3. **Implement environment/sample validity**
   - Validate exclusive GPU, power/clock policy, thermals, errors/fallbacks, warmup stability, minimum samples and timing completeness.
   - Retain outliers/rejections with reasons; never silently delete samples.

4. **Choose and validate decision statistics**
   - Use robust summaries and repeat confirmation with a practical speedup threshold defined in the manifest.
   - Simulate stable win/no-change/regression, high variance, drift, warmup contamination and thermal-throttle fixtures.
   - Report uncertainty and scoped applicability; no binary “faster” without workload identity.

5. **Implement promotion records**
   - Bundle base/candidate commit, patch, manifest, all gate results, raw samples, summaries, decision, reviewer status and target catalog update.
   - Generate an atomic, scoped candidate commit only after approval policy; never merge/push autonomously.
   - Update provider/tuning catalog by stable ID and link its evidence record.

6. **Rehearse rollback and rejected-candidate retention**
   - Revert a fixture promotion and prove baseline catalog/artifact rebuild is restored.
   - Retain valuable negative result metadata without accumulating uncontrolled build/worktree data.

## Verification

- Known wrong-but-fast, flaky, thermal-invalid, noisy false-win and out-of-scope candidates are rejected.
- Known correct stable winner reaches “eligible for review,” not autonomous merge.
- Promotion bundle validates independently and rebuilds the same selected catalog entry.
- Rollback leaves no dangling catalog/evidence references.

## Completion Evidence

- Gate matrix with passed/rejected fixtures.
- Statistical validation report.
- Full promotion and rollback rehearsal bundle.
