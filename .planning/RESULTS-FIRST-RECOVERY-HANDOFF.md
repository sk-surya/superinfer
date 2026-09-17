# Results-First Recovery — ai90 Handoff

## Authority

The repo owner approved `.planning/RESULTS-FIRST-RECOVERY-DESIGN.md` and explicitly directed autonomous continuation without further approval pauses. This applies to the bounded recovery sprint defined in `.planning/RESULTS-FIRST-RECOVERY-PLAN.md`. Do not mark understanding gates user-passed; retain them as study evidence under D-014.

## Immediate objective

Produce the earliest defensible evidence that SuperInfer's specialization thesis improves real Qwen3.8-27B execution on RTX 5090.

## Execution order

1. Reconcile live local `.planning/ROADMAP.md` and `.planning/STATE.md` to D-020/results-first ordering.
2. Execute S03-R same-artifact correctness closure.
3. If S03 closes, immediately capture R01 ugly baseline + profile.
4. Select exactly one dominant actionable decode bottleneck from the profile.
5. Write a narrow R02 child plan from the actual files/functions/profile evidence, self-review it, and execute without asking for approval.
6. Run R03 same-manifest before/after correctness + second-session reproduction.
7. Stop only at the success/hard-blocker condition in `.planning/RESULTS-FIRST-RECOVERY-PLAN.md`.

## Anti-drift rules

- Do not resume one-off S03 numerical hypotheses unless same-artifact evidence identifies a concrete divergence.
- Do not start S03F-02+ before R03.
- Do not build the full S04 portfolio.
- Do not build full autoresearch yet.
- Do not optimize a kernel because it looks interesting; profiler critical-path share chooses the target.
- Do not claim progress from a microbenchmark alone; R03 requires reproduced end-to-end improvement.
- Do not loosen a numerical threshold merely to make a run green.
- Preserve user-owned processes/GPU workloads and unrelated untracked artifacts.

## Reporting contract

At each major gate report only:

```text
CURRENT GATE:
EVIDENCE:
DECISION:
BLOCKER: none | <precise blocker>
NEXT ACTION:
```

Continue automatically after reporting unless the stop condition is met.

## Starting references

- `.planning/RESULTS-FIRST-RECOVERY-DESIGN.md`
- `.planning/RESULTS-FIRST-RECOVERY-PLAN.md`
- `.planning/DECISIONS.md` D-006, D-014, D-019, D-020
- `.planning/phases/S03-qwen38-e2e/S03-03-REVIEW-LATEST.md`
- `tools/qwen38_nvfp4_e2e_reference.py`
- `tools/qwen38_s03_acceptance.py`
- `AGENTS.md`

Do not reinterpret the sprint as a planning exercise. Execute it.