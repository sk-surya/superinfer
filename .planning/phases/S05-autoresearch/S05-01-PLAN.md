---
phase: "S05-autoresearch"
plan: "S05-01"
type: "feature"
wave: 1
depends_on: [S04-03]
files_modified:
  - schemas/research/**
  - python/superinfer/research/{manifest,isolation,runner,capture,replay}.py
  - tests/{unit,integration}/research/**
  - docs/autoresearch.md
autonomous: true
requirements_addressed: [RES-001, RES-003, RES-005, BEN-001]
must_haves:
  truths:
    - "An experiment cannot change workload semantics or exceed its declared search budget."
    - "Every run is isolated and captures enough identity/state to replay."
    - "Invalid environment or missing evidence is a failed experiment, not a noisy sample."
  artifacts:
    - "Versioned experiment/search-space/run/evidence schemas"
    - "Isolated runner with immutable capture and bounded execution"
    - "Replay command and known-good/known-invalid fixture runs"
---

# S05-01 — Declarative Experiment Capture and Replay

## Objective

Create the safe, reproducible substrate that turns a kernel hypothesis and bounded patch into a fully identified run bundle.

## Tasks

1. **Define versioned research schemas**
   - Specify hypothesis, base commit, allowed file/path scope, tunable domain, maximum candidates/time/builds, target profile, artifact hash, correctness suite, benchmark manifest, seeds and accept rule.
   - Separate immutable workload/correctness definitions from candidate-controlled parameters.
   - Define run state transitions, outcome taxonomy, evidence hashes and schema migration/compatibility.

2. **Implement manifest validation and budget enforcement**
   - Validate all paths against repository scope, all tunables against typed bounds, and all referenced artifacts/manifests by hash.
   - Reject attempts to modify tests, tolerances, benchmark inputs, evidence validators, CI/research control code or out-of-scope files.
   - Enforce wall-clock, candidate/build, disk/log and GPU timeout budgets.

3. **Implement isolated candidate execution**
   - Create a clean isolated Git worktree/snapshot from the declared base, apply a recorded patch, and use separate build/result directories.
   - Never run dirty/unrecorded candidate state; capture status/diff/submodules/dependencies.
   - Serialize exclusive GPU ownership and cleanly terminate timed-out/error runs without touching user work.

4. **Capture environment and immutable evidence**
   - Record toolchain/driver/GPU/power/clocks/thermals/host/software, build commands/logs, artifact and provider/tuning IDs, seeds, workload and raw outputs.
   - Hash every evidence member and write atomically; redact only documented sensitive machine fields.
   - Mark environment validity separately from performance result.

5. **Implement deterministic replay**
   - Reconstruct candidate from base+patch, verify all input hashes, run the same gates/manifest, and compare structured outcomes.
   - Allow a diagnostic replay mode but clearly distinguish it from evidence-valid reproduction.

6. **Test with fixture experiments**
   - Known-good no-op/compile-constant experiment.
   - Disallowed benchmark/test/tolerance edit, over-budget search, dirty patch, missing artifact, invalid thermal/environment and timeout cases.
   - Confirm cleanup does not delete or rewrite the developer worktree.

## Verification

- Schema property tests and migration/unknown-version negatives pass.
- Two known-good replays produce equivalent identified outcomes (allowing normalized run IDs/timestamps).
- Every malicious/out-of-scope fixture fails before GPU benchmark.
- Interrupt/timeout leaves repository and GPU queue clean, with a final typed outcome.

## Completion Evidence

- Example experiment and complete replayable evidence bundle.
- Threat/boundary test report for path/budget/workload protection.
- Operator documentation for create/run/inspect/replay/clean workflows.
