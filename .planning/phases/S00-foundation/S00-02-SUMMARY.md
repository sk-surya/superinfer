---
phase: S00-foundation
plan: S00-02
subsystem: testing
tags: [python, unittest, ci, evidence, packaging]
requires:
  - phase: S00-foundation
    provides: C++20 contract targets and CPU CTest harness
provides:
  - Dependency-light typed Python package and stable CLI shell
  - Deterministic seed/workspace/evidence/subprocess/reference fixture helpers
  - CPU fast/full CI workflows and one-command local validation
affects: [S01-artifact-ir, S02-sm120-baseline, S03-qwen38-e2e]
actuals:
  tokens: 2700
  tasks: 6
  commits: 1
tech-stack:
  added: [Python 3.12+, setuptools PEP 517, unittest]
  patterns: [optional extras, standard-library CPU tests, structured failure evidence, canonical golden fixtures]
key-files:
  created:
    - pyproject.toml
    - python/superinfer/cli.py
    - python/superinfer/reference.py
    - python/superinfer/support/evidence.py
    - python/superinfer/support/seeds.py
    - python/superinfer/support/subprocesses.py
    - tools/check_python.py
    - tools/validate.py
    - .github/workflows/fast.yml
    - .github/workflows/full-cpu.yml
    - docs/development.md
  modified: []
key-decisions:
  - "Use unittest and dependency-free checks as the guaranteed CPU path; pytest/ruff/mypy/build remain named optional dev extras."
  - "Invoke setuptools.build_meta directly for wheel validation so the repository build directory cannot shadow an optional build module."
requirements-completed: [ARCH-008, QUA-001, QUA-003]
coverage:
  - id: D1
    description: "Core Python package and CLI import without GPU/model frameworks"
    requirement: ARCH-008
    verification:
      - kind: unit
        ref: "tests/unit/test_support.py#test_core_import_does_not_load_gpu_or_model_frameworks"
        status: pass
    human_judgment: false
  - id: D2
    description: "Deterministic seeds, workspaces, evidence, subprocess diagnostics, reference comparison, and fixtures"
    requirement: QUA-003
    verification:
      - kind: unit
        ref: "tests/unit/test_support.py; tests/property/test_fixtures_property.py; tests/golden/test_fixture_golden.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "CPU fast/full validation, sanitizer lane, package wheel, and install-tree consumer"
    requirement: QUA-001
    verification:
      - kind: integration
        ref: "python3 tools/validate.py --full"
        status: pass
    human_judgment: false
duration: 30min
completed: 2026-08-15
status: complete
---

# Phase S00 Plan S00-02 Summary

**Dependency-light Python control plane with deterministic test evidence and reproducible CPU CI**

## Performance

- **Duration:** 30 min
- **Started:** 2026-08-15T03:10:00Z
- **Completed:** 2026-08-15T03:14:45Z
- **Tasks:** 6
- **Files modified:** 27

## Accomplishments

- Added the typed Python package, stable exit-category CLI, optional model/GPU extras, and package metadata.
- Added independent reference comparison, deterministic RNG restoration, temporary evidence workspaces, JSON/JSONL writers, subprocess failure records, bounded property checks, and golden fixtures.
- Added local fast/full validation, CPU fast/full GitHub workflows, pre-commit hooks, wheel validation, and an install-tree C++ consumer.
- Retained a failed wheel-validation JSON artifact while fixing the module-shadowing issue, preserving actionable failure evidence.

## Task Commits

1. **S00-02 Python/test/CI foundation** - `9cc7c2c` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Optional build module shadowed by generated build directory**
- **Found during:** full validation wheel step
- **Issue:** `python -m build` resolved the repository's generated `build/` directory and failed because no `build.__main__` existed.
- **Fix:** Call the configured `setuptools.build_meta` backend directly for wheel generation.
- **Verification:** `python3 tools/validate.py --full` passed, and the original failure JSON remains under the ignored validation evidence directory.
- **Committed in:** `9cc7c2c`

**Total deviations:** 1 auto-fixed. **Impact:** Validation is more reproducible and does not require the optional `build` CLI package.

## Issues Encountered

None unresolved.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

S01 can add concrete Semantic/Lowered/Physical IR and `.sinf` schemas without introducing model or
CUDA dependencies into the CPU path. All required S00 CPU evidence is green.

---
*Phase: S00-foundation*
*Plan: S00-02*

