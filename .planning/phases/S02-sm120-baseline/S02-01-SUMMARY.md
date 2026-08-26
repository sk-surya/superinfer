---
phase: S02-sm120-baseline
plan: S02-01
subsystem: sm120-compiler
tags: [sm120a, target-profile, memory-planner, physical-plan, liveness]
requires:
  - phase: S01-artifact-ir
    provides: Semantic IR, Lowered IR, Physical Plan, deterministic artifact contracts
provides:
  - Validated offline `sm_120a` target profile and compatibility fingerprint
  - Checked liveness-based device/workspace memory planner with deterministic ledgers
  - Synthetic Lowered IR to Physical Plan specialization with stable baseline kernel IDs
affects: [S02-02, S02-03, S03-02]
requirements-completed: [BCK-001, BCK-002, BCK-003, BCK-005]
status: complete
completed: 2026-08-26
---

# S02-01 — `sm120` Target, Memory Planning, and Specialization

## Accomplishments

- Added a typed `TargetProfile` for declared RTX 5090 `sm_120a` compilation, including target
  validation, catalog ABI identity, resource limits, and fail-closed compatibility checks.
- Added a CPU-only `MemoryPlanner` with checked arithmetic, separate device/workspace budgets,
  half-open command lifetimes, aligned first-fit reuse, deterministic ID ordering, and a machine-
  readable memory ledger.
- Added the `sm120::Specializer`, which converts verified Lowered IR into an immutable Physical Plan
  using only capability/operation facts and stable baseline kernel IDs. No model frontend or model
  family appears in the backend boundary.
- Added unit, randomized property, adversarial, and golden tests for profile compatibility, overflow,
  budget rejection, alignment, liveness reuse, deterministic ledgers, and plan specialization.

## Evidence

- `python3 tools/validate.py --full` — PASS: Python checks/tests, CPU build/CTest, install consumer,
  sanitizer build/CTest, and wheel build.
- Fresh `/srv` CTest run — 13/13 tests passed, including target profile, memory planner/property/
  golden, and specializer tests.
- The first `cuda-sm120a` configure used a stale cache and did not discover the installed toolkit.
  An explicit configure with `/usr/local/cuda-13.1/bin/nvcc` enabled CUDA 13.1 and `sm_120a`.
  The follow-up S02-02 CUDA ownership smoke launches on the RTX 5090 and passes; this is now
  recorded as GPU evidence rather than a toolchain blocker.

## Deliberate boundary

The current Lowered IR tensor descriptor does not carry semantic tensor roles, so S02-01 does not
guess whether a tensor is a persistent weight. Synthetic tensors are recorded as activations. Weight,
KV, and decode-state classification belongs in the later storage/model lowering contracts where that
information is explicit.

## Next plan

S02-02 implements the explicit runtime ownership layer, baseline provider registry, minimal executor,
and independent CPU/reference executor. The CUDA ownership boundary is now smoke-qualified; physical
plan kernel execution and differential GPU graphs remain the next implementation evidence.
