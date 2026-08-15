---
phase: "S02-sm120-baseline"
plan: "S02-01"
type: "feature"
wave: 1
depends_on: [S01-02, S01-03]
files_modified:
  - backends/sm120/compiler/**
  - backends/sm120/target/**
  - include/superinfer/compiler/{target,memory_planner}.h
  - src/compiler/memory_planner.cc
  - tests/{unit,property,golden}/sm120/compiler/**
autonomous: true
requirements_addressed: [BCK-001, BCK-002, BCK-003, BCK-005]
must_haves:
  truths:
    - "Target compatibility and resource budgets are explicit compile/validation inputs."
    - "Memory arenas are derived from liveness with checked, non-overlapping offsets."
    - "The exact model/target profile resolves workspace and schedule before runtime."
  artifacts:
    - "`sm120` target probe/profile and compatibility fingerprint"
    - "Deterministic memory/workspace planner"
    - "Synthetic Lowered IR to Physical Plan compiler with resource ledger"
---

# S02-01 — `sm120` Target, Memory Planning, and Specialization

## Objective

Compile verified synthetic Lowered IR into resource-valid `sm120` Physical Plans and prove all memory/scheduling decisions happen before execution.

## Tasks

1. **Implement target capability probe and profile**
   - Query device/runtime/toolchain capabilities outside the hot path and normalize them into a versioned target profile.
   - Capture compute capability, supported kernel catalog/ABI, shared-memory/register/launch constraints, alignment, device memory budget and relevant feature flags.
   - Support an offline declared profile for conversion; runtime later checks it against the actual device.
   - Reject non-`sm_120a` or incompatible catalog/toolchain with typed diagnostics.

2. **Implement checked memory/workspace planning**
   - Classify persistent weights, constants, KV state, decode state, activation/scratch lifetimes, host staging, and device arenas.
   - Compute liveness and aligned reuse with checked arithmetic; make aliasing explicit and safe.
   - Honor maximum sequence/batch/context/strategy declarations and device memory budget.
   - Emit human/machine-readable memory ledger explaining every allocation and peak.

3. **Implement synthetic target lowering**
   - Resolve layouts, padding, dtypes, KV form, workspace, stream/event schedule, capability requirements, and candidate IDs for synthetic operations.
   - Keep selection deterministic using a pinned provider catalog/tuning input.
   - Record all decisions and rationales in plan provenance/dumps.

4. **Build Physical Plan materialization**
   - Assign arena offsets, constants, command dependencies, entry schedules, stable kernel IDs and checked launch descriptors.
   - Re-run the generic Physical Plan verifier plus target-specific resource/capability validation.
   - Ensure failure leaves no partially valid plan/artifact.

5. **Test adversarial resource cases**
   - Cover overflow, impossible alignment, arena overlap, excessive context/KV, invalid stream/event dependencies, unsupported capability, unknown kernel ID and insufficient memory.
   - Property-test liveness allocation against a simple non-reusing planner and interval oracle.

## Verification

- Deterministic plan/memory-ledger hashes across clean processes.
- Offline vs live target profile compatibility tests on available RTX 5090; record hardware skip otherwise.
- Adversarial plans fail before runtime construction/device allocation.
- Architecture test proves compiler target code does not depend on model frontends.

## Completion Evidence

- `sm120` capability/profile schema and probe output.
- Golden Physical Plans and memory ledgers for small dense/attention/MoE fixtures.
- Resource-negative test matrix.
