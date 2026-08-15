---
phase: "S02-sm120-baseline"
plan: "S02-03"
type: "verification"
wave: 3
depends_on: [S02-02]
files_modified:
  - tests/{gpu,integration}/sm120/**
  - tools/runtime_trace/**
  - .github/workflows/gpu-smoke.yml
  - docs/gpu-validation.md
  - .planning/understanding-packets/GATE-B.md
  - .planning/{UNDERSTANDING.md,STATE.md}
autonomous: false
understanding_gate: "B"
requirements_addressed: [BCK-001, BCK-003, BCK-004, BCK-005, KER-001, QUA-002, GOV-002, GOV-003, GOV-004]
must_haves:
  truths:
    - "Invalid plans cannot allocate or launch."
    - "Decode steady state allocates nothing and performs no implicit device-wide sync."
    - "Lifecycle and bounds are clean under target-supported sanitizers."
  artifacts:
    - "GPU smoke lane and target qualification record"
    - "Hot-path allocation/synchronization trace assertion"
    - "Sanitizer/lifecycle evidence bundle"
    - "Gate B Understanding Packet linked from the ledger"
---

# S02-03 — GPU Safety and Hot-Path Qualification

## Objective

Turn architectural hot-path and lifecycle rules into blocking executable evidence on an RTX 5090.

## Tasks

1. **Build low-overhead runtime trace instrumentation**
   - Count/tag host/CUDA allocations, frees, module resolution, filesystem access, plan selection, stream/event creation, and device-wide synchronization by lifecycle phase.
   - Compile trace hooks out or make them branch-free/no-op in release; never format logs in the token loop.
   - Define expected construction/prefill/decode/teardown event envelopes.

2. **Create invalid-plan preallocation tests**
   - Feed corrupt capabilities, buffer overlaps, oversized workspaces, unknown kernel IDs, bad launches and cyclic dependencies.
   - Assert zero device allocations/launches occurred before rejection and errors identify the violated contract.

3. **Create steady-state decode trace tests**
   - Warm a synthetic session, then execute many decode steps.
   - Assert no heap/CUDA allocation/free, filesystem, JIT/module load, registry mutation, model branch, or device-wide sync after steady-state boundary.
   - Retain bounded trace summary on failure.

4. **Run sanitizer and lifecycle matrix**
   - Exercise boundary shapes, long repeated state updates, session rebuild after clean teardown, injected CUDA error, and OOM/resource rejection.
   - Run compute-sanitizer modes supported by the target/toolchain and host ASan/UBSan where compatible.
   - Minimize and persist any failing seed/plan.

5. **Establish GPU smoke workflow**
   - Provision exclusive target runner assumptions, capability preflight, timeout, artifact upload, and explicit skip/fail rules.
   - Keep performance out of this lane; it proves compatibility and correctness only.
   - Document local reproduction and hardware/toolchain qualification.

## Verification

- Run smoke/sanitizer matrix twice on target hardware.
- Deliberately add a test-only decode allocation and confirm the trace gate fails.
- Deliberately corrupt a plan and confirm zero CUDA allocation/launch.
- Confirm runner metadata captures target/toolchain identity without publishing sensitive machine data.

## Completion Evidence

- RTX 5090 qualification manifest.
- Passing sanitizer and allocation/synchronization trace bundle.
- GPU smoke workflow run and documented reproduction command.

## Understanding Gate B

Create the L2 packet from a one-token execution trace, the tiny reference-transformer intermediates, and tensor/plan shapes derived from the Qwen config pinned during S03 preparation. The hands-on change must have a prediction recorded before execution. Mark Gate B reached, update both state files, and proactively emit `UNDERSTANDING STATUS` before work could reach Gate C with Gate B still outstanding.
