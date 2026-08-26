# S03F-02 Multi-Device Physical Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend target specialization and runtime execution to deterministic two-device plans with explicit transfers and no model-specific placement logic.

**Architecture:** Add placement domains to physical resources, explicit kernel/transfer/barrier command variants, per-device arenas/streams, and compiler-selected contiguous layer placement. Model frontends remain device-agnostic.

**Tech Stack:** C++20, CUDA, existing Physical Plan/MemoryPlanner/SM120 executor.

**Spec:** `.planning/FLASH-NEXT-DESIGN.md`

## Global Constraints

- Blocked until S03 closes.
- Initial strategy is contiguous pipeline/layer partitioning only.
- No tensor-parallel collectives.
- Placement is derived from packed bytes and headroom constraints; never hard-code 24/24.
- Transfer commands are first-class Physical Plan commands.

---

### Task 1: Add placement-domain contracts

**Files:**
- Modify: `include/superinfer/base/memory_space.hpp`
- Modify: `include/superinfer/ir/physical_plan.hpp`
- Modify: `tests/unit/ir/lowered_plan_test.cpp`

**Interfaces:**
- Produces: `PlacementDomain`, `BufferPlacement`, `ExecutionPlacement`, and explicit command variants for kernel/transfer/barrier execution.

- [ ] Write failing tests for device ordinal validation, host/pinned/mmap placement, invalid cross-domain buffers, transfer source/destination validation, and serialization determinism.
- [ ] Run focused CTest and verify failure.
- [ ] Implement the minimal typed placement/command contracts without CUDA calls in IR headers.
- [ ] Run focused CTest and architecture dependency checks.
- [ ] Commit as `feat(S03F-02): add physical placement domains`.

### Task 2: Extend memory planning to per-domain arenas

**Files:**
- Modify: `include/superinfer/compiler/memory_planner.h`
- Modify: `src/compiler/memory_planner.cc`
- Modify: `tests/unit/sm120/compiler/memory_planner_test.cpp`
- Modify: `tests/property/sm120/compiler/memory_planner_property_test.cpp`

**Interfaces:**
- Produces: allocation plans scoped by placement domain/device ordinal with independent memory budgets.

- [ ] Add failing tests proving GPU0/GPU1 budgets are independent and host/pinned allocations do not consume device arenas.
- [ ] Implement domain-aware allocation while preserving the single-device API behavior used by S03.
- [ ] Run unit/property tests and full CPU validation.
- [ ] Commit as `feat(S03F-02): plan per-device memory arenas`.

### Task 3: Compile contiguous layer placement and transfer boundaries

**Files:**
- Modify: `backends/sm120/compiler/specializer.h`
- Create: `backends/sm120/compiler/device_placement.h`
- Create: `tests/unit/sm120/compiler/device_placement_test.cpp`
- Modify: `tests/unit/sm120/compiler/specializer_test.cpp`

**Interfaces:**
- Produces: deterministic contiguous partition selected from layer packed bytes and configured headroom, plus one explicit inter-device activation transfer at each partition boundary.

- [ ] Write failing tests where uneven layer sizes make 24/24 suboptimal and where no feasible partition exists.
- [ ] Implement the offline partition solver minimizing max device occupancy subject to headroom.
- [ ] Emit device-local kernel commands and explicit transfer commands without inspecting model names.
- [ ] Verify plan dump/fingerprint determinism.
- [ ] Commit as `feat(S03F-02): specialize dual-device schedules`.

### Task 4: Execute transfer commands on CUDA

**Files:**
- Modify: `backends/sm120/runtime/cuda_ownership.cuh`
- Modify: `backends/sm120/runtime/cuda_plan_executor.cuh`
- Modify: `tests/gpu/sm120/cuda_plan_executor_test.cu`

**Interfaces:**
- Consumes: physical placement/transfer commands.
- Produces: per-device streams/arenas and peer-copy execution with staged pinned-host fallback.

- [ ] Add a failing two-device fixture that transforms data on GPU0, transfers it, transforms it on GPU1, and validates the result.
- [ ] Implement peer-access probing and `cudaMemcpyPeerAsync` when available; implement explicit pinned-host staged fallback when peer copy is unavailable.
- [ ] Ensure all allocations occur during session construction, not steady-state execution.
- [ ] Run RTX 5090 CUDA tests and compute-sanitizer subset.
- [ ] Commit as `feat(S03F-02): execute multi-device physical plans`.

### Task 5: Close multi-device gate

**Files:**
- Create: `.planning/phases/S03F-flash-next/S03F-02-SUMMARY.md`
- Modify: `.planning/STATE.md`

- [ ] Record placement policy, peer/fallback evidence, arena usage, transfer sizes, and tests.
- [ ] Confirm existing single-device Qwen S03 fixtures still pass unchanged.
- [ ] Commit as `docs(S03F-02): close multi-device plan gate`.
