# S03F-04 MoE Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Flash-Next routed and shared experts correctly with layer-local persistent residency on the device assigned by S03F-02.

**Architecture:** Preserve separate route/top-k/dispatch/expert/combine semantics. Initial grouped expert compute is correctness-first and uses compile-time resident expert buffers; expert prediction, dynamic caching, and cross-device expert parallelism are excluded.

**Tech Stack:** C++20/CUDA, existing MoE Semantic IR kinds, NVFP4 artifact bindings/provider contracts.

**Spec:** `.planning/FLASH-NEXT-DESIGN.md`

## Global Constraints

- All experts for a layer are resident on that layer's assigned device unless an explicit capacity ADR changes the policy.
- Routing semantics are model/reference derived.
- Correctness gate precedes grouped-kernel optimization.

---

### Task 1: Complete MoE semantic/lowered contracts

**Files:**
- Modify: `include/superinfer/ir/semantic/module.hpp`
- Modify: `include/superinfer/compiler/semantic_lowering.hpp`
- Modify: `include/superinfer/ir/lowered/module.hpp`
- Modify: `tests/unit/ir/semantic_ir_test.cpp`

- [ ] Add failing tests for router logits, top-k count, shared expert, dispatch mapping, and combine weights.
- [ ] Extend existing generic MoE kinds only where Flash-Next requires missing information; do not introduce model-named operations.
- [ ] Preserve authored routing attributes into Lowered IR.
- [ ] Run focused tests.
- [ ] Commit as `feat(S03F-04): complete MoE lowering contracts`.

### Task 2: Materialize expert artifact bindings

**Files:**
- Modify: `python/superinfer/convert/flash_next.py`
- Modify: `backends/sm120/compiler/specializer.h`
- Modify: `tests/unit/test_flash_next.py`
- Modify: `tests/unit/sm120/compiler/specializer_test.cpp`

- [ ] Add failing tests proving routed/shared expert NVFP4 weights and all scale sidecars bind to the correct layer/expert/device.
- [ ] Implement typed artifact-to-physical bindings without copying expert weights between devices at decode time.
- [ ] Verify per-device packed bytes agree with S03F-01 ledger.
- [ ] Commit as `feat(S03F-04): bind resident expert weights`.

### Task 3: Implement correctness-first route/dispatch/compute/combine

**Files:**
- Modify: `backends/sm120/kernels/baseline/provider.h`
- Create: `backends/sm120/kernels/baseline/moe_reference.cuh`
- Modify: `backends/sm120/runtime/cuda_plan_executor.cuh`
- Create: `tests/gpu/sm120/moe_reference_test.cu`

- [ ] Add failing CUDA fixtures with deterministic router logits, ties, top-k ordering, shared expert contribution, and combine weights.
- [ ] Implement route/top-k/dispatch/grouped expert/shared/combine provider commands using existing NVFP4 projection primitives where possible.
- [ ] Reject unsupported shapes/encodings rather than falling back silently.
- [ ] Run GPU differential tests and compute-sanitizer.
- [ ] Commit as `feat(S03F-04): execute baseline MoE path`.

### Task 4: Layer differential

**Files:**
- Create: `tools/flash_next_moe_differential.py`
- Create: `.planning/phases/S03F-flash-next/S03F-04-SUMMARY.md`

- [ ] Compare router logits, selected expert IDs, routing weights, shared-expert output, routed-expert outputs, combine output, and final MoE residual against the pinned reference.
- [ ] Run single-token and short-prefill fixtures.
- [ ] Record exact device residency and assert zero expert-weight transfer commands during execution when the S03F-01 ADR says full residency is required/feasible.
- [ ] Commit as `test(S03F-04): qualify Flash-Next MoE layer`.
