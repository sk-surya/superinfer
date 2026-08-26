# S03F-05 QSA and Gated Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Flash-Next sparse-attention selection/execution and four-branch gated residual semantics without coupling either mechanism to model identity.

**Architecture:** QSA index/selection and sparse attention are separate semantic operations. Gated residual read/write behavior is modeled from pinned reference equations and remains independent from attention/MoE kernels.

**Tech Stack:** C++20/CUDA, Semantic/Lowered IR, SM120 provider/runtime, Python reference differentials.

**Spec:** `.planning/FLASH-NEXT-DESIGN.md`

## Global Constraints

- Never hide QSA indexing inside a model-named attention kernel.
- Sparse selection results are explicit data dependencies.
- Gated residual equations must be pinned before selecting the final semantic API.

---

### Task 1: Pin QSA and gated-residual equations

**Files:**
- Modify: `docs/models/flash-next.md`
- Create: `artifacts/S03F/flash-next-qsa-residual-contract.json`

- [ ] Trace the pinned reference implementation for indexer inputs/outputs, compression/block semantics, masks, and four-branch residual read/write equations.
- [ ] Record exact tensor shapes and branch state carried between sublayers.
- [ ] Add source revision/hash to the evidence artifact.
- [ ] Commit as `research(S03F-05): pin sparse attention and residual equations`.

### Task 2: Add semantic contracts

**Files:**
- Modify: `include/superinfer/ir/semantic/module.hpp`
- Modify: `include/superinfer/compiler/semantic_lowering.hpp`
- Modify: `tests/unit/ir/semantic_ir_test.cpp`

- [ ] Write failing verifier tests for invalid indexer budgets/block selections, illegal sparse-attention indices, and mismatched residual branch counts/low-rank dimensions.
- [ ] Implement model-independent QSA index/select + sparse-attention concepts and the minimal generic gated-residual contract justified by Task 1.
- [ ] Preserve all execution-relevant attributes through lowering.
- [ ] Run CPU/architecture tests.
- [ ] Commit as `feat(S03F-05): add sparse attention and gated residual semantics`.

### Task 3: Implement correctness-first CUDA paths

**Files:**
- Modify: `backends/sm120/kernels/baseline/provider.h`
- Create: `backends/sm120/kernels/baseline/sparse_attention_reference.cuh`
- Create: `backends/sm120/kernels/baseline/gated_residual_reference.cuh`
- Modify: `backends/sm120/runtime/cuda_plan_executor.cuh`
- Create: `tests/gpu/sm120/sparse_attention_test.cu`
- Create: `tests/gpu/sm120/gated_residual_test.cu`

- [ ] Add failing fixtures comparing selected blocks and sparse attention against dense masked reference for small deterministic tensors.
- [ ] Add failing fixtures for gated residual read/write and multi-step branch continuation.
- [ ] Implement baseline providers with strict typed-shape contracts.
- [ ] Run GPU tests/compute-sanitizer.
- [ ] Commit as `feat(S03F-05): execute baseline QSA and gated residual`.

### Task 4: Complete GDN/QSA layer differentials

**Files:**
- Create: `tools/flash_next_layer_differential.py`
- Create: `.planning/phases/S03F-flash-next/S03F-05-SUMMARY.md`

- [ ] Compare selected intermediate tensors for one GDN layer and one QSA layer against the pinned reference.
- [ ] Include PLE contribution where applicable, MoE output, residual branch state, and recurrent/KV continuation.
- [ ] Fail on unexplained numerical drift; record explicit tolerances only where quantization requires them.
- [ ] Commit as `test(S03F-05): qualify Flash-Next layer families`.
