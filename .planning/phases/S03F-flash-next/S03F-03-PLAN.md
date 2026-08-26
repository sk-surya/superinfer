# S03F-03 PLE Host-Resident Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Flash-Next PLE/N-gram lookup as a first-class host-resident semantic/storage path with sparse asynchronous transfer to the target GPU.

**Architecture:** Token-history hashing/index semantics are pinned from the reference model. PLE storage is independently quantized and host/mmap resident. Only gathered vectors cross H2D through preallocated pinned staging buffers.

**Tech Stack:** C++20, CUDA async copies/events, Python conversion tooling, `.sinf` storage metadata.

**Spec:** `.planning/FLASH-NEXT-DESIGN.md`

## Global Constraints

- Do not model PLE as arbitrary CPU offload.
- Initial PLE is read-only.
- Never transfer the full PLE table per request/token.
- Hash/index semantics must be reference-derived and differentially tested.

---

### Task 1: Define semantic PLE contracts

**Files:**
- Modify: `include/superinfer/ir/semantic/module.hpp`
- Modify: `include/superinfer/compiler/semantic_lowering.hpp`
- Modify: `tests/unit/ir/semantic_ir_test.cpp`

**Interfaces:**
- Produces semantic operations equivalent to `ngram_index`, `ngram_embedding_lookup`, and `ngram_combine`, with explicit table/storage inputs and model-authored attributes.

- [ ] Write failing verifier tests for invalid n-gram widths/head counts/table references and illegal storage semantics.
- [ ] Implement model-independent semantic contracts and deterministic dumps.
- [ ] Run focused tests and architecture checks.
- [ ] Commit as `feat(S03F-03): add semantic n-gram memory contracts`.

### Task 2: Add PLE storage-policy artifact metadata

**Files:**
- Modify: `python/superinfer/convert/flash_next.py`
- Modify: `include/superinfer/artifact/storage_policy.hpp`
- Modify: `tests/unit/test_flash_next.py`
- Modify: `tests/integration/artifact/artifact_test.cpp`

- [ ] Add failing tests for read-only host/mmap residency, independent PLE quantization metadata, offset bounds, and checksum mismatch.
- [ ] Implement deterministic PLE sections/metadata without requiring device materialization at load.
- [ ] Verify large-table inspection remains metadata-only.
- [ ] Commit as `feat(S03F-03): encode host-resident PLE storage`.

### Task 3: Implement sparse gather and pinned staging

**Files:**
- Create: `backends/sm120/runtime/ple_prefetch.cuh`
- Modify: `backends/sm120/runtime/cuda_plan_executor.cuh`
- Create: `tests/gpu/sm120/ple_prefetch_test.cu`

- [ ] Write a failing fixture over a small host-resident quantized table that requests noncontiguous entries and validates gathered/dequantized vectors on device.
- [ ] Preallocate pinned staging and device destination buffers during session construction.
- [ ] Implement CPU index/gather plus asynchronous H2D scheduled on an explicit transfer stream/event dependency.
- [ ] Ensure steady-state execution performs no allocation or filesystem parsing.
- [ ] Run CUDA tests/compute-sanitizer.
- [ ] Commit as `feat(S03F-03): add asynchronous PLE sparse prefetch`.

### Task 4: Differential qualification

**Files:**
- Create: `tools/flash_next_ple_differential.py`
- Create: `.planning/phases/S03F-flash-next/S03F-03-SUMMARY.md`

- [ ] Compare reference indices, retrieved vectors, and combined injection output for fixed token histories.
- [ ] Exercise at least boundary collisions/repeated histories and multi-token continuation.
- [ ] Record transferred bytes/token and overlap timeline as evidence without claiming performance leadership.
- [ ] Commit as `test(S03F-03): qualify PLE lookup path`.
