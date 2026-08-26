# S03F-06 Dual-5090 Text Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reference-equivalent text-only greedy Flash-Next generation on 2x RTX 5090 through a real `.sinf` artifact and deterministic multi-device Physical Plan.

**Architecture:** Compose already-qualified PLE, MoE, GDN, QSA, gated residual, typed state, artifact residency, and explicit transfer commands into the full 48-layer text graph. No performance-specific fusion is required.

**Tech Stack:** SuperInfer `.sinf`, C++20/CUDA runtime, pinned Transformers/reference implementation, dual RTX 5090.

**Spec:** `.planning/FLASH-NEXT-DESIGN.md`

## Global Constraints

- Text-only; no vision or MTP.
- Greedy deterministic generation first.
- No model-name branches in runtime/kernels.
- Runtime must execute from artifact-bound physical buffers rather than checkpoint-side helpers.

---

### Task 1: Emit and materialize the complete text Physical Plan

**Files:**
- Modify: `python/superinfer/convert/flash_next.py`
- Modify: `backends/sm120/compiler/specializer.h`
- Modify: `tests/unit/sm120/compiler/specializer_test.cpp`

- [ ] Add failing tests asserting every executable semantic op has a physical provider, every persistent tensor is artifact-bound, every state slot is physically represented, and every command has a placement domain.
- [ ] Compile the full text graph using the S03F-01 capacity policy and S03F-02 partitioner.
- [ ] Reject plans with accidental host expert buffers or implicit transfers.
- [ ] Verify deterministic plan/artifact fingerprints.
- [ ] Commit as `feat(S03F-06): materialize full Flash-Next text plan`.

### Task 2: Cross-device/state continuation differential

**Files:**
- Create: `tools/flash_next_dual_gpu_differential.py`
- Modify: `tests/gpu/sm120/cuda_plan_executor_test.cu`

- [ ] Compare one prefill segment and at least two subsequent decode steps against the pinned reference.
- [ ] Assert KV/GDN/gated-residual states after each step and verify the cross-device activation after the partition boundary.
- [ ] Assert PLE table remains host-resident and report actual sparse H2D bytes.
- [ ] Assert expert-transfer behavior matches the S03F-01 ADR.
- [ ] Commit as `test(S03F-06): qualify dual-device continuation`.

### Task 3: Greedy token qualification

**Files:**
- Create: `artifacts/S03F/flash-next-greedy-golden.json`
- Modify: `tools/flash_next_dual_gpu_differential.py`

- [ ] Run fixed tokenized prompts and compare logits/argmax for token 0 before testing longer generation.
- [ ] Extend to a fixed N-token greedy sequence, checking token IDs and selected logit/intermediate checkpoints.
- [ ] Record model/artifact SHA, code SHA, CUDA/driver/GPU identities, placement plan hash, and numerical tolerances.
- [ ] Run full CPU validation, full RTX 5090 CUDA suite, and compute-sanitizer reduced fixture.
- [ ] Commit as `test(S03F-06): prove Flash-Next greedy correctness`.

### Task 4: Close S03F and hand off to S04

**Files:**
- Create: `.planning/phases/S03F-flash-next/S03F-06-SUMMARY.md`
- Modify: `.planning/STATE.md`
- Modify: `.planning/ROADMAP.md`

- [ ] Record final residency, per-device bytes, PLE residency, transfer topology, state evidence, and golden generation hashes.
- [ ] Explicitly separate correctness evidence from performance claims.
- [ ] Mark S03F complete only when text-only dual-GPU greedy generation is reference-equivalent.
- [ ] Hand off optimization questions to S04/S05 rather than optimizing inside S03F.
- [ ] Commit as `docs(S03F-06): close Flash-Next correctness phase`.
