---
phase: "S01-artifact-ir"
plan: "S01-01"
type: "feature"
wave: 1
depends_on: [S00-01, S00-02]
files_modified:
  - include/superinfer/ir/semantic/**
  - src/ir/semantic/**
  - tests/{unit,property,golden}/ir/semantic/**
autonomous: true
requirements_addressed: [ARCH-001, ARCH-002, MOD-002, MOD-003]
must_haves:
  truths:
    - "Semantic IR describes model meaning without target launch or storage offsets."
    - "Malformed graphs and model inputs fail with actionable, deterministic diagnostics."
    - "Deterministic dumps make semantic changes reviewable."
  artifacts:
    - "Typed Semantic IR operation/state schema"
    - "Semantic verifier and deterministic canonical dump"
    - "Property and golden tests for required operation families"
---

# S01-01 — Semantic IR and Verification

## Objective

Implement the model-independent semantic vocabulary required by Qwen/Gemma while keeping CUDA/layout/storage facts out of the representation.

## Tasks

1. **Define semantic type and shape system**
   - Implement stable value/tensor/node IDs, scalar dtypes, quantization intent (not storage encoding), static/symbolic dimensions, tensor roles, and graph entry points.
   - Represent state edges explicitly for KV cache and decode state.
   - Make shapes/dtypes immutable after graph construction or mutate only through a checked builder transaction.

2. **Define canonical operations**
   - Add typed operations for embedding, RMSNorm/layer norm where required, RoPE semantics, QKV projection, MHA/GQA/local attention, residuals, gated dense FFN, MoE route/top-k/experts/combine, LM head, and decode logits/sampling inputs.
   - Express optional/variant semantics with explicit attributes and capabilities; avoid Qwen/Gemma-named operations.
   - Document mathematical semantics, axis conventions, broadcast rules, numerical intent, and state effects.

3. **Implement safe construction and immutable graph handoff**
   - Provide a builder that checks unique names/IDs and returns a frozen Semantic IR.
   - Avoid partially valid graphs escaping a failed build.
   - Preserve source-location/provenance references for diagnostics without embedding absolute paths in deterministic dumps.

4. **Implement the semantic verifier**
   - Check definitions/uses, topological/state ordering, entry point signatures, ranks/shapes/dtypes, group/head divisibility, RoPE dimensions, expert/router consistency, residual compatibility, and output contracts.
   - Bound sizes with checked arithmetic and cap attacker-controlled counts.
   - Aggregate useful errors when safe; stop on structural corruption.

5. **Create deterministic dump and fixtures**
   - Canonically order IDs/attributes, normalize float/text representation, and exclude nondeterministic addresses/paths.
   - Build tiny dense, GQA, local-attention, and MoE graphs that exercise the vocabulary.
   - Add golden tests and property mutations that generate both valid and invalid graphs.

## Verification

- Unit tests cover every op verifier and error category.
- Property tests mutate shapes/dtypes/edges/counts and always either verify or return a typed error without crash/overflow.
- Two construction orders for the same canonical graph produce identical dump bytes.
- Dependency test proves semantic IR includes no CUDA headers, launch parameters, file offsets, or provider IDs.

## Completion Evidence

- Checked-in semantic IR schema/API docs.
- Golden dumps for dense, GQA/local-attention, and MoE fixtures.
- Requirement mapping from MOD-003 operations to fixtures/tests.
