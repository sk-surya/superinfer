---
phase: "S03-qwen38-e2e"
plan: "S03-01"
type: "feature"
wave: 1
depends_on: [S01-03, S02-03]
files_modified:
  - frontends/qwen38/**
  - python/superinfer/convert/qwen38.py
  - tests/{unit,golden,property}/frontends/qwen38/**
  - tests/fixtures/qwen38/**
  - docs/models/qwen38-27b.md
autonomous: false
requirements_addressed: [MOD-001, MOD-002, MOD-003, MOD-004, FMT-004]
must_haves:
  truths:
    - "The supported Qwen source is pinned by immutable identity, not a marketing name."
    - "The frontend emits only canonical model-independent Semantic IR."
    - "Tensor/config/tokenizer mismatches fail before compilation with exact diagnostics."
  artifacts:
    - "Pinned source/provenance and license manifest"
    - "Qwen3.8-27B ModelFrontend and normalized tensor mapping"
    - "Synthetic and extracted golden semantic/tokenizer fixtures"
---

# S03-01 — Pin and Implement the Qwen Frontend

## Objective

Translate the exact supported Qwen3.8-27B source checkpoint into verified canonical Semantic IR and an explicit tensor/tokenizer inventory.

## Tasks

1. **Pin and audit the source model**
   - Resolve the exact upstream repository ID and immutable revision with the user/project owner.
   - Hash config, tokenizer, chat template, index files and a complete tensor name/shape/dtype inventory; record license/redistribution terms.
   - Read the pinned reference implementation to document layer topology, attention/RoPE behavior, FFN or MoE behavior, normalization, biases, tying, optional heads and decode state.
   - Treat any mismatch with packet assumptions as a recorded decision/update, not a silent implementation choice.

2. **Build normalized frontend input/schema validation**
   - Parse config with explicit required/optional fields and safe numeric bounds.
   - Validate exact tensor presence, uniqueness, shapes/dtypes, shard indexes, byte sizes and roles before reading bulk payloads.
   - Reject unknown architecture values unless explicitly declared metadata-only.
   - Emit stable error codes that point to field/tensor and expected vs actual contract.

3. **Implement ModelFrontend emission**
   - Map the pinned architecture to existing canonical Semantic IR operations/state edges.
   - Add a generic semantic operation only when the behavior cannot be expressed correctly; document mathematical semantics and add independent tests.
   - Preserve semantic-origin references from graph values to source tensors/config without absolute path nondeterminism.
   - Keep target layout, quantization storage, CUDA and kernel choices out of the frontend.

4. **Map tensors and tokenizer/template metadata**
   - Produce a deterministic normalized tensor mapping consumed by StoragePolicy/converter.
   - Capture tying/alias behavior explicitly and validate it.
   - Preserve tokenizer files, special token IDs, normalization/pretokenization identity, chat-template hash and license metadata in the artifact manifest.

5. **Create small independent fixtures**
   - Extract/configure legal tiny synthetic fixtures for every semantic variant in the pinned model; do not check in restricted real weights.
   - Capture reference tokenizer prompt IDs, semantic IR dumps, tensor maps and selected operation outputs.
   - Add malformed config/inventory/property tests for missing, extra, wrong-shaped and unsafe values.

## Verification

- Re-run source inventory generation twice and compare canonical hashes.
- Frontend Semantic IR passes the generic verifier and contains no model-named operations.
- Tokenizer/chat-template test inputs match the pinned trusted reference IDs.
- All negative fixtures fail before bulk tensor conversion and identify the contract.

## Completion Evidence

- Model support manifest with immutable revision/hashes/license.
- Config/tensor/semantic mapping document and golden fixtures.
- Review attestation that no Qwen dependency entered runtime/provider layers.
