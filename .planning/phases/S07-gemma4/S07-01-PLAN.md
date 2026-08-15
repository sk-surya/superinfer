---
phase: "S07-gemma4"
plan: "S07-01"
type: "feature"
wave: 1
depends_on: [S06-02]
files_modified:
  - frontends/gemma4/**
  - python/superinfer/convert/gemma4.py
  - tests/{unit,golden,property}/frontends/gemma4/**
  - tests/fixtures/gemma4/**
  - docs/models/gemma4-26b-a4b.md
autonomous: false
requirements_addressed: [MOD-005, ARCH-005, FMT-004]
must_haves:
  truths:
    - "Pinned Gemma inputs map to canonical semantics without executor changes."
    - "Real semantic differences are explicit and verified, not approximated as Qwen behavior."
    - "Config/tensor/tokenizer/provenance validation is as strict as Qwen."
  artifacts:
    - "Pinned Gemma source/provenance/license manifest"
    - "Gemma ModelFrontend and semantic/tensor mapping"
    - "Generic semantic gap analysis and fixtures"
---

# S07-01 — Pin and Implement the Gemma Frontend

## Objective

Translate the exact supported Gemma 4 26B-A4B source into verified canonical Semantic IR and explicitly identify any generic semantic gaps.

## Tasks

1. **Pin and audit the model source**
   - Resolve exact upstream repository/revision; hash config/tokenizer/template/indexes and enumerate all tensors/shapes/dtypes/roles.
   - Document topology, attention/RoPE/masking, normalization, FFN/MoE/routing, embedding/head tying, biases, state and optional modalities/heads from pinned reference code.
   - State V0 text-generation scope explicitly if the source exposes broader modalities/capabilities.

2. **Perform semantic gap analysis**
   - Map every behavior to existing Semantic IR operations and identify genuine missing mathematical semantics.
   - For each gap, propose a model-independent operation/attribute, verifier rules, reference formula and impact on lowering/artifact compatibility.
   - Reject Qwen-specific reuse that would change Gemma semantics.

3. **Implement strict Gemma frontend**
   - Validate config/tensor schema with safe bounds and actionable errors.
   - Emit canonical Semantic IR using generic operations, explicit state edges and source provenance.
   - Keep target, storage and kernel policy out of the frontend.

4. **Implement tensor and tokenizer mapping**
   - Produce deterministic source-to-semantic tensor inventory, alias/tie rules and manifest provenance.
   - Capture tokenizer/special-token/template identity and golden prompt IDs.
   - Preserve source licenses and scope limitations.

5. **Create tiny fixtures and generic-op references**
   - Add synthetic legal fixtures for every unique semantic variant.
   - Add independent reference utilities, golden Semantic IR and malformed config/tensor cases.
   - Re-run Qwen semantic fixtures to prove generic changes did not alter its graph.

## Verification

- Semantic gap review is approved before adding generic operations.
- Gemma and Qwen frontends both pass strict/golden/property suites.
- Runtime/executor tree is byte-for-byte unchanged during this plan.
- No Gemma-named operations enter shared IR/provider/runtime APIs.

## Completion Evidence

- Pinned source and license manifest.
- Semantic/tensor/tokenizer mapping and gap-decision record.
- Frontend golden/negative test report and executor unchanged hash.
