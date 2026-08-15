---
phase: "S07-gemma4"
plan: "S07-03"
type: "verification"
wave: 3
depends_on: [S07-02]
files_modified:
  - tests/corpora/gemma4/**
  - tests/{integration,gpu}/gemma4/**
  - artifacts/S07/**
  - docs/models/gemma4-26b-a4b.md
  - docs/architecture/extensibility-proof.md
  - .planning/understanding-packets/S07-ARCHITECTURE-AUDIT.md
  - .planning/{UNDERSTANDING.md,STATE.md}
autonomous: false
understanding_gate: "conditional-architecture-audit"
requirements_addressed: [MOD-005, ARCH-005, ARCH-006, QUA-004, REL-002, GOV-002, GOV-004]
must_haves:
  truths:
    - "Gemma generates reference-aligned outputs from `.sinf` on RTX 5090."
    - "The second model landed through the approved extension surfaces without executor changes."
    - "Qwen remains correct and supported after generic extensions."
  artifacts:
    - "Gemma correctness corpus and RTX 5090 acceptance report"
    - "Five-surface architecture conformance/diff report"
    - "Cross-model compatibility and regression matrix"
---

# S07-03 — Gemma Correctness and Extensibility Proof

## Objective

Prove both user-visible Gemma correctness and the architectural claim that a second model can be added without changing the minimal runtime executor.

## Tasks

1. **Define Gemma acceptance corpus/contracts**
   - Pin prompts/token IDs/reference outputs, short/long contexts, special tokens/templates, deterministic generation and selected intermediate/logit summaries.
   - Approve numerical contracts for the chosen storage/quantization policy.
   - Include fixtures for every semantic difference identified in S07-01.

2. **Run artifact-to-token acceptance**
   - Execute only the validated `.sinf` plus runtime options on qualified RTX 5090.
   - Compare trusted reference tokens/logits/state, repeated sessions, reset/reuse, boundary capacity and failure behavior.
   - Capture hot-path trace, peak memory and resource plan; label timing diagnostic only.

3. **Run cross-model regression/compatibility matrix**
   - Validate/read supported Qwen and Gemma artifact/schema versions.
   - Run representative Qwen S03 corpus and Gemma corpus on the same release candidate.
   - Test unsupported version/target/model-config cases fail with useful diagnostics.

4. **Audit the five extension surfaces**
   - Map every Gemma change to ModelFrontend, GraphPass, KernelProvider, DecodeStrategy, StoragePolicy or a canonical IR schema owned by those flows.
   - Produce source/dependency diff proving executor unchanged from the S06 baseline.
   - Search runtime/kernel selector for Gemma/Qwen family identifiers and fail on matches.
   - Document reusable generic semantics/providers and any stress discovered in the interfaces.

5. **Publish the architecture proof**
   - Summarize exact source/artifact/target identity, correctness, compatibility, changed components, unchanged executor, known limitations and future model guidance.
   - Record any proposed architecture changes as future decisions; do not retroactively hide deviations.

## Verification

- Full Gemma corpus passes twice in separate sessions.
- Representative Qwen corpus remains green.
- Executor source/dependency hashes match approved S06 baseline.
- Architecture conformance report accounts for every changed file and all S07 requirements.

## Completion Evidence

- Gemma acceptance and cross-model compatibility bundles.
- Extensibility proof with executor unchanged result.
- Explicit statement of supported Gemma scope and limitations.

## Conditional Architecture Gate

Present the five-surface diff as an L1 audit. If any proposed Gemma change would modify the executor or alter a core extension boundary, stop before implementation, elevate the audit to L2, create the canonical packet, update state, and emit `UNDERSTANDING STATUS`. If no trigger occurs, record the unchanged-executor evidence without adding understanding debt.
