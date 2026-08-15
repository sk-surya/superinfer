---
phase: "S03-qwen38-e2e"
plan: "S03-03"
type: "verification"
wave: 3
depends_on: [S03-02]
files_modified:
  - tests/corpora/qwen38/**
  - tests/{integration,gpu}/qwen38/**
  - tools/correctness_report/**
  - artifacts/S03/**
  - docs/models/qwen38-27b.md
autonomous: false
requirements_addressed: [MOD-001, MOD-004, DEC-001, KER-006, BCK-004, QUA-002, QUA-003]
must_haves:
  truths:
    - "Pinned prompts produce reference-aligned logits/tokens from `.sinf` on RTX 5090."
    - "Long/repeated decode preserves KV/state bounds and determinism."
    - "The Qwen correctness proof is reproducible and independent of performance tuning."
  artifacts:
    - "Versioned Qwen correctness corpus and numerical contract"
    - "RTX 5090 artifact-to-token acceptance report"
    - "Long-context, repeatability and hot-path trace evidence"
---

# S03-03 — Qwen End-to-End Correctness Acceptance

## Objective

Prove the narrow product path—pinned source to `.sinf` to correct RTX 5090 generation—before kernel optimization begins.

## Tasks

1. **Define the acceptance corpus and contracts**
   - Include short/plain, chat-template, Unicode/special-token, varied prompt length, stop behavior, deterministic greedy continuations, and contexts near declared KV boundaries.
   - Pin tokenized inputs, reference implementation/build, seeds/settings, expected token sequences and selected logits/intermediate summaries.
   - Approve operation/layer/model numerical thresholds based on dtype/quantization analysis; changes require review and evidence.

2. **Implement artifact-to-token test harness**
   - Start from only `.sinf` plus declared runtime options; do not access the source HF model during SuperInfer execution.
   - Capture load/validation, tokenization identity, prefill outputs, per-step tokens/state hashes, error/context and hot-path trace.
   - Run the trusted reference separately and compare under the pinned contract.

3. **Test deterministic and long-lived behavior**
   - Repeat prompts in fresh and reused sessions, reset state, run to maximum declared context and test clean rejection beyond capacity.
   - Cover multiple output lengths and early-stop/no-stop cases.
   - Detect state leakage across sessions/prompts and inconsistent tokenizer/template behavior.

4. **Qualify failure behavior**
   - Corrupt artifact/tensor/plan hashes; mismatch target fingerprint; request excessive context; inject controlled CUDA failure.
   - Assert safe rejection, bounded diagnostics, no partial output presented as valid, and explicit session recovery/reconstruction requirements.

5. **Produce reviewed acceptance report**
   - Summarize exact source/artifact/runtime/target identities, corpus coverage, errors, tokens, memory peak, trace assertions and known limitations.
   - Keep timing labeled diagnostic only; do not make comparative performance claims in S03.
   - Record all hardware-required test run IDs and raw evidence checksums.

## Verification

- Run the full corpus twice in separate target sessions and compare results/evidence schema.
- Independently validate `.sinf` before the run and hash the artifact/corpus/report inputs.
- Review every tolerance and any token divergence; unresolved deterministic token divergence blocks S04.
- Confirm all S03 requirements have an evidence link in the report.

## Completion Evidence

- Signed/checked acceptance summary and machine-readable report.
- Passing long-context/repeatability/failure matrix.
- Explicit statement: Qwen3.8-27B generates correctly from `.sinf` on the qualified RTX 5090 under the declared configuration.
