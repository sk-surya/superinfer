---
phase: S03-qwen38-e2e
plan: S03-01
status: complete
completed: 2026-08-26
commits: [f884b5d, 85c2143, 039f7f9, 776e07e, 7bf62e7]
---

# S03-01 — Pin and Implement the Qwen Frontend

## Delivered

- Pinned `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and the
  `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` derivative at
  `0cc27958cefbbe231782ec8511de8c4eb5233348`.
- Identified the local `LMHead4` export with a deterministic metadata/tensor-hash identity and
  recorded Apache-2.0 provenance in `frontends/qwen38/manifest.json`.
- Added a dependency-free validator that checks config, tokenizer identity, tensor index/header
  bijection, shapes, offsets, revisions, hashes, and normalized tensor roles before payload reads.
- Added a canonical Qwen frontend that emits 64 language layers: 48 `gated_delta_attention` and
  16 grouped-query full-attention layers, with embedding, RMSNorm, residual, gated FFN, and LM-head
  semantics. No target layouts, CUDA calls, or kernel IDs enter the emitted graph.
- Added the generic `gated_delta_attention` semantic operation and recorded D-015; physical lowering
  remains a later generic pass/provider task.

## Evidence

- Python suite: 16 tests passed, including malformed config and index/header fixtures.
- Full local source inventory: 2,402 tensors; repeated validator runs produced identical manifests.
- CPU CTest: 18/18 passed, including `superinfer.qwen38.frontend`.
- CUDA CTest: 20/20 passed after the semantic IR extension.
- `python3 tools/validate.py --full`: all CPU, install-consumer, sanitizer, and wheel stages passed.

## Remaining work

S03-02 must add generic lowering/provider coverage for the emitted operations, explicit KV/storage
planning, deterministic `.sinf` conversion, and a trusted differential path. The current frontend is
semantic topology evidence; it is not yet a claim of full Qwen generation correctness.
