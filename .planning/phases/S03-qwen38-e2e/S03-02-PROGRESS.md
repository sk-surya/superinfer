---
phase: S03-qwen38-e2e
plan: S03-02
status: in_progress
updated: 2026-08-26
---

# S03-02 Progress Checkpoint

## Delivered

- Generic Semantic-to-Lowered lowering preserves static shapes, target capability requirements,
  semantic tensor roles, and explicit operands without selecting model-specific kernels.
- Qwen source validation authenticates the pinned repositories/revisions, exact layer schedule and
  linear-attention dimensions, tokenizer structure, indexed shard containment, metadata hashes, and
  streamed payload hashes.
- Qwen Semantic IR preserves full-attention KV state and Gated DeltaNet recurrent/convolution state
  through explicit state tensors and 128 state edges.
- KV cache and greedy decode contracts are bounded, allocation-free after construction, deterministic,
  and covered by CPU tests for append, commit, rollback, reset, argmax ties, stop, and bounds.
- An independent CPU Gated DeltaNet oracle covers Q/K normalization, grouped key/value heads, decay,
  beta delta updates, recurrent state, and split-step continuation; no provider advertises this
  operation yet.
- Deterministic metadata `.sinf` recipe emits source mapping, operation coverage, and a 32-GiB memory
  ledger. It explicitly records that payload materialization and physical provider coverage remain
  pending.
- Deterministic payload-bearing `.sinf` conversion now streams all three pinned safetensors shards
  in bounded chunks, preserves relative tensor offsets, and authenticates every source shard before
  writing. The artifact-size guard is 32 GiB so the 18.77-GB Qwen payload is representable.

## Verification

- `python3 tools/validate.py --full` passes all Python, CMake, CPU CTest, sanitizer, install-consumer,
  and wheel stages.
- Strict validation of the local pinned source passes 2,402 tensors, the canonical inventory digest,
  all required metadata hashes, and all three safetensors shard hashes.
- Real-source metadata-only rehearsal produced a 1,017,232-byte `.sinf` with SHA-256
  `9f80231a9dc59c214ddf4aafd52d27dfbc10def2ce7dcceaaad9e44c2cfa414e`; its ledger requires
  27,778,892,088 bytes and leaves 6,580,846,280 bytes against the declared 32-GiB budget. This is
  provenance evidence, not a payload-bearing execution artifact.
- Real pinned payload conversion completed in 13m12.57s using 791.80s CPU time and produced the
  current-recipe `build/evidence/qwen38-payload-v1-second.sinf` at 18,766,682,312 bytes with
  SHA-256 `2c228c1f0a4083ac82a8b89a928cc1f620152661cc721ad06c27bab410fc2b69`. A third clean
  conversion is byte-identical. Header inspection confirms five aligned sections, an
  18,765,513,016-byte payload section, 2,402 tensor records, and the pinned derivative revision.
  This is a payload/provenance artifact, not yet an executable plan.
- Independent streaming inspection validates the full payload checksum and integrity table with
  approximately 18 MiB resident memory. `Specializer` now receives a `KernelProvider`, selects a
  deterministic candidate through that extension surface, and propagates candidate workspace into
  the Physical Plan; it no longer owns a kernel-ID table.
- The explicit [operation coverage matrix](S03-02-COVERAGE.md) records which baseline commands are
  executable and which Qwen operations remain reference-only or unavailable.
- Independent CPU primitive contracts now cover embedding, linear/LM projection, gated FFN, and
  grouped attention with shape, finite-input, and negative-path tests. Full graph weight binding
  and CUDA-provider differential execution remain open.

## Remaining S03-02 work

- Connect the validated tensor inventory to semantic weight records consumed by the compiler rather
  than only the serialized tensor table.
- Bind the independent primitive contracts into a full graph reference harness and add the
  weight-connected CUDA providers before advertising Qwen capabilities.
- Implement or explicitly stage target-provider capability coverage and compile a non-placeholder
  Physical Plan.
- Complete conversion and runtime differential evidence before S03-03 acceptance.
