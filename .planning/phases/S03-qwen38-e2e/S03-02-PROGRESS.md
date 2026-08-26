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
- Qwen conversion now validates and records the ModelOpt NVFP4 contract, group size 16, FP8 KV
  policy, excluded module patterns, and observed BF16/F32/FP8/U8 tensor storage dtypes.
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
- The compiler-side `SourceInventory` now carries validated tensor records with semantic role, source
  dtype/shape, and artifact-payload ranges. Qwen frontend emission consumes those records to create
  deterministic semantic weight tensors and binds the embedding, LM head, and available layer-norm
  weights without selecting layouts, CUDA kernels, or physical allocations.
- Qwen layer operations now require and bind the pinned gated-FFN gate/up/down records plus the
  linear-attention or grouped full-attention projection/normalization parameter records. Missing
  names fail at frontend emission with operation-specific context.
- ModelOpt scale/input-scale records are classified as storage metadata and are excluded from
  semantic weight creation; the inventory pin was updated to the resulting canonical digest.

## Verification

- `python3 tools/validate.py --full` passes all Python, CMake, CPU CTest, sanitizer, install-consumer,
  and wheel stages.
- Strict validation of the local pinned source passes 2,402 tensors, the canonical inventory digest,
  all required metadata hashes, and all three safetensors shard hashes.
- Real-source metadata-only rehearsal produced a 1,017,232-byte `.sinf` with SHA-256
  `9f80231a9dc59c214ddf4aafd52d27dfbc10def2ce7dcceaaad9e44c2cfa414e`; its ledger requires
  27,778,892,088 bytes and leaves 6,580,846,280 bytes against the declared 32-GiB budget. This is
  provenance evidence, not a payload-bearing execution artifact.
- Final pinned payload conversions completed under the quantization-aware recipe. The regenerated
  `build/evidence/qwen38-payload-v1-final-a.sinf` and `...-final-b.sinf` are byte-identical at
  18,766,674,736 bytes; both SHA-256 values are
  `e25022c8592875449968b9d0b1f56e6800971e0ba04d8a43eec980fe60dc65d5`. Header inspection confirms
  five aligned sections, an 18,765,513,016-byte payload section, 2,402 tensor records, the pinned
  derivative revision, the updated inventory digest `7342659a53eecbb04c47b5de89d957ca47cb021970cb252575b8b9161d0a84fc`,
  and the NVFP4/FP8 quantization contract. This is a payload/provenance
  artifact, not yet an executable plan.
- Independent streaming inspection validates the full payload checksum and integrity table with
  approximately 18 MiB resident memory. `Specializer` now receives a `KernelProvider`, selects a
  deterministic candidate through that extension surface, and propagates candidate workspace into
  the Physical Plan; it no longer owns a kernel-ID table.
- The explicit [operation coverage matrix](S03-02-COVERAGE.md) records which baseline commands are
  executable and which Qwen operations remain reference-only or unavailable.
- Independent CPU primitive contracts now cover embedding, linear/LM projection, gated FFN, and
  grouped attention with shape, finite-input, and negative-path tests. Full graph weight binding
  and CUDA-provider differential execution remain open.
- The CPU reference graph now composes weight-connected embedding and LM-head operations through
  those independent primitives, with strict integer-token, shape, finite-input, and error-context
  checks. This remains an oracle only; no provider capability is advertised.
- The same graph oracle now composes gated dense FFN projections with explicit gate/up/down weight
  shapes and a checked SiLU-gating numerical contract; NVFP4 decoding and CUDA execution remain
  intentionally unavailable.
- A generic FP32 embedding baseline is now registered through `KernelProvider` ID 7, resolved by
  the SM120 CUDA executor, and checked on RTX 5090. It is not promoted into the Qwen artifact
  matrix because the pinned embedding is BF16 and needs a matching materialization/differential
  contract.
- An independent CPU NVFP4 dequantization oracle now pins the ModelOpt layout: E2M1 low/high
  nibble order, FP8 E4M3FN per-16-element scales, positive `weight_scale_2`, and fail-closed
  handling for invalid shapes, scale counts, NaN encodings, and non-positive tensor scales. This
  is reference evidence only; no NVFP4 provider is advertised until it is differentially checked
  against source tensors and a target implementation.
- The generic embedding capability now carries a compile-time storage-dtype fact in `KernelQuery`:
  FP32 gathers remain kernel ID 7 and BF16-to-FP32 gathers use kernel ID 8. The BF16 path has
  explicit buffer validation and an RTX 5090 test using known IEEE BF16 encodings; Qwen execution
  is still blocked later by attention/DeltaNet and quantized projection coverage.
- The validated `.sinf` reader now exposes a borrowed, bounds-checked payload-range view. It keeps
  ownership in `ArtifactView`, rejects offset/length overflow and out-of-section access, and is
  covered by the binary artifact test; tensor-table parsing and artifact-to-token binding remain
  intentionally separate work.
- Python conversion tooling now reads one validated tensor payload by name through the signed
  tensor-table offsets without materializing the full artifact payload. It rejects metadata-only
  artifacts, missing names, malformed ranges, and truncation, and is covered by deterministic
  payload-artifact tests.
- A generic `nvfp4_dequantize` baseline command is now provider ID 9. It uses an explicit physical
  command scalar for `weight_scale_2`, decodes packed E2M1 plus FP8 E4M3 block scales into FP32,
  and passes an RTX 5090 differential fixture against the CPU oracle's known vector. It remains a
  materialization primitive, not yet a Qwen linear/FFN provider; source-scale validation and
  artifact binding still precede promotion.
- CPU CTest passes 22/22, CUDA CTest passes 24/24 including the RTX 5090 ownership and plan-executor
  tests, and the complete `tools/validate.py --full` gate passes Python, build, install-consumer,
  sanitizer, and wheel stages.

## Remaining S03-02 work

- Complete the remaining per-operation source-weight bindings and bind the independent primitive
  contracts into a full graph reference harness; then add the
  weight-connected CUDA providers before advertising Qwen capabilities.
- Implement or explicitly stage target-provider capability coverage and compile a non-placeholder
  Physical Plan.
- Complete conversion and runtime differential evidence before S03-03 acceptance.
