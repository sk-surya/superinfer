# S03F-01 Summary — Flash-Next contract and capacity research

## Status

The official model and source identities are now pinned, but capacity/quality qualification remains
blocked because the complete checkpoint is not local and the official repository has no executable
reference checkout. Nearby Qwen3.8-DFlash2 and Qwen3-Coder-Next GGUF artifacts were inspected and
excluded as non-matches. A complete local AtomicChat GGUF conversion is now separately inventoryable,
but it is not substituted for the official source or quality oracle.

## Work completed

- Added a metadata-only, fail-closed validator requiring an explicit model/reference contract,
  provenance revisions, PLE metadata, QSA metadata, and consistent safetensors headers.
- Added deterministic tensor records, category classification, and an offline contiguous two-device
  residency projection utility. These utilities are tested only with synthetic fixtures and do not
  establish Flash-Next facts.
- Pinned the official Hugging Face model revision, source revision, config/index hashes, 131 shards,
  1,658 tensor entries, and reported BF16 payload size. Added canonical blocked evidence for the
  incomplete local source, exact-byte memory ledger, and residency options. Missing values remain
  `null`; no parameter-count or quality substitute is used.
- Recorded D-019: residency feasibility is unknown and expert paging/staging/caching is forbidden
  until exact packed bytes and quality evidence exist.
- Added a deterministic formula-only state/workspace estimator parameterized by context, batch,
  activation, KV, and recurrent-state byte widths. It reports KV state, recurrent state,
  convolution state, and a reusable decode-workspace lower bound without implying complete model
  residency.
- No Physical Plan, IR, compiler, specializer, runtime, CUDA, kernel, or memory-planner files were
  changed. S03F-02 remains blocked until S03 closes.
- Added a metadata-only GGUF split reader and reproducible inventory command. The 33-shard AtomicChat
  candidate passes its supplied SHA256 manifest, contains 1,224 tensors and 94,514,368,000 packed tensor
  bytes, and leaves 56,079,162,880 computational bytes when the 38,435,205,120-byte PLE table remains
  host-mmap resident.
- A deterministic capacity-only projection with 4 GiB headroom per 32-GiB card places layers 0–25 on
  device 0 and 26–47 on device 1. This is not an official residency or quality decision.

## Evidence and verification

- `PYTHONPATH=python python -m unittest -v tests/unit/test_flash_next.py` — 9 tests passed after
  fail-closed hash/path/category/capacity fixes.
- Canonical serialization check for `artifacts/S03F/flash-next-source-evidence.json` — passed.
- `pytest tests/unit/test_flash_next.py -q` could not run because `pytest` is not installed in this
  environment.
- `SUPERINFER_VALIDATION_TMP=/srv/repos/superinfer/build/validation-tmp TMPDIR=/srv/repos/superinfer/build/superinfer-tmp CMAKE_BUILD_PARALLEL_LEVEL=2 python3 tools/validate.py --full` — all validation stages passed after the final research/tooling changes.
- `TMPDIR=/srv/repos/superinfer/build/tmp uv run --extra dev pytest -q tests/unit/test_flash_next.py tests/unit` — 56 tests and 9 subtests passed after the runtime-state formula addition.
- `TMPDIR=/srv/repos/superinfer/build/tmp python tools/validate.py --full` — all 11 repository validation stages passed.

The formula evidence is `artifacts/S03F/flash-next-runtime-state-formula.json`. For the local
candidate's 48-layer config, it estimates 216,883,200 bytes of text state/workspace at context
4,096 and 6,558,670,848 bytes at context 262,144 under BF16 KV/convolution, FP32 recurrent state,
and 2-byte activation assumptions. These values are not packed model totals and do not change the
capacity/quality blocker.

## Exact blockers

1. Supply the complete exact model artifact (`config.json`, safetensors index and shards, or an equivalent
   authenticated packed artifact) and its immutable source hash.
2. Supply an executable reference checkout for the pinned source, plus the pinned semantic metadata
   for PLE/N-gram and QSA where the technical report is insufficient.
3. Run reference quality evaluation before selecting any quantization/residency candidate.

## Partial NVFP4 evidence update

The local RadixArk NVFP4 candidate is more exact than a generic "near match" label. Header-only
inspection finds 91 routed-expert safetensors shards containing 139,776 tensors and exactly
`32,204,669,952` tensor-payload bytes for 23 present layers. Twenty-two layers are complete; the
missing layer-21 `experts-0256-0383` shard means one range is absent. All 22 matching complete
layers have exactly `1,415,589,888` tensor-payload bytes, yielding a bounded projection of
`67,948,314,624` routed-expert tensor bytes for all 48 layers. This remains a projection because
layers 24–47 and all non-routed checkpoint families are absent locally.

The candidate's checked-in audit reports 1,562 unchanged non-routed BF16 tensors comprising
`118,408,052,728` bytes, including 31 MTP tensors. Its README records GSM8K `0.9727065959059894`
and AIME26 `pass@1 0.9875` / `majority@8 1.0`. These are upstream serving evaluations, not
SuperInfer reference-equivalence evidence and not authorization for residency selection.
The detailed, hash-bound observation is
`artifacts/S03F/flash-next-partial-nvfp4-ledger.json`. It does not change D-019 or unblock S03F-02.

## Complete local GGUF candidate update

The complete local conversion is recorded in `artifacts/S03F/flash-next-gguf-candidate-inventory.json`.
It is authenticated by `/srv/models/qwen3.8-flash-next/service/evidence/SHA256SUMS` and pinned to
llama.cpp `6c84c7d5d8833c6e0df69628f75a0f599797934e`. The inventory is exact for this packed
conversion, but no official-checkpoint equivalence or SuperInfer quality result is claimed. The
candidate therefore improves capacity evidence without changing D-019 or opening S03F-02.
