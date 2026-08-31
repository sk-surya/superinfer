# S03F-01 Summary — Flash-Next contract and capacity research

## Status

The official model and source identities are now pinned, but capacity/quality qualification remains
blocked because the complete checkpoint is not local and the official repository has no executable
reference checkout. Nearby Qwen3.8-DFlash2 and Qwen3-Coder-Next GGUF artifacts were inspected and
excluded as non-matches.

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
- No Physical Plan, IR, compiler, specializer, runtime, CUDA, kernel, or memory-planner files were
  changed. S03F-02 remains blocked until S03 closes.

## Evidence and verification

- `PYTHONPATH=python python -m unittest -v tests/unit/test_flash_next.py` — 9 tests passed after
  fail-closed hash/path/category/capacity fixes.
- Canonical serialization check for `artifacts/S03F/flash-next-source-evidence.json` — passed.
- `pytest tests/unit/test_flash_next.py -q` could not run because `pytest` is not installed in this
  environment.
- `SUPERINFER_VALIDATION_TMP=/srv/repos/superinfer/build/validation-tmp TMPDIR=/srv/repos/superinfer/build/superinfer-tmp CMAKE_BUILD_PARALLEL_LEVEL=2 python3 tools/validate.py --full` — all validation stages passed after the final research/tooling changes.

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
