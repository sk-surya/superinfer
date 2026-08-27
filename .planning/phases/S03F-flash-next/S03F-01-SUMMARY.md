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

- `PYTHONPATH=python python -m unittest -v tests/unit/test_flash_next.py` — 7 tests passed.
- Canonical serialization check for `artifacts/S03F/flash-next-source-evidence.json` — passed.
- `pytest tests/unit/test_flash_next.py -q` could not run because `pytest` is not installed in this
  environment.
- `python tools/validate.py --full` — passed before this research update; rerun at integration.

## Exact blockers

1. Supply the complete exact model artifact (`config.json`, safetensors index and shards, or an equivalent
   authenticated packed artifact) and its immutable source hash.
2. Supply an executable reference checkout for the pinned source, plus the pinned semantic metadata
   for PLE/N-gram and QSA where the technical report is insufficient.
3. Run reference quality evaluation before selecting any quantization/residency candidate.
