# S03F-01 Summary — Flash-Next contract and capacity research

## Status

Blocked by missing exact source evidence. The repository has the approved Flash-Next design and
plan, but no exact `Qwen3.8-Flash-Next` model directory, config, safetensors index/shards, or pinned
reference implementation/revisions. Nearby Qwen3.8-DFlash2 and Qwen3-Coder-Next GGUF artifacts were
inspected and excluded as non-matches.

## Work completed

- Added a metadata-only, fail-closed validator requiring an explicit model/reference contract,
  provenance revisions, PLE metadata, QSA metadata, and consistent safetensors headers.
- Added deterministic tensor records, category classification, and an offline contiguous two-device
  residency projection utility. These utilities are tested only with synthetic fixtures and do not
  establish Flash-Next facts.
- Added canonical blocked evidence for source qualification, the exact-byte memory ledger, and
  residency options. Missing values are `null`; no parameter-count or quality substitute is used.
- Recorded D-019: residency feasibility is unknown and expert paging/staging/caching is forbidden
  until exact packed bytes and quality evidence exist.
- No Physical Plan, IR, compiler, specializer, runtime, CUDA, kernel, or memory-planner files were
  changed. S03F-02 remains blocked until S03 closes.

## Evidence and verification

- `PYTHONPATH=python python -m unittest -q tests/unit/test_flash_next.py` — 5 tests passed.
- Canonical serialization check for `artifacts/S03F/flash-next-source-evidence.json` — passed.
- `pytest tests/unit/test_flash_next.py -q` could not run because `pytest` is not installed in this
  environment.
- `python tools/validate.py --full` is the remaining repository-wide check; its result is recorded
  in the handoff if the local dependency environment permits it.

## Exact blockers

1. Supply the exact model artifact (`config.json`, safetensors index and shards, or an equivalent
   authenticated packed artifact) and its immutable source hash.
2. Supply the exact upstream/reference repositories and 40-character revisions, plus the pinned
   semantic metadata for PLE/N-gram and QSA.
3. Run reference quality evaluation before selecting any quantization/residency candidate.
