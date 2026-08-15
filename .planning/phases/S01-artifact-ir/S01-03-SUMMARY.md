---
phase: S01-artifact-ir
plan: S01-03
subsystem: artifact
tags: [sinf, checksum, converter, inspector, storage-policy]
requires:
  - phase: S01-artifact-ir
    provides: Semantic IR, Lowered IR, and Physical Plan contracts
provides:
  - Deterministic `.sinf` v1 writer/reader with required sections and integrity table
  - Defensive CPU validation, aligned host StoragePolicy, and cross-language byte layout
  - Synthetic converter, inspect/validate CLI, golden/negative/fuzz-smoke evidence
affects: [S02-sm120-baseline, S03-qwen38-e2e, S06-performance-proof]
actuals:
  tokens: 3800
  tasks: 7
  commits: 3
tech-stack:
  added: [fixed little-endian container, FNV-1a 64 checksum]
  patterns: [atomic partial-file replace, read-only validated views, canonical JSON]
key-files:
  created:
    - include/superinfer/artifact/sinf.hpp
    - include/superinfer/artifact/host_storage_policy.hpp
    - python/superinfer/artifact.py
    - python/superinfer/convert.py
    - python/superinfer/inspect.py
    - schemas/sinf/v1.json
    - docs/sinf-format.md
    - tests/integration/artifact/artifact_test.cpp
    - tests/fuzz/test_artifact_fuzz_smoke.py
  modified: [python/superinfer/cli.py, .planning/DECISIONS.md]
key-decisions:
  - "D-013: use a fixed little-endian sectioned container with canonical JSON/text, 8-byte alignment, per-section FNV-1a checksums, and integrity records."
  - "FNV-1a provides corruption detection only; authenticity/signature is a reserved future concern."
requirements-completed: [FMT-001, FMT-002, FMT-003, FMT-004, FMT-005, FMT-006, GOV-002, GOV-003, GOV-004]
coverage:
  - id: D1
    description: "Deterministic sectioned .sinf writer/reader and integrity validation"
    requirement: FMT-001
    verification:
      - kind: integration
        ref: "tests/integration/artifact/artifact_test.cpp"
        status: pass
      - kind: integration
        ref: "tests/unit/test_artifact.py#test_converter_bytes_and_inspection_are_deterministic"
        status: pass
    human_judgment: false
  - id: D2
    description: "Truncation, corruption, version, required-section, and bounded mutation rejection"
    requirement: FMT-002
    verification:
      - kind: integration
        ref: "tests/integration/artifact/artifact_test.cpp; tests/fuzz/test_artifact_fuzz_smoke.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "CPU converter, inspect/validate CLI, host storage policy, and install-independent tooling"
    requirement: FMT-006
    verification:
      - kind: integration
        ref: "python3 tools/validate.py --full"
        status: pass
    human_judgment: false
duration: 45min
completed: 2026-08-15
status: complete
---

# Phase S01 Plan S01-03 Summary

**Deterministic, checksummed `.sinf` deployment artifacts with CPU inspection and normalized conversion**

## Accomplishments

- Implemented `.sinf` v1 fixed header/directory, required manifest/tensor/plan/payload/integrity sections, alignment, checksums, and atomic writes.
- Implemented defensive C++ and Python readers that fail closed on truncation, corruption, version skew, overlap, unknown required sections, and integrity mismatch.
- Added host StoragePolicy baseline, converter/inspect/validate CLI commands, schema/documentation, golden/negative/fuzz-smoke tests, and C++/Python interoperability.

## Task Commits

1. **S01-03 artifact format and tooling** - `c5d23d8` (feat)
2. **S01-03 CLI commands** - `b30777e` (feat)
3. **S01-03 CLI import normalization** - `8d4ed65` (style)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Host LeakSanitizer tracing incompatibility expanded to S01 tests**
- **Found during:** full sanitizer verification
- **Issue:** New C++ tests aborted before application code under ptrace.
- **Fix:** Extended the existing explicit `ASAN_OPTIONS=detect_leaks=0` CTest environment to all C++ sanitizer tests.
- **Verification:** `ctest --preset cpu-sanitize` and `python3 tools/validate.py --full` pass.
- **Committed in:** `c5d23d8` implementation state (test configuration staged with S01 artifact work).

**Total deviations:** 1 auto-fixed. **Impact:** AddressSanitizer/UBSan remain active; only the host-incompatible leak detector is disabled.

## Verification

`python3 tools/validate.py --full` passes Python checks/tests, CPU build/CTest, install consumer,
sanitizer CTest, and wheel build. A Python-produced artifact is also accepted by the C++ reader
fixture (`build/cpu-dev/tests/superinfer_artifact_test /tmp/superinfer-cross.sinf`).

## Next Phase Readiness

S01 is complete. Gate A is reached and its packet is presented separately; S02 implementation must
wait until the user passes the semantic/compiler boundary ownership exercise.

---
*Phase: S01-artifact-ir*
*Plan: S01-03*

