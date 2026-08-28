---
phase: S03-03-and-S03F-01
reviewed: 2026-08-28T00:02:31Z
depth: deep
files_reviewed: 17
files_reviewed_list:
  - .planning/STATE.md
  - .planning/phases/S03-qwen38-e2e/S03-03-PROGRESS.md
  - .planning/phases/S03F-flash-next/S03F-01-SUMMARY.md
  - artifacts/S03/qwen38-e2e-corpus-acceptance-failure.json
  - artifacts/S03/qwen38-e2e-kv-boundary.json
  - artifacts/S03/qwen38-tokenizer-contract.json
  - artifacts/S03F/flash-next-memory-ledger.json
  - artifacts/S03F/flash-next-residency-options.json
  - artifacts/S03F/flash-next-source-evidence.json
  - docs/models/flash-next.md
  - python/superinfer/convert/flash_next.py
  - tests/gpu/sm120/qwen38_e2e_artifact_test.cu
  - tests/unit/test_flash_next.py
  - tests/unit/test_qwen38_corpus_reference.py
  - tools/qwen38_corpus_reference.py
  - tools/qwen38_s03_batched_acceptance.py
  - tools/qwen38_tokenizer_check.py
findings:
  critical: 6
  warning: 6
  info: 0
  total: 12
status: issues_found
---

# S03-03 / S03F-01: Code Review Report

**Reviewed:** 2026-08-28T00:02:31Z  
**Depth:** deep  
**Files Reviewed:** 17  
**Status:** issues_found

## Summary

The submitted evidence does not falsely close either gate: the S03 corpus artifact is `fail`, S03 remains
`In progress`, and S03F-01 remains capacity/quality blocked. The official model and QwenLM repository
identities are consistently recorded, the local Flash-Next candidate is correctly described as partial, and
no runtime/Physical Plan/kernel changes were found in the S03F-01 commit.

The implementation still has blockers in the acceptance harness and Flash-Next capacity validator. In
particular, numerical metrics are recorded but never gate acceptance, reference sidecars are trusted without
integrity checks, and the planner can report a feasible two-device placement that exceeds a device budget.
There are also path-safety, unknown-category, deterministic-evidence, test-wiring, and planning-state defects.

## Critical Issues

### CR-01 (BLOCKER): Logit metrics are diagnostic only and cannot fail acceptance

**File:** `tools/qwen38_s03_batched_acceptance.py:91-99,113-115`

**Issue:** `metrics()` is computed for every row, but its `max_abs`, `mean_abs`, and `rmse` values are never
compared with a pinned numerical contract. The report status depends only on the parsed stdout token sequence
and capture hash. A candidate can therefore produce materially wrong logits while preserving greedy tokens and
still receive `token_match=true` and, if repeatable, an overall `pass`. This contradicts S03-03's requirement
to compare under reviewed tolerances; the committed failure artifact already demonstrates that large logit
drift is being recorded without a pass/fail threshold.

**Fix:** Add versioned per-operation/row tolerances and finite-value checks, make each row's numerical result
part of `token_match`/acceptance status, and serialize the thresholds and their pass/fail result in the report.

### CR-02 (BLOCKER): Reference and corpus integrity metadata is trusted rather than verified

**File:** `tools/qwen38_s03_batched_acceptance.py:54-61,96-114`

**Issue:** The harness copies `case["token_ids_sha256"]`, `reference_diagnostics["output_sha256"]`, and
`reference_diagnostics["greedy_sequence"]` into the result without recomputing or cross-checking them. It does
not verify that the sidecar's `tokens`/`steps` match the corpus, that the sidecar hash matches the reference
capture bytes, or that the sidecar greedy sequence is the argmax of those bytes. A stale or mismatched reference
capture/sidecar can therefore become the oracle for a seemingly valid acceptance run.

**Fix:** Validate the corpus token hash from the actual list, validate reference sidecar schema and token/step
counts, recompute the reference capture hash and greedy sequence from the capture, and fail closed on any
mismatch before launching the target executable.

### CR-03 (BLOCKER): Official hashes and checkpoint completeness are not enforced by the validator

**File:** `python/superinfer/convert/flash_next.py:17-22,124-184`

**Issue:** `official_contract()` exposes the pinned config/index hashes and the evidence claims 131 shards and
1,658 tensor entries, but `validate_source()` never compares either file hash and has no expected shard or
entry-count fields. It only trusts provenance strings and a few config values. Thus a partial or altered index
with the same self-reported revisions can pass all implemented checks if its mapped headers are locally valid,
despite `docs/models/flash-next.md:27-29` claiming that the validator fails closed on a partial source.

**Fix:** Bind the official contract to expected config/index hashes, expected shard names/count, and expected
tensor-entry count (or an authenticated manifest), and reject any missing, extra, or altered source material
before constructing an inventory. Keep the generic-contract path separate from the official fail-closed path.

### CR-04 (BLOCKER): Safetensors shard paths permit traversal outside the model directory

**File:** `python/superinfer/convert/flash_next.py:166-183`

**Issue:** `shard_value` comes from the index and is directly appended to `model_dir` for both header parsing and
whole-file hashing. Absolute paths and `../` components can escape the supplied model directory, causing the
validator to read and hash arbitrary files selected by a crafted index. This is an information-disclosure and
untrusted-input boundary violation.

**Fix:** Require a non-absolute relative shard name, resolve it against the model root, reject any resolved path
outside that root (including symlink escapes), and use the validated path for both reads.

### CR-05 (BLOCKER): Unknown tensors are silently counted as non-expert text

**File:** `python/superinfer/convert/flash_next.py:187-212`

**Issue:** The function documentation says unknown tensor names fail closed, but the final branch assigns every
unrecognized name to `non_expert_text`. A new or misspelled vision, MTP, router, expert, or storage tensor can
therefore be included in the wrong byte bucket and produce an invalid capacity result while still reconciling
the total byte sum.

**Fix:** Use an explicit, tested naming/schema mapping and raise `FlashNextValidationError` for any name not
recognized by that mapping. Add fixtures for unknown, shared-expert, router/indexer, vision, and MTP names.

### CR-06 (BLOCKER): Residency feasibility ignores per-device layer capacity

**File:** `python/superinfer/convert/flash_next.py:226-244`

**Issue:** The planner checks only `total <= 2 * capacity` and splits layers without checking the first layer
against `capacity` when `running == 0`, or checking the resulting partition totals. For example, with capacity
100 and layers `(150, 50)`, it returns two partitions and `full_expert_residency_feasible: true`, even though
device 0 must hold 150 bytes. The output also omits per-device/category projections required by S03F-01.

**Fix:** Reject any individual layer larger than capacity, accumulate and validate every partition's actual
layer bytes (plus explicitly accounted state/workspace/headroom), and emit per-device totals before setting the
feasibility bit.

## Warnings

### WR-01 (WARNING): Evidence metadata embeds machine-specific paths

**File:** `tools/qwen38_corpus_reference.py:148-170`; `tools/qwen38_tokenizer_check.py:78-85`; `tools/qwen38_s03_batched_acceptance.py:137-138`

**Issue:** Reports serialize raw model, artifact, corpus, output, and reference paths. The committed tokenizer
artifact consequently contains `/srv/models/...`, and the reference report mixes an absolute model path with
relative output paths. Equivalent reruns from another checkout or host produce different metadata even when
bytes and results are identical, undermining the deterministic evidence contract.

**Fix:** Store stable logical names or normalized repository-relative paths in canonical reports, and retain
machine-local paths only in a separate environment section. Hash the actual inputs and include those hashes in
the canonical metadata.

### WR-02 (WARNING): Stale continuation environment variables leak between cases

**File:** `tools/qwen38_s03_batched_acceptance.py:68-79`

**Issue:** Each child environment starts from `os.environ`, and the one-token branch never removes
`SUPERINFER_QWEN38_CONTINUATION`, `SUPERINFER_QWEN38_CONTINUATION_TOKENS`, or
`SUPERINFER_QWEN38_CONTINUATION_LOGITS_F32`. If those variables are present in the parent environment, a case
that should execute only its initial token can execute an unintended continuation; the C++ test enables this
mode by variable presence alone.

**Fix:** Remove all harness-owned `SUPERINFER_QWEN38_*` variables before setting the case-specific environment,
or construct a clean allowlisted environment for every subprocess.

### WR-03 (WARNING): Tokenizer failures do not produce bounded failure evidence

**File:** `tools/qwen38_tokenizer_check.py:54-66,87-90`

**Issue:** Missing expected tokenizer files, malformed corpus fields, or tokenizer exceptions escape before the
report is written. The command returns an uncaught traceback rather than a structured `fail` artifact, so a
failed verification cannot be audited with the same metadata contract as a token mismatch.

**Fix:** Validate the corpus schema and expected file names, catch input/tokenizer/file errors, write a bounded
failure report containing the diagnostic and input hashes, and return nonzero after the report is durable.

### WR-04 (WARNING): Project state is stale relative to the submitted evidence

**File:** `.planning/STATE.md:34-35,44-46`

**Issue:** The progress table still says S03-03 full token execution is next and says Flash-Next
model-contract evidence is pending, while the same commit records S03 execution plus pinned S03F official
identity. It also says Unicode model execution remains open although the S03 progress and corpus artifact mark
the Unicode case as passing. These contradictions make downstream phase routing and evidence interpretation
ambiguous, even though the gates are correctly not closed.

**Fix:** Update the progress table/current-focus text to say S03 acceptance is still failing on the long cases,
S03F identity research is complete, and only S03F capacity/quality evidence remains blocked.

### WR-05 (WARNING): S03F summary claims completion without post-change full-validation evidence

**File:** `.planning/phases/S03F-flash-next/S03F-01-SUMMARY.md:26-32`

**Issue:** The summary records the focused `unittest` run, says `pytest` was unavailable, and says
`python tools/validate.py --full` passed before this research update. The plan requires the focused test and
full validation after the implementation changes. The current environment's full validation passes now, but
that result is not the evidence recorded by the submitted summary.

**Fix:** Re-run and record the exact required commands after the final commit, or explicitly mark the summary's
post-change verification as pending rather than describing the research gate as complete.

### WR-06 (WARNING): Legal KV boundary checks are opt-in and absent from the normal GPU test

**File:** `tests/gpu/sm120/qwen38_e2e_artifact_test.cu:216-224,273-290`; `tests/CMakeLists.txt:184-186`

**Issue:** The 4096 rejection and 4095 legal-boundary execution are guarded by environment-variable presence,
but the CTest registration does not set either variable. A normal GPU test run therefore exercises neither
boundary assertion; the committed JSON is manual evidence rather than a regression gate.

**Fix:** Register dedicated CTest invocations with the two environment settings (or split them into explicit test
executables) and assert the expected `out_of_range` status, no launch before rejection, stable allocation count,
and the documented zero-initialized-cache limitation.

---

_Reviewed: 2026-08-28T00:02:31Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
