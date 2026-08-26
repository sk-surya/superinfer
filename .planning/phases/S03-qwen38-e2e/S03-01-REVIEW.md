---
phase: S03-qwen38-e2e
reviewed: 2026-08-26T05:12:42Z
depth: deep
files_reviewed: 10
files_reviewed_list:
  - docs/models/qwen38-27b.md
  - frontends/qwen38/frontend.hpp
  - frontends/qwen38/manifest.json
  - include/superinfer/ir/semantic/module.hpp
  - python/superinfer/convert/__init__.py
  - python/superinfer/convert/qwen38.py
  - tests/CMakeLists.txt
  - tests/unit/frontends/qwen38_frontend_test.cpp
  - tests/unit/ir/semantic_ir_test.cpp
  - tests/unit/test_qwen38.py
findings:
  critical: 9
  warning: 4
  info: 0
  total: 13
status: issues_found
---

# Phase S03: Code Review Report

**Reviewed:** 2026-08-26T05:12:42Z  
**Depth:** deep  
**Files Reviewed:** 10  
**Status:** issues_found

## Summary

The focused C++ tests and Python tests pass, but they only validate a synthetic topology and shallow metadata checks. The submitted validator does not establish the claimed pinned source, and the frontend emits a hard-coded IR skeleton that is not connected to the validated tensor/config inventory. The issues below block treating this as a correct or provenance-safe Qwen frontend.

## Critical Issues

### CR-01: Revision arguments are labels, not provenance verification

**Severity:** BLOCKER  
**File:** `python/superinfer/convert/qwen38.py:252-270`

**Issue:** `validate_source` accepts any two 40-character lowercase hex strings and arbitrary repository overrides; it never reads or compares `frontends/qwen38/manifest.json`, verifies a repository commit, or checks the local directory against the pinned input hashes. `Qwen38Inventory.manifest()` then echoes those caller-supplied identities (`:70-83`). A different directory can therefore be reported as the pinned Qwen source.

**Fix:** Make repository/revision constants non-overridable, verify every pinned metadata and shard hash against a checked-in manifest, and include the verified inventory digest in the identity passed to C++.

### CR-02: Tensor validation is not an exact pinned schema check

**Severity:** BLOCKER  
**File:** `python/superinfer/convert/qwen38.py:213-249`

**Issue:** The validator checks only index/header name equality, positive dimensions, and in-file offsets. It does not require the 2,402 pinned names, expected shapes/dtypes/byte sizes, tied aliases, or authoritative roles; `_tensor_role` is only a substring heuristic. The test helper creates and accepts a one-tensor source (`tests/unit/test_qwen38.py:32-55`), despite the checked-in manifest declaring 2,402 tensors.

**Fix:** Compare the complete normalized inventory to a generated, checked-in expected schema; validate allowed dtypes, exact shape/byte contracts, aliases, and role mapping before returning an inventory.

### CR-03: Generated integrity metadata omits all weight payload hashes

**Severity:** BLOCKER  
**File:** `python/superinfer/convert/qwen38.py:87-97,274-279`

**Issue:** `_REQUIRED_FILES` contains metadata only, so `file_sha256` never hashes `model-*.safetensors`. `tensor_inventory_sha256` hashes descriptors, not payload bytes. The shard hashes shown in `frontends/qwen38/manifest.json:35-41` are never generated or compared by this validator. A modified weight payload with unchanged headers passes the provenance gate.

**Fix:** Hash every indexed shard (or use a cryptographically authenticated artifact), compare those hashes to the pinned manifest, and make payload hashes part of the canonical source identity.

### CR-04: Shard names permit path traversal and absolute-path reads

**Severity:** BLOCKER  
**File:** `python/superinfer/convert/qwen38.py:217-223`

**Issue:** Any index value ending in `.safetensors` is accepted. `model_dir / shard` therefore follows `../outside.safetensors` and absolute paths, causing the validator to inspect files outside the selected source directory and preserving that path in `source_shard`.

**Fix:** Require a plain basename, reject absolute paths and `..`, resolve the candidate, and require it to remain under the resolved model directory before opening it.

### CR-05: Accepted layer metadata can disagree with emitted topology

**Severity:** BLOCKER  
**File:** `python/superinfer/convert/qwen38.py:144-150`; `frontends/qwen38/frontend.hpp:85-99`

**Issue:** Python validates only that there are 64 entries from two allowed names and that the interval is 4; it does not validate the exact schedule or linear-attention dimensions. The C++ frontend ignores `layer_types` and hard-codes full attention at `(layer % 4) == 3` plus hard-coded 48/16/128/128/4 attributes. The unit fixture uses an alternating 32/32 schedule (`tests/unit/test_qwen38.py:23-24`) and still passes validation. A source with a different valid-looking schedule or linear dimensions is accepted and compiled as another model.

**Fix:** Validate the exact pinned per-layer schedule and all linear-attention fields, then pass the normalized schedule/config through the frontend instead of reconstructing it from a modulo rule.

### CR-06: C++ frontend is disconnected from the source inventory

**Severity:** BLOCKER  
**File:** `frontends/qwen38/frontend.hpp:28-49,51-141`; `include/superinfer/compiler/model_frontend.hpp:11-13`

**Issue:** `SourceInventory` contains only an identity string, and `Frontend::validate` checks only that string. `emit` creates hard-coded activation placeholders and operations; it emits no source weight tensors, tensor-origin references, tokenizer/config facts, or normalized tensor mapping. Thus any caller can construct the matching string without having run the Python validator, and the resulting module has no information from which a real Qwen artifact can be materialized.

**Fix:** Define a versioned inventory contract carrying the validated config/tensor/tokenizer data, reject missing or mismatched records in `validate`, and emit semantic tensors with deterministic source origins. Keep storage layout decisions outside the frontend.

### CR-07: Token IDs are modeled as int8

**Severity:** BLOCKER  
**File:** `frontends/qwen38/frontend.hpp:44-45`; `include/superinfer/ir/semantic/module.hpp:33`

**Issue:** The vocabulary is 248,320 (`frontend.hpp:48`), but `token_ids` is declared `DType::int8`, which cannot represent ordinary token IDs above 127. This makes embedding lookup semantics incorrect before lowering.

**Fix:** Add an explicit integer dtype with sufficient range (normally int32/uint32), use it for token IDs, and add verifier and boundary tests for the maximum vocabulary ID.

### CR-08: `gated_delta_attention` has no state or mathematical contract

**Severity:** BLOCKER  
**File:** `frontends/qwen38/frontend.hpp:97-101`; `include/superinfer/ir/semantic/module.hpp:101-106,193-197`

**Issue:** The new operation is emitted with one hidden-state input and one hidden-state output, but the frontend adds no recurrent state edges, projections, convolution state, gates, or state tensor. The verifier checks only that three scalar attributes are nonzero. The decision for this operation explicitly requires preserving state-transition semantics, so this IR cannot represent or execute the linear-attention block correctly.

**Fix:** Specify the operation’s inputs/outputs, state lifetime, update equations, and numerical contract; emit the required state edges and add an independent reference implementation with boundary/differential tests before allowing this operation into a compilable frontend graph.

### CR-09: Corrupt or wrong tokenizer/template data can pass the gate

**Severity:** BLOCKER  
**File:** `python/superinfer/convert/qwen38.py:159-175,274-281`; `tests/unit/test_qwen38.py:34-44`

**Issue:** Only three tokenizer-config fields are checked. `tokenizer.json`, `vocab.json`, and `merges.txt` are not parsed, special-token IDs and normalization/pretokenization are not validated, and the chat template is accepted if nonempty. The tests deliberately use `{}` for tokenizer/vocabulary JSON and a blank merges file, so a source with incompatible tokenization can pass.

**Fix:** Validate the complete tokenizer schema and special-token mapping, canonicalize and hash tokenizer/template behavior, and add golden prompt-ID/template-output comparisons against the pinned reference.

## Warnings

### WR-01: Checked-in manifest is machine-specific and has an unaudited license claim

**Severity:** WARNING  
**File:** `frontends/qwen38/manifest.json:3,11-13`; `docs/models/qwen38-27b.md:8-11`

**Issue:** The manifest embeds `/srv/models/...`, making provenance nonportable and leaking a local filesystem layout. It records only `"apache-2.0"` and does not retain the upstream/derivative license texts or redistribution terms required by the plan. The local variant hash is not independently reproducible from checked-in inputs.

**Fix:** Remove absolute paths, use a canonical relative source label, record SPDX/license-file hashes and derivative terms, and generate/verify the variant identity from a reproducible manifest process.

### WR-02: Malformed-file errors are not consistently converted to stable diagnostics

**Severity:** WARNING  
**File:** `python/superinfer/convert/qwen38.py:100-106,221-227,280-281`

**Issue:** Invalid UTF-8 can escape `_read_json` as `UnicodeDecodeError`; the later `stat`/second `open` and `struct.unpack` are outside the initial I/O guard; and template decoding has no diagnostic wrapper. These cases violate the promised stable field-oriented error contract.

**Fix:** Wrap all metadata/header reads, decoding, stat, and unpack operations and translate `UnicodeError`, `OSError`, and `struct.error` into `Qwen38ValidationError` with file/field context.

### WR-03: Tests do not exercise the pinned manifest or negative schema surface

**Severity:** WARNING  
**File:** `tests/unit/test_qwen38.py:58-90`; `tests/unit/frontends/qwen38_frontend_test.cpp:9-25`

**Issue:** The Python tests check repeatability by validating the same synthetic directory twice and cover only one config and one index mismatch. The C++ test checks operation counts and a dump substring, not source-to-IR mapping, weights, state, shapes, tokenizer identity, or output behavior. Passing tests therefore provide no evidence for the pinned-source or full-frontend claims.

**Fix:** Add a checked-in manifest golden test, actual inventory/hash comparison, missing/extra/wrong-shaped/wrong-dtype/overlap/traversal fixtures, tokenizer golden tests, and independent semantic differential tests.

### WR-04: Frontend failures discard the builder’s actionable error

**Severity:** WARNING  
**File:** `frontends/qwen38/frontend.hpp:61-64,81-84,100-103,110-123,133-139`

**Issue:** Multiple failed `Result`/`Status` branches replace the original field/name error with a generic message. This makes future schema or emission failures difficult to diagnose and conflicts with the plan’s exact diagnostics requirement.

**Fix:** Preserve the returned status and attach the operation/tensor name, e.g. `return result.error().with_context("Qwen3.8 layer_... attention");`.

---

_Reviewed: 2026-08-26T05:12:42Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
