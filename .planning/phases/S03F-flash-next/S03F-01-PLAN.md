# S03F-01 Flash-Next Contract and Capacity Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin the Flash-Next text architecture and prove an exact dual-5090 residency/quantization budget before runtime changes.

**Architecture:** This is a research-only plan. It inventories the pinned reference/model artifacts, derives a semantic feature contract and exact byte ledger, evaluates candidate `.sinf` quantization/residency recipes, and records a decision boundary for expert residency. It must not modify runtime, Physical Plan, kernel, or memory-planner code.

**Tech Stack:** Python, safetensors metadata, existing SuperInfer artifact/provenance utilities, JSON/Markdown evidence.

**Spec:** `.planning/FLASH-NEXT-DESIGN.md`

## Global Constraints

- May run in parallel with S03.
- No runtime changes.
- Pin exact upstream/reference revisions before deriving behavior.
- Report packed bytes from artifacts, not parameter-count estimates, when available.
- Keep vision and MTP in the ledger but exclude them from initial execution target.

---

### Task 1: Pin and validate the source contract

**Files:**
- Create: `python/superinfer/convert/flash_next.py`
- Create: `tests/unit/test_flash_next.py`
- Create: `docs/models/flash-next.md`

**Interfaces:**
- Produces: `FlashNextInventory` with pinned repositories/revisions, config, tensor records, feature flags, and hashes.
- Consumes: existing artifact/provenance helpers only.

- [ ] Write failing tests that reject wrong model type, layer count, expert count/top-k, missing PLE metadata, unexpected QSA configuration, and changed revisions.
- [ ] Run `pytest tests/unit/test_flash_next.py -q` and verify those tests fail because the validator does not exist.
- [ ] Implement a metadata-only validator that parses config/safetensors headers without loading full tensors and returns deterministic inventory records.
- [ ] Run `pytest tests/unit/test_flash_next.py -q` and verify the contract tests pass.
- [ ] Record the pinned contract in `docs/models/flash-next.md`, including which facts are semantic and which are storage/checkpoint-specific.
- [ ] Commit as `research(S03F-01): pin Flash-Next contract`.

### Task 2: Produce exact tensor and capacity ledgers

**Files:**
- Modify: `python/superinfer/convert/flash_next.py`
- Modify: `tests/unit/test_flash_next.py`
- Create: `artifacts/S03F/flash-next-memory-ledger.json`

**Interfaces:**
- Produces: deterministic category totals for routed experts, shared experts, PLE, router/indexer, non-expert text weights, embedding/LM head, recurrent/KV state, workspace estimate, vision, and MTP.

- [ ] Add failing tests for category completeness, total-byte reconciliation, and deterministic ordering/hash.
- [ ] Run the focused test and verify failure.
- [ ] Implement category classification from exact tensor names and packed byte ranges; every tensor must belong to exactly one category.
- [ ] Add state/workspace formulas parameterized by context and batch=1 without counting excluded vision/MTP toward the initial text runtime budget.
- [ ] Run focused tests and then `python tools/validate.py --full`.
- [ ] Generate `artifacts/S03F/flash-next-memory-ledger.json` from the pinned artifact/model directory and include source hashes and timestamp-independent canonical content.
- [ ] Commit as `research(S03F-01): record Flash-Next capacity ledger`.

### Task 3: Evaluate residency/quantization recipes

**Files:**
- Modify: `python/superinfer/convert/flash_next.py`
- Modify: `tests/unit/test_flash_next.py`
- Create: `artifacts/S03F/flash-next-residency-options.json`
- Modify: `.planning/DECISIONS.md`

**Interfaces:**
- Produces: candidate per-category bit-width/storage recipes with per-GPU packed-byte projections, minimum headroom, and a binary `full_expert_residency_feasible` result.

- [ ] Add failing tests for deterministic contiguous partitioning over layer byte totals and rejection when either device exceeds its configured budget/headroom.
- [ ] Implement an offline planner that evaluates candidate expert/PLE/non-expert storage recipes without changing runtime code.
- [ ] Record at least one fidelity-first recipe and one fit-first recipe. Do not claim quality equivalence without model evaluation evidence.
- [ ] If no acceptable full-residency recipe fits, record an explicit ADR requirement before expert staging/caching may be implemented.
- [ ] Run focused tests and full validation.
- [ ] Commit as `research(S03F-01): evaluate dual-5090 residency`.

### Task 4: Close research-only gate

**Files:**
- Create: `.planning/phases/S03F-flash-next/S03F-01-SUMMARY.md`
- Modify: `.planning/STATE.md`

- [ ] Summarize pinned architecture, exact bytes, candidate residency recipes, unresolved quality measurements, and the recommended S03F-02 memory budget.
- [ ] State explicitly that no runtime code was changed and that S03F-02 remains blocked until S03 closes.
- [ ] Update planning state with S03F-01 research status without changing the primary implementation phase away from S03.
- [ ] Commit as `docs(S03F-01): close Flash-Next research gate`.
