---
phase: "S08-release"
plan: "S08-03"
type: "verification"
wave: 3
depends_on: [S08-02]
files_modified:
  - .planning/STATE.md
  - .planning/REQUIREMENTS.md
  - .planning/RISKS.md
  - artifacts/S08/release-audit/**
  - docs/releases/v0.md
autonomous: false
requirements_addressed: [QUA-001, QUA-002, QUA-003, QUA-004, QUA-005, REL-001, REL-002, BEN-005]
must_haves:
  truths:
    - "Every V0 requirement and public claim is backed by retrievable passing evidence."
    - "No High/Critical release blocker is hidden or waived without a recorded decision."
    - "The release tag/package corresponds exactly to the audited commit and artifact set."
  artifacts:
    - "Requirement/evidence and risk-closure audit"
    - "Final V0 release manifest and publication checklist"
    - "Archived benchmark/correctness/compatibility/rehearsal index"
---

# S08-03 — Milestone Evidence Audit and V0 Publication Packet

## Objective

Perform a goal-backward audit of the complete milestone and prepare the exact, immutable V0 publication set. This plan may block release; it may not invent missing evidence.

## Tasks

1. **Audit requirement coverage**
   - For every ARCH/FMT/MOD/BCK/KER/DEC/RES/BEN/QUA/REL requirement, link implementation commit, automated/manual test, evidence bundle and reviewer status.
   - Mark Passed, Partial, Failed or Not Applicable with rationale; Partial/Failed milestone requirements block release unless scope is explicitly changed by a recorded decision.
   - Verify evidence hashes/retrieval and rerun a representative sample.

2. **Audit architectural invariants**
   - Re-run dependency/model-identifier/three-IR/five-surface/executor-immutability and hot-path allocation/sync fitness tests.
   - Review `.sinf` validation, target compatibility, provider fallback and DecodeStrategy/DSpark placement.
   - Confirm Gemma executor-unchanged proof against the audited release commit.

3. **Audit correctness and benchmark claims**
   - Verify Qwen and Gemma acceptance reports against exact release artifacts/runtime/target.
   - Re-run report checksum/point traceability and clean-checkout reproduction verification for the S06 graph.
   - Compare every README/release/social-ready performance sentence to the audited evidence; narrow/remove unsupported claims.

4. **Review and close the risk register**
   - Reassess every risk, attach mitigation evidence and identify remaining limitations.
   - No unresolved High/Critical safety/correctness/evidence risk may be silently accepted; record explicit owner/decision/scope change if release proceeds.

5. **Create final release manifest**
   - Pin release commit/tag, source/package checksums, SBOM/license inventory, supported `.sinf`/model/target matrix, Qwen/Gemma artifact recipes/hashes, benchmark report/evidence index and known limitations.
   - Ensure all publication artifacts derive from the audited commit and are immutable/retrievable.

6. **Execute final publication checklist**
   - Required CI green, clean-machine rehearsals green, docs links/commands green, no secrets/restricted weights, vulnerability/reporting links valid, changelog/version consistent.
   - Obtain human approval for tag/release publication; do not publish automatically from the planning executor.
   - Update project state/requirements only after the exact tag and checksums are confirmed.

## Verification

- Independent requirement/evidence script reports full milestone coverage.
- All architecture/correctness/compatibility/benchmark reproduction gates pass at the release commit.
- Release manifest/checksums match built artifacts and tag candidate.
- Human release approval is recorded with any explicit residual limitations.

## Completion Evidence

- Final V0 requirement/risk/claim audit.
- Immutable release manifest and evidence index.
- Approved publication checklist and release/tag identifiers.
