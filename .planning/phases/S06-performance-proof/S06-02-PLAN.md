---
phase: "S06-performance-proof"
plan: "S06-02"
type: "verification"
wave: 2
depends_on: [S06-01]
files_modified:
  - python/superinfer/report/**
  - benchmarks/reports/rtx5090-qwen38-v0/**
  - tests/{unit,golden,integration}/report/**
  - docs/benchmarks/rtx5090-qwen38-v0.md
  - artifacts/S06/**
autonomous: false
requirements_addressed: [BEN-002, BEN-003, BEN-004, BEN-005, QUA-003, QUA-005]
must_haves:
  truths:
    - "The first graph is generated from eligible raw data and can be rebuilt on a clean checkout."
    - "Every visible claim names its exact workload, semantics, versions and uncertainty."
    - "An independent audit can trace each plotted value to raw samples and correctness/environment evidence."
  artifacts:
    - "Generated RTX 5090 Qwen3.8-27B graph, accessible table and report"
    - "Claim/equivalence audit and raw-evidence checksum manifest"
    - "Clean-checkout and second-session reproduction records"
---

# S06-02 — Generate, Audit, and Reproduce the First Performance Graph

## Objective

Create the first public-ready Qwen3.8-27B RTX 5090 graph with complete traceability and independent reproduction evidence.

## Tasks

1. **Implement deterministic report generation**
   - Read only schema-valid eligible run bundles; verify all hashes/identities and matched-comparison flags.
   - Compute declared robust summaries, spread/percentiles and repeat/session consistency without deleting raw samples.
   - Produce machine-readable report JSON, CSV table, accessible Markdown/HTML table and SVG/PNG as needed.
   - Normalize timestamps/run IDs for deterministic golden structure while retaining real identity in provenance.

2. **Design the narrow first graph**
   - Include separate prefill/TTFT and decode/TPOT-or-tokens-per-second panels across the declared context/prompt grid; include peak memory table/panel.
   - Clearly distinguish SuperInfer baseline/optimized and any matched external baseline.
   - Put artifact quantization, output length, strategy, concurrency, power/clock policy and versions in caption/metadata.
   - Avoid truncated axes, dual-axis ambiguity, unsupported precision and microbenchmark-to-E2E implication.

3. **Build value-to-evidence traceability**
   - Every point links to manifest/run IDs and raw sample indexes.
   - Report includes correctness status, environment validity, rejected runs, exact commands, known limitations and unmatched dimensions.
   - Generate a checksum manifest for all report inputs/outputs.

4. **Run claim and comparison audit**
   - Review model/artifact/quantization/tokenizer/prompt/sampling/context/output/concurrency/timing/GPU equivalence.
   - Review statistical/practical interpretation, title/caption/alt text and whether every comparative statement is exactly supported.
   - Remove or narrow claims that cannot pass audit; never fill missing evidence with estimates.

5. **Perform clean-checkout reproduction**
   - On a fresh checkout/environment, acquire or reference the exact artifact by hash, validate inputs, run the documented command, regenerate summary/table/graph and verify expected structural/value tolerance.
   - Perform a second target session/run on the original or independent qualified 5090 and compare against repeat policy.
   - Record all deviations and whether they invalidate the public artifact.

6. **Publish the evidence packet in repository-safe form**
   - Check in manifests, report schema output, tables, graph, narrative and raw-bundle checksums/retrieval instructions.
   - Do not check in model weights or large raw data contrary to license/repository limits.

## Verification

- Report golden/unit tests and invalid-bundle rejection pass.
- Clean checkout regenerates the graph/table from documented evidence.
- Claim audit checklist has no unresolved blocker.
- A reviewer can select any graph point and locate exact eligible raw samples.

## Completion Evidence

- First reproducible RTX 5090 Qwen graph and accessible table.
- Signed/approved claim audit.
- Clean-checkout plus second-session reproduction transcript/checksums.
