---
phase: "S08-release"
plan: "S08-01"
type: "hardening"
wave: 1
depends_on: [S07-03]
files_modified:
  - tests/{fuzz,negative,compatibility,gpu}/**
  - .github/workflows/**
  - schemas/**
  - SECURITY.md
  - docs/{compatibility,security}.md
  - artifacts/S08/hardening/**
autonomous: false
requirements_addressed: [QUA-001, QUA-002, QUA-003, QUA-004, FMT-002, BCK-003, REL-002]
must_haves:
  truths:
    - "Untrusted artifacts/plans/configs cannot bypass validation into allocation or launch."
    - "Supported schema/artifact versions have an explicit tested matrix."
    - "Release-blocking CPU/GPU correctness and sanitizer lanes are stable and owned."
  artifacts:
    - "Negative/fuzz/security hardening report and minimized corpus"
    - "Artifact/plan/schema compatibility matrix"
    - "Release CI matrix with required GPU lanes"
---

# S08-01 — Security, Compatibility, and CI Hardening

## Objective

Close unsafe-input, lifecycle, compatibility and test-infrastructure gaps before packaging V0.

## Tasks

1. **Threat-model release inputs and boundaries**
   - Review `.sinf`, config/tokenizer metadata, experiment/benchmark manifests, CLI paths, generated reports, CUDA plan launches and external model acquisition.
   - Identify integer overflow, path traversal, decompression/resource exhaustion, alias/OOB, malicious capability IDs, log/data leakage and supply-chain risks.
   - Map mitigations/tests and document V0 trust boundary (local single-user research runtime, not multi-tenant service).

2. **Expand parser/plan fuzz and negative suites**
   - Run sustained artifact/schema/frontend/Physical Plan fuzzing; minimize crashes/hangs/high-resource cases into permanent fixtures.
   - Systematically cover truncation, offset/count overflow, overlap, checksum, version/capability skew, cycles, invalid launch/workspace and excessive resource declarations.
   - Assert rejection before device allocation/launch where required.

3. **Finalize compatibility policy and matrix**
   - Define container/schema/kernel-catalog/provider/tuning compatibility ranges and exact diagnostic/migration policy.
   - Test current reader with all supported golden writer versions and current writer with the release reader.
   - Test Qwen/Gemma artifacts, unknown optional sections, unknown required capabilities and target mismatch.

4. **Harden GPU lifecycle/sanitizer suites**
   - Run compute-sanitizer modes supported on target, long-context/canary, repeated load/unload, error poisoning/reconstruction, OOM/resource rejection and hot-path trace.
   - Persist target/toolchain manifest, failing seeds and bounded diagnostics.

5. **Stabilize release CI tiers**
   - Make CPU fast/full required and deterministic.
   - Define GPU smoke, correctness, scheduled sanitizer and controlled benchmark triggers/ownership/timeouts/artifacts.
   - Ensure unavailable hardware creates an explicit pending/not-run release state, never a silent pass.

6. **Create security/reporting documentation**
   - Publish supported trust boundary, safe artifact acquisition/validation, vulnerability reporting, sensitive evidence handling and known non-goals.
   - Inventory dependency/model/code licenses and provenance.

## Verification

- Sustained fuzz campaign completes without unresolved crash/hang; minimized corpus reruns in CI.
- Full compatibility matrix passes and unsupported cases return stable diagnostics.
- All release CPU/GPU correctness/sanitizer lanes pass on qualified target.
- Threat model has test/evidence or explicit accepted limitation for every High/Critical item.

## Completion Evidence

- Hardening/fuzz report and corpus checksums.
- Compatibility matrix and release CI run IDs.
- Security/trust-boundary/license documentation.
