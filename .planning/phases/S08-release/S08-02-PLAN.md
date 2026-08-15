---
phase: "S08-release"
plan: "S08-02"
type: "release"
wave: 2
depends_on: [S08-01]
files_modified:
  - README.md
  - LICENSE*
  - CHANGELOG.md
  - docs/**
  - packaging/**
  - examples/**
  - .github/workflows/release.yml
  - artifacts/S08/rehearsal/**
autonomous: false
requirements_addressed: [QUA-005, REL-001, REL-002, MOD-001, MOD-005, BEN-005]
must_haves:
  truths:
    - "A clean supported machine can build/install, validate/inspect an artifact, and generate tokens from documented commands."
    - "Support and limitations are exact; model weights are not improperly redistributed."
    - "Release packages identify source/toolchain/schema versions and are reproducible enough to audit."
  artifacts:
    - "User/developer/model/benchmark documentation and examples"
    - "Versioned source/package release candidates with checksums/SBOM/license inventory"
    - "Clean-machine Qwen and Gemma rehearsal transcripts"
---

# S08-02 — Documentation, Packaging, and Clean-Machine Rehearsal

## Objective

Make the proven system understandable and installable by a new contributor/operator, and rehearse the complete supported workflows from clean machines.

## Tasks

1. **Finalize support/version policy**
   - Declare V0 semantic version, supported OS/compiler/CMake/Python/CUDA/driver/RTX 5090 envelope, `.sinf` versions, model repositories/revisions/configurations, quantization/context constraints and non-goals.
   - Define deprecation/compatibility expectations for pre-1.0 artifacts/APIs.

2. **Create user workflow documentation**
   - Install/build with CPU-only validation and optional CUDA target.
   - Acquire/pin model inputs legally, convert to `.sinf`, inspect/validate, run prompts, select declared decode strategy and interpret diagnostics.
   - Reproduce the S06 benchmark/report without implying any unsupported configuration.

3. **Create contributor/architecture documentation**
   - Explain three representations, five extension surfaces, source ownership, how to add a frontend/pass/provider/strategy/storage policy, testing/evidence and autoresearch safety.
   - Include a traceable Qwen tensor-to-kernel walkthrough and Gemma extensibility proof.
   - Document debug/fallback/reference modes and hot-path constraints.

4. **Build release packaging**
   - Produce source archive and selected binary/Python packages appropriate to the supported environment, with versions, checksums, dependency/SBOM/license inventory and reproducible build commands.
   - Keep CUDA/library redistributability and model licenses explicit.
   - Package example manifests/fixtures only; never include restricted weights.

5. **Rehearse Qwen clean-machine workflow**
   - From clean checkout/toolchain, build/install, acquire/point to pinned inputs, convert or validate exact artifact, inspect and generate acceptance tokens on qualified RTX 5090.
   - Regenerate the S06 report from evidence and record all manual prerequisites/time/disk/memory.

6. **Rehearse Gemma clean-machine workflow**
   - Repeat install/acquisition/artifact validation/generation for supported Gemma scope.
   - Verify no undocumented local cache/environment dependency.

7. **Draft release notes/changelog**
   - State what works, exact model/hardware configurations, performance evidence link, architecture, known limitations, security/trust boundary and next milestones.

## Verification

- Documentation link/command checks and package install/import/link tests pass.
- Qwen and Gemma rehearsals succeed from clean machines/environments.
- Checksums/SBOM/license inventory validate and packages contain no forbidden data/secrets/absolute paths.
- A new reader can follow README to the first safe CPU validation and target GPU generation.

## Completion Evidence

- Release-candidate artifacts/checksums/inventory.
- Qwen/Gemma clean-machine transcripts and environment manifests.
- Reviewed docs/support matrix/release notes.
