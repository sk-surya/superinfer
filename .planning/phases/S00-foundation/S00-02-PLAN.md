---
phase: "S00-foundation"
plan: "S00-02"
type: "feature"
wave: 2
depends_on: [S00-01]
files_modified:
  - pyproject.toml
  - python/superinfer/**
  - tests/{support,unit,property,golden}/**
  - .github/workflows/**
  - .pre-commit-config.yaml
  - docs/development.md
autonomous: true
requirements_addressed: [ARCH-008, QUA-001, QUA-003]
must_haves:
  truths:
    - "A contributor can run useful validation without an NVIDIA GPU."
    - "Reference/oracle utilities are independent from optimized production code."
    - "Failures retain deterministic seeds and actionable artifacts."
  artifacts:
    - "CPU fast and full CI lanes"
    - "Typed Python package and CLI/test skeleton"
    - "Reusable deterministic fixture, property, golden and evidence helpers"
---

# S00-02 — Test, CI, and Developer Workflow Foundation

## Objective

Make correctness and readability the default development path before functionality grows. This plan owns common test/evidence infrastructure and CPU CI.

## Tasks

1. **Create the typed Python control-plane package**
   - Configure Python 3.12+, formatting, linting, strict-enough type checking, tests, and a `superinfer` CLI shell.
   - Keep optional model/GPU/research dependencies in named extras; importing the core package must not import Torch/CUDA/Hugging Face.
   - Add typed result/error translation at CLI boundaries with stable exit categories.

2. **Build deterministic test support**
   - Create deterministic RNG/seed capture, temporary artifact workspace, golden update guard, structured evidence writer, and subprocess helpers.
   - Create tiny tensor/graph fixture builders that do not depend on target models.
   - Define how C++ tests emit machine-readable failure context consumed by CI.

3. **Establish independent reference scaffolding**
   - Add a simple high-precision tensor/reference utility boundary for future op and graph oracles.
   - Ensure production libraries cannot link to test reference code.
   - Provide comparison utilities with dtype-specific contracts and useful max-error/index diagnostics; do not choose final tolerances for operations not yet defined.

4. **Add local quality commands**
   - One documented command configures/builds/tests C++ CPU code and validates Python.
   - Separate fast and full profiles; full includes sanitizers/property/golden checks available without GPU.
   - Add format/check modes that never rewrite files in CI.

5. **Create CPU CI lanes**
   - Fast PR lane: formatting, lint, types, configure/build, focused units, architecture fitness.
   - Full CPU lane: all unit/property/golden tests, sanitizer configuration, package/build/install check.
   - Cache only reproducible dependencies/build outputs; print compiler/tool versions and upload structured failure artifacts.
   - Add timeouts and cancellation/concurrency behavior.

6. **Document contributor workflow and test ownership**
   - Explain directory ownership, how to add a test, update a golden, preserve a failing property seed, and attach evidence.
   - Document CUDA/GPU lanes as future work without making CPU contributors install CUDA.
   - Link `AGENTS.md`, architecture, quality, and the active phase workflow.

## Verification

- Run all local fast/full commands twice from clean directories.
- Build/install Python wheel and C++ install tree, then import/link a minimal consumer.
- Deliberately trigger one C++ and one Python test failure and verify seed/context artifacts are retained.
- Validate workflow syntax and confirm no job assumes repository secrets or GPU access.

## Completion Evidence

- Green CPU CI run URL/ID recorded in the phase summary.
- Clean install/import/link transcript.
- Example structured failing-test artifact.
- Test ownership/developer guide reviewed against `AGENTS.md`.
