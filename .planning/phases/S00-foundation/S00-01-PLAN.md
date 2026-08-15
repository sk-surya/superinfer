---
phase: "S00-foundation"
plan: "S00-01"
type: "feature"
wave: 1
depends_on: []
files_modified:
  - CMakeLists.txt
  - cmake/**
  - include/superinfer/base/**
  - include/superinfer/{ir,compiler,artifact,runtime,kernels,decode}/**
  - src/base/**
  - tests/unit/base/**
  - tests/architecture/**
autonomous: true
requirements_addressed: [ARCH-001, ARCH-004, ARCH-005, ARCH-008]
must_haves:
  truths:
    - "The three representations and five extension surfaces are visible, narrow contracts."
    - "Ownership and errors are explicit before CUDA/model code exists."
    - "The runtime layer cannot depend on frontends or semantic compiler implementation."
  artifacts:
    - "A reproducible C++20 project skeleton and target graph"
    - "Typed Status/Result, checked arithmetic, IDs and non-owning view primitives"
    - "Compile-time interface conformance and dependency-boundary tests"
---

# S00-01 — Build Skeleton and Architectural Contracts

## Objective

Create the smallest buildable C++20/CUDA-ready skeleton that makes the approved boundaries executable. This plan owns foundational interfaces, not their model/GPU implementations.

## Tasks

1. **Select and pin the build/tooling baseline**
   - Evaluate the installed CUDA/C++ environment and record minimum compiler/CMake/tool versions in a checked-in toolchain document.
   - Configure CMake presets for CPU developer/CI and optional CUDA `sm_120a`; CPU configuration must not require a GPU driver.
   - Use target-scoped compile features/options and warnings. Keep third-party dependencies explicit and minimal.
   - Add install/export namespaces without promising stable ABI in V0.

2. **Create layer targets and enforce dependency direction**
   - Add targets for base, IR, compiler contracts, artifact contracts, kernel contracts, decode contracts, and runtime contracts.
   - Define allowed dependencies from `.planning/ARCHITECTURE.md`; encode them in CMake rather than relying on convention.
   - Add a test/script that fails when runtime includes frontend or semantic compiler headers, and when provider selection references a model identifier.

3. **Implement base value and error primitives**
   - Implement typed `StatusCode`, contextual `Status`, and move-aware `Result<T>` without exceptions across runtime/CUDA boundaries.
   - Add checked size/offset arithmetic, strongly typed IDs, spans/views, byte-size/alignment helpers, and explicit host/device memory-space enums.
   - Document ownership, lifetime, thread safety, and failure behavior on every public primitive.
   - Test move-only values, context propagation, overflow, alignment, empty views, and misuse detected at compile time.

4. **Define representation shells**
   - Define distinct root types/namespaces for Semantic IR, Lowered IR, and immutable Physical Plan.
   - Do not add operation catalogs yet; expose stable identity/version, deterministic traversal/dump contracts, and verifier interfaces.
   - Make illegal cross-representation implicit conversion impossible.

5. **Define the five extension interfaces**
   - `ModelFrontend`: validate source inventory and emit Semantic IR.
   - `GraphPass`: declare representation, preconditions/effects/invalidation and apply deterministically.
   - `KernelProvider`: enumerate capability-described candidates without model names.
   - `DecodeStrategy`: declare compile-time state/graph/resource requirements and runtime transition interface.
   - `StoragePolicy`: plan/package/materialize typed tensor storage without exposing file offsets to kernels.
   - Favor value descriptors and explicit registries assembled by the application; defer shared-library plugin loading.

6. **Add architecture fitness tests**
   - Compile a minimal fake implementation of each extension surface.
   - Assert three representations are distinct types and a Physical Plan is immutable to executor consumers.
   - Scan dependency metadata/source includes for forbidden edges and model-family strings in runtime.
   - Add a failing test fixture demonstrating the boundary check catches a prohibited edge.

## Verification

- Configure, build, install, and test with CUDA disabled from a clean build directory.
- Configure CUDA mode far enough to validate target capability flags when the toolchain is present; skip with an explicit reason otherwise.
- Run unit and architecture suites under sanitizers available on the host.
- Inspect generated dependency graph and retain it as `artifacts/S00/architecture-targets.json` or equivalent test output.

## Completion Evidence

- Build/test transcript with tool versions.
- Passing interface conformance and forbidden-dependency checks.
- Public API documentation demonstrates ownership/error contracts.
- No model, tensor format, or kernel implementation has leaked into the foundation.
