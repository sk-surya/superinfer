---
phase: "S02-sm120-baseline"
plan: "S02-02"
type: "feature"
wave: 2
depends_on: [S02-01]
files_modified:
  - backends/sm120/{runtime,kernels/baseline}/**
  - include/superinfer/runtime/**
  - src/runtime/**
  - tests/{unit,integration,gpu}/sm120/**
autonomous: true
requirements_addressed: [ARCH-004, BCK-004, BCK-005, BCK-006, KER-001, KER-002, KER-005]
must_haves:
  truths:
    - "The executor validates/binds once and executes only Physical Plan commands."
    - "Every synthetic required operation has a correct baseline and fallback."
    - "Runtime code contains no model-family knowledge."
  artifacts:
    - "RAII CUDA/device/session layer and minimal plan executor"
    - "Baseline `sm120` KernelProvider and stable kernel registry"
    - "Independent CPU/reference graph executor for differential tests"
---

# S02-02 — Minimal Executor and Baseline Kernel Provider

## Objective

Execute small validated Physical Plans on RTX 5090 with explicit lifecycle and independent differential correctness, without optimizing for speed.

## Tasks

1. **Implement explicit CUDA ownership wrappers**
   - Add move-only device buffer/arena, stream, event, module/kernel registry and session owners with contextual errors.
   - Define teardown/error poisoning behavior and prevent use-after-failure.
   - Keep CUDA API calls behind a narrow backend boundary and make synchronization explicit.

2. **Implement runtime construction/binding**
   - Parse/validate artifact and target profile, materialize StoragePolicy plan, allocate all arenas, resolve stable kernel IDs, upload constants/weights, and bind immutable command views.
   - Reject any unsupported resource/capability before first launch.
   - Provide bounded trace metadata outside hot execution for test/profiling.

3. **Implement minimal Physical Plan executor**
   - Execute plan-authored command dependencies/entry schedules over prebound buffers/launch descriptors.
   - Add prefill/decode entry APIs with caller-owned token/output views and preallocated state.
   - Do not inspect semantic IR, model config, model name, filesystem, or compiler/provider policy.

4. **Implement baseline KernelProvider**
   - Register simple correct kernels/library calls for copy/cast, elementwise/residual, norm, RoPE, small matmul/GEMV, embedding, softmax/attention primitive, top-k/router, activation/gate and sampling primitives needed by synthetic graphs.
   - Each capability declares dtype/layout/shape/alignment/workspace/determinism/numerical contract.
   - Return explicit inapplicable reasons and allow selector fallback.

5. **Implement independent reference executor**
   - Execute small Semantic/Lowered fixtures on CPU/high-precision reference utilities without calling production GPU kernels.
   - Capture intermediate named tensors for differential localization.
   - Use fixed/randomized fixtures and dtype-derived tolerance records.

6. **Add executor/provider integration tests**
   - Run synthetic dense, attention-state-update, and tiny MoE plans end to end.
   - Compare intermediate/final outputs and state across repeated decode steps.
   - Exercise fallback selection and unsupported capability diagnostics.

## Verification

- GPU integration tests match independent references over random/boundary shapes.
- Executor source/dependency fitness test contains no frontend/model-family imports/identifiers.
- Repeated session construct/execute/destroy succeeds without leaks on target hardware.
- CPU CI builds runtime interfaces and runs selector/reference tests with GPU tests explicitly skipped.

## Completion Evidence

- Baseline operation/capability matrix.
- Differential result bundles for dense, attention, and MoE fixtures.
- Runtime construction/execution trace showing prebound resources.
