---
phase: S02-sm120-baseline
reviewed: 2026-08-26T04:40:25Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - CMakeLists.txt
  - cmake/SuperInferOptions.cmake
  - tests/CMakeLists.txt
  - backends/sm120/runtime/cuda_ownership.cu
  - backends/sm120/runtime/cuda_ownership.cuh
  - backends/sm120/runtime/cuda_plan_executor.cuh
  - tests/gpu/sm120/cuda_ownership_test.cu
  - tests/gpu/sm120/cuda_plan_executor_test.cu
findings:
  critical: 9
  warning: 3
  info: 0
  total: 12
status: issues_found
---

# Phase S02: Code Review Report

**Reviewed:** 2026-08-26T04:40:25Z  
**Depth:** deep  
**Files Reviewed:** 8  
**Status:** issues_found

## Summary

The CUDA ownership smoke and executor tests build and pass on the available GPU, but the current integration is not safe to ship. The executor accepts and launches kernel IDs whose implementations are no-ops, does not bind the catalog ABI to the implementation registry, and interprets command buffers with conventions that do not match the existing specializer. It also fails to validate the actual device, ignores command workspace slices, and does not poison the session when asynchronous CUDA failures are reported. These are correctness and lifecycle blockers against S02-02 and BCK-001/BCK-003/BCK-004/BCK-005/KER-001/KER-002.

## Critical Issues

### CR-01: Most advertised kernel IDs execute a no-op

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:99-114`  
**Issue:** `resolve()` maps IDs 2, 3, and 6 through 13 to `launch_stub`, which only launches an empty kernel. Those IDs are advertised as valid candidates by `BaselineProvider` (`backends/sm120/kernels/baseline/provider.h:25-34`) and are emitted by `Specializer`. A valid cast, elementwise, layer norm, RoPE, matmul, embedding, attention, MoE, activation, or sampling command therefore reports success while producing no result. The integration test uses IDs 2 and 6 but checks only launch counters, so it masks the defect.
**Fix:** Implement and differential-test each registered baseline before accepting its ID, or remove unsupported IDs from the provider and reject them during plan binding instead of mapping them to a successful stub.

### CR-02: Kernel catalog strings are not bound to a known implementation ABI

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:135-153`  
**Issue:** Creation only checks that the caller supplied the same string stored in the plan. It never verifies that the string is a supported catalog or that the numeric IDs belong to that catalog; `resolve()` is a catalog-independent switch. A plan carrying an arbitrary catalog such as `future-v2` is accepted if the caller repeats that string, then its IDs are interpreted using the hardcoded baseline mapping. This violates fail-closed capability binding and can execute an artifact against the wrong kernel ABI.
**Fix:** Resolve through a registry keyed by the catalog fingerprint, reject unknown catalogs before allocation, and validate every ID against that registry's exact operation, launch contract, and capability metadata.

### CR-03: CUDA launchers disagree with the physical plan's buffer binding

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:55-95`; caller `backends/sm120/compiler/specializer.h:81-101`  
**Issue:** The CUDA launchers assign positional meanings such as source/destination or left/right/output to the first two or three command buffers. The existing specializer, however, passes the entire physical buffer list to every command (`buffers` is reused for each `add_command`). Consequently a plan produced by the compiler can make copy, residual, and RMSNorm read or write unrelated tensors; the hand-built GPU fixture passes only the buffers it wants and does not exercise the compiler-produced binding.
**Fix:** Emit an operation-specific, validated operand list in each command (or an explicit binding table), and have the launcher consume those bindings rather than assuming global buffer order. Add an end-to-end specializer-to-CUDA test.

### CR-04: Mismatched operands are silently truncated

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:55-95`  
**Issue:** Copy, residual, and RMSNorm compute their byte/element count with `std::min` and proceed. A short destination loses the tail without an error; mismatched residual inputs produce a partial output; and a smaller RMSNorm output silently receives only a prefix. A malformed or incorrectly lowered plan therefore returns success with corrupted state.
**Fix:** Require exact compatible sizes, dtype, and layout for each operation at binding time and return a typed validation error before launch; do not use `min` as implicit shape validation.

### CR-05: RMSNorm is not a model-correct baseline contract

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:33-38,84-96`  
**Issue:** The kernel hardcodes epsilon to `1.0e-5F` and accepts only input/output buffers. It has no binding for the learned RMSNorm scale or a plan-authored epsilon. It therefore cannot implement the model RMSNorm used by the target frontend and produces incorrect activations whenever epsilon differs or the scale is non-unit. The test asserts only the hardcoded formula, making the test contract narrower than the required operation.
**Fix:** Encode epsilon and the scale operand in the lowered/physical command contract, validate them during binding, apply the scale in the kernel, and compare against the independent reference over model-representative parameters.

### CR-06: Runtime trusts a caller-supplied capability instead of probing the active GPU

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:135-143`  
**Issue:** `create()` accepts `target_capability == 120` without calling `cudaGetDevice`, `cudaGetDeviceProperties`, or checking the active device's feature/architecture limits. A caller can pass 120 while the current device is incompatible; the session then allocates resources and only discovers the mismatch at a later kernel launch (or runs a plan with insufficient resources). This violates BCK-001's requirement to reject incompatible hardware clearly before execution.
**Fix:** Probe and capture the selected device during construction, compare the observed profile and memory/resource limits to the plan fingerprint, and fail before any allocation when the device is not a supported `sm_120a` target.

### CR-07: Asynchronous CUDA failures do not poison the session

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:216-239,250-258`  
**Issue:** `execute()` checks only immediate launch, wait, and event-record return values. A kernel execution fault can be reported later by synchronization, but `synchronize_for_test()` returns that error without setting `poisoned_`; subsequent `execute()` calls are then allowed to launch more work. The copy APIs also omit the poisoned-state check. This contradicts the repository failure model that a CUDA failure poisons the active session and stops further launches.
**Fix:** Funnel every synchronization/error-observation path through a `poison(error, context)` helper, check `poisoned_` in copies and all execution entry points, and make the first asynchronous failure permanently reject further work.

### CR-08: Command workspace slices are ignored

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:41-42,55-96,225-227`  
**Issue:** The executor validates the plan's workspace bounds but passes the base of `workspace_` to every launcher and none of the launchers applies `command.workspace_offset` or `command.workspace_size`. A command that declares a nonzero workspace slice is accepted but cannot access its assigned slice, and any future kernel using the passed pointer would use the wrong region or overlap other commands.
**Fix:** Bind a per-command workspace view (`workspace_.data() + workspace_offset`, with checked size) into the launch descriptor, or reject all nonzero workspace commands until the provider/runtime contract supports them.

### CR-09: Host-device copies are not ordered with the nonblocking execution streams

**Severity:** BLOCKER  
**File:** `backends/sm120/runtime/cuda_plan_executor.cuh:76-82,241-257`  
**Issue:** Session streams are created with `cudaStreamNonBlocking`, while `copy_to_device()` and `copy_from_device()` use synchronous `cudaMemcpy` on the default stream and establish no event dependency with the session streams. A caller copying outputs immediately after `execute()` can race an in-flight kernel, and a copy used to upload inputs can race a previous execution. The test calls `synchronize_for_test()` first, so it does not expose the API ordering bug.
**Fix:** Provide explicit stream-ordered async copies with a session event/stream dependency, or make the public copy operation wait on the relevant plan events and document the synchronization boundary.

## Warnings

### WR-01: RAII reset failures can leak CUDA handles and allocations

**Severity:** WARNING  
**File:** `backends/sm120/runtime/cuda_ownership.cuh:43-63,87-103,124-141`  
**Issue:** Move assignment calls `reset()` and discards its status before overwriting the old handle. Each `reset()` also leaves the handle/pointer non-null when destruction fails. A failed `cudaFree`, `cudaStreamDestroy`, or `cudaEventDestroy` can therefore leak the old resource, and destructors silently discard the only diagnostic. This weakens the explicit ownership and teardown/error behavior promised by the plan.
**Fix:** Make replacement impossible until the old resource has been successfully released, retain teardown diagnostics in an owner/session error state, and define the behavior for device/context loss rather than silently overwriting a still-owned handle.

### WR-02: GPU tests fail instead of skipping unsupported runtime environments

**Severity:** WARNING  
**File:** `tests/gpu/sm120/cuda_ownership_test.cu:20-24`; `tests/gpu/sm120/cuda_plan_executor_test.cu:49-53`  
**Issue:** CTest is configured with skip code 77, but both tests assert that `cudaGetDeviceCount()` succeeds before they can return 77. A CUDA toolkit with no device or an insufficient driver will fail the test instead of skipping. A machine with a CUDA device of the wrong compute capability also fails the ownership test at `assert(properties.major == 12)` rather than being classified as an inapplicable sm120 lane.
**Fix:** Treat `cudaErrorNoDevice`, insufficient-driver/no-kernel-device conditions, and non-sm120 devices as `return 77` before assertions; reserve assertions for an actually applicable target.

### WR-03: The CUDA runtime library contains no compiled executor implementation

**Severity:** WARNING  
**File:** `CMakeLists.txt:59-66`; `backends/sm120/runtime/cuda_ownership.cu:1`  
**Issue:** The target named `superinfer_sm120_cuda_runtime` compiles only a translation unit that includes the ownership header. `CudaPlanSession`, its kernel resolver, and all device kernels are header-only and are compiled into each consumer/test translation unit. The added executor test therefore does not validate a compiled runtime boundary or a centralized kernel registry, and the static library itself contributes no executor implementation.
**Fix:** Move the executor and registry implementation into the CUDA runtime target (with a narrow public interface), or explicitly make the backend header-only and remove the misleading empty runtime library target; in either case, test and package the same implementation used by consumers.

## Verdict

**BLOCKED — do not ship S02-02.** The current tests pass, but they do not establish numerical correctness for the advertised operation IDs and mask the asynchronous ordering/failure paths. The runtime must first bind only implemented kernels, use correct operand/workspace contracts, validate the actual GPU, and poison sessions on deferred CUDA failures.

---

_Reviewed: 2026-08-26T04:40:25Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
