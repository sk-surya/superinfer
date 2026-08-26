---
phase: S02-sm120-baseline
plan: S02-02
status: complete
completed: 2026-08-26
commit: 21ecf1d
---

# S02-02 — Minimal Executor and Baseline Kernel Provider

## Delivered

- Added move-only CUDA device-buffer, stream, and event ownership with contextual status values.
- Added a CUDA Physical Plan session that probes the active device, validates the `baseline-v1`
  catalog, binds launch functions before allocation, orders command dependencies, and executes
  without allocation or device-wide synchronization in `execute()`.
- Added exact physical operand binding from Lowered IR tensor operands through the specializer;
  launchers no longer infer operands from the complete allocation list.
- Added explicit norm affine operands and plan epsilon metadata. The CUDA baseline now supports
  exact-shape `copy`, `residual`, `rms_norm`, and `layer_norm` commands (IDs 1, 4, 5, and 6).
- Added an independent CPU reference path with named intermediate tensors and optional norm affine
  tensors.
- Added catalog, target, operand-size, workspace, lifecycle, compiler-produced-plan, and GPU
  numerical tests.
- Preserved the review report at [S02-02-REVIEW.md](S02-02-REVIEW.md); its stale no-op findings
  were resolved by removing unsupported IDs from both provider and specializer, then adding the
  binding/device/error-path fixes it identified.

## Capability boundary

The provider intentionally does not advertise `cast`, `elementwise`, `rope`, `matmul`, `embedding`,
`attention`, `moe_route`, `activation`, or `sampling` until each has a physical operand/parameter
contract and an independent differential fixture. A plan requesting one receives an explicit
unsupported status. This is a correctness boundary, not a claim that the Qwen operation matrix is
complete; those baselines remain required work before model integration.

## Evidence

- `ctest --test-dir build/cuda-sm120a-13.1 --output-on-failure`: 19/19 passed on the RTX 5090.
- `python3 tools/validate.py --full`: all validation stages passed, including CPU, install-consumer,
  sanitizer, and wheel lanes.
- Focused `superinfer_sm120_cuda_plan_executor` passed with residual, RMSNorm, LayerNorm, and a
  compiler-produced Lowered-IR-to-CUDA plan.
- Compute Sanitizer remains a separate S02-03 item; its run was blocked by CUDA memory allocation
  failure while user model processes occupied the GPU, and those processes were not disturbed.

## Follow-up

S02-03 must qualify invalid-plan rejection, steady-state allocation/synchronization envelopes,
poisoning and teardown behavior, sanitizer applicability, and the RTX 5090 smoke workflow before
the backend is treated as safety-qualified.
