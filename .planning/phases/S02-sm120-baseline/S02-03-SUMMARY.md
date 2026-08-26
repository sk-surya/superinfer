---
phase: S02-sm120-baseline
plan: S02-03
status: complete
completed: 2026-08-26
commit: 023c505
---

# S02-03 — GPU Safety and Hot-Path Qualification

## Delivered

- Added bounded `CudaLifecycleTrace` counters for device allocation/free, stream/event ownership,
  kernel binding, and explicit synchronization boundaries.
- Qualified repeated prebound execution: three `execute()` calls add no device synchronization or
  lifecycle allocation activity.
- Added fail-closed tests for target/catalog mismatch, unknown IDs, missing IDs, exact-size operand
  violations, and nonzero unsupported workspace slices.
- Added an opt-in injected asynchronous fault lane proving synchronization errors poison the session
  and prevent later execution.
- Added GPU environment preflight/skip behavior and local RTX 5090 reproduction documentation.
- Added the machine-readable lifecycle trace schema and runtime-trace guidance.

## Evidence

- RTX 5090 devices: two GPUs, compute capability 12.0, 32,607 MiB each.
- CUDA compiler: `/usr/local/cuda-13.1/bin/nvcc`, release 13.1.115.
- `ctest --test-dir build/cuda-sm120a-13.1 --output-on-failure`: 19/19 passed twice.
- `python3 tools/validate.py --full`: all CPU, install-consumer, sanitizer, and wheel stages passed.
- Clean Compute Sanitizer memcheck on GPU1: zero errors.
- Negative poisoning fixture with `SUPERINFER_INJECT_ASYNC_FAULT=1`: passed its expected
  failure-observation assertions.
- GPU0 sanitizer was attempted but could not launch because the serving workload left insufficient
  memory. No model process was terminated or reset.

## Boundary

Gate B is reached and its packet is retained at
[GATE-B.md](../../understanding-packets/GATE-B.md). It is not marked passed by the agent. Per the
explicit autonomous-execution decision, S03 implementation may proceed while this study checkpoint
remains visible in the ledger.
