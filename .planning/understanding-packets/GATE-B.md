# Gate B — One-token transformer execution

## 1. What changed

SuperInfer now constructs a validated `sm_120a` Physical Plan, binds exact tensor operands to four
correctness-first CUDA baselines, owns device memory/streams/events, and executes dependency order
without hot-path allocation or device-wide synchronization. The CPU reference path independently
computes residual and normalization intermediates.

## 2. Why it exists

This protects the central boundary: semantic meaning and compiler decisions become an immutable
Physical Plan before runtime. The executor must not rediscover model policy, infer operands from
allocation order, silently truncate shapes, accept unknown catalogs, or claim success for a no-op.

## 3. One execution path

For the current four-element synthetic norm fixture:

`f32 input [4] + f32 scale [4]`
→ Lowered tensor operands
→ Physical buffers at 16-byte-aligned arena offsets
→ command `(kernel=5, buffers=[input, output, scale], epsilon=1e-5)`
→ prebound CUDA launcher on an `sm_120a` stream
→ one-thread RMSNorm kernel
→ event record
→ explicit test-boundary synchronization
→ copied output `[4]` compared with the independent CPU formula.

Qwen-derived `[batch, sequence, hidden]` and KV shapes are intentionally pinned in S03-01; this
packet uses the smallest executable trace rather than inventing model dimensions.

## 4. Important data structures

- Semantic tensor: meaning-level shape/dtype/role; no CUDA address.
- Lowered tensor: physical shape/layout/storage dtype and explicit kernel operands.
- Physical buffer: arena offset, byte size, alignment.
- Physical command: stable kernel ID, ordered buffer bindings, dependencies, stream, workspace slice,
  and norm epsilon.
- CUDA session: copied plan, prebound function pointers, arenas, streams, events, command order,
  and bounded lifecycle trace.

## 5. Core invariants

- Only `baseline-v1` is accepted by this registry.
- Every command is validated before device arena allocation.
- Supported commands require exact operand counts and equal f32 sizes where applicable.
- `execute()` performs no heap allocation, filesystem access, model-family branch, kernel lookup, or
  device-wide synchronization.
- Any observed CUDA execution/synchronization error poisons the session.
- Unsupported operations fail explicitly; they are not mapped to stubs.

## 6. Performance model

The current kernels are correctness baselines, not performance candidates. Norms use one thread and
are intentionally memory/latency poor. The runtime target is eliminating per-token control-plane
work: all allocation, binding, dependency ordering, and stream/event creation are construction-time
costs. A later optimized kernel must preserve the same operand contract and differential evidence.

## 7. Likely failure modes

- Wrong command buffer ordering: inspect lowered operands and Physical Plan command buffers.
- Wrong tensor sizes: creation returns an invalid-argument diagnostic before `cudaMalloc`.
- Wrong target/catalog: session creation returns unsupported before allocation.
- Async device fault: explicit synchronization returns an error and `poisoned()` becomes true.
- Misleading timing: `execute()` is asynchronous; use explicit synchronization or CUDA events.

## 8. Exactly three files to read

1. `backends/sm120/compiler/specializer.h`
2. `backends/sm120/runtime/cuda_plan_executor.cuh`
3. `backends/sm120/kernels/baseline/reference_executor.h`

## 9. Hands-on experiment

Change the first command in the GPU fixture from `kernel=4` residual to `kernel=5` RMSNorm without
changing its three-buffer operand list. Prediction: session creation succeeds because the arity is
valid, but the result changes from `left + right` to normalized `input * scale`; changing the scale
values should change each output element proportionally. Restore the command after observing it.

## 10. Five questions

1. Why must the specializer provide explicit operand buffer IDs instead of passing every allocation?
2. Which work happens during session construction, and which work remains in `execute()`?
3. Trace one `[4]` input element from host upload to the RMSNorm CUDA operation and output copy.
4. What observable status and session state should follow an asynchronous CUDA fault?
5. Why would the current one-thread norm be a correctness baseline but not a performance candidate?
