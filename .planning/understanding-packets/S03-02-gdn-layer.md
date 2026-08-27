# S03-02 Understanding Packet: artifact-bound Gated-DeltaNet state

## 1. What changed

Layer 0 now executes the complete Qwen3.8 Gated-DeltaNet decoder layer from the real NVFP4
artifact. Two one-token calls share the same Physical Plan and preserve BF16 convolution history
and FP32 recurrent state; the path then performs the gated MLP and final residual.

## 2. Why it exists

Gated-DeltaNet is the stateful half of Qwen3.8. Proving only one invocation could hide a broken
state transition, so the fixture checks the same plan over two segments against the pinned
Transformers implementation. The recurrent kernel remains a generic state update; model topology
and projection composition stay in compile-time lowering.

## 3. Execution/data path

The test memory-maps the authenticated `.sinf`, resolves layer-0 tensor records, converts only
BF16 auxiliary weights to FP32 for the baseline kernels, and uploads the required ranges once.
Each execution normalizes the input, performs QKV projection and causal convolution, splits Q/K/V,
derives decay and beta, updates recurrent state, gates with Z, projects out, and runs the MLP.
The second call reuses the mutated state buffers.

## 4. Important shapes and data structures

- hidden input/output: `[5120]` FP32;
- convolved QKV: `[10240] = 16*128 + 16*128 + 48*128`;
- Z projection: `[6144] = 48*128`;
- query/key: 16 heads × 128; value: 48 heads × 128;
- convolution state: BF16 `[4, 10240]`;
- recurrent state: FP32 `[48, 128, 128]`;
- auxiliary A/B/log-decay/beta vectors: `[48]`;
- plan: 20 commands per segment, 40 launches over the two calls.

## 5. Core invariants

- State buffers are initialized before first execution and are not reallocated between segments.
- The recurrent state is updated in place only by the generic Gated-Delta command.
- Query/key normalization and value-head repetition match the pinned reference contract.
- Artifact NVFP4 sidecars remain explicit operands; BF16 auxiliary weights are explicitly converted.
- No model identity, artifact lookup, or allocation occurs in the executor hot path.

## 6. Performance model

This is correctness evidence, not a performance claim. The baseline is launch- and memory-bound:
three large NVFP4 projections and the MLP dominate work, while the recurrent update touches about
3 MiB of FP32 state per segment. S04 may optimize only after this stateful contract is stable.

## 7. Likely failure modes

- First segment passes but second diverges: inspect convolution state layout and recurrent state copy.
- State boundary drifts while final output is large: inspect decay/beta derivation before MLP output.
- Large attention boundary drift: inspect Q/K/V split or value-head grouping.
- Setup failure: inspect BF16-to-FP32 materialization sizes and tensor-table sidecar names.

## 8. Exactly three files to read

1. `tools/qwen38_nvfp4_gdn_reference.py`
2. `tests/gpu/sm120/qwen38_gdn_artifact_test.cu`
3. `backends/sm120/runtime/cuda_plan_executor.cuh`

## 9. One hands-on experiment

Run the two-segment artifact differential on GPU 0, then zero the recurrent state between the two
`session.execute()` calls. Predict that segment 0 remains within contract while segment 1 changes
substantially, proving that state continuation is observable.

## 10. Five questions

1. Which state is BF16 and which state is FP32, and why are their precisions different?
2. How does the QKV vector split into the authored Qwen head topology?
3. Where are `A_log`, `dt_bias`, and the input-dependent A/B projections combined?
4. Which buffer is mutated by the recurrent kernel, and how does the second call consume it?
5. Why is a final-output comparison insufficient to diagnose a state-transition failure?
