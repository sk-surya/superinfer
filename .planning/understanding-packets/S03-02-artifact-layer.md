# S03-02 Understanding Packet: artifact-bound full-attention layer

## 1. What changed

The S03-02 path now parses the authenticated `.sinf` tensor table into typed records and binds
those records directly to a manually composed Physical Plan. A complete Qwen3.8 layer-3
full-attention path executes on an RTX 5090: Qwen RMSNorm, four NVFP4 projections, explicit
interleaved q/gate split, q/k normalization, partial RoPE, BF16 KV-cache append, cached GQA,
sigmoid output gating, output projection, residual, post-attention RMSNorm, gated NVFP4 MLP,
and final residual.

## 2. Why it exists

This closes the gap between a synthetic compilable graph and real model bytes. The artifact table
is the ownership boundary for payload ranges and physical encodings; the Physical Plan owns
typed buffers and command order; the executor only launches validated generic kernel IDs. The
split-last command is explicit because Qwen's q projection interleaves query and output-gate
channels; model meaning is carried above the executor.

## 3. Execution/data path

`qwen38-payload-v1-final-a.sinf` is memory-mapped and integrity-validated. The tensor-table parser
resolves layer-3 names, shapes, packed NVFP4 bytes, FP8 group scales, and tensor-scale sidecars.
The test creates typed artifact-backed buffer descriptors, copies only the required ranges to
GPU 0, and executes a 20-command Physical Plan. The Python reference reads the same packed
weights and independently evaluates the layer. The final FP32 output is compared elementwise.

## 4. Important shapes and data structures

- hidden input/output: `[5120]` FP32;
- q projection: `[5120 -> 12288]`, split-last into query/gate `[6144] + [6144]`;
- k/v projections: `[5120 -> 1024]`, four KV heads of width 256;
- q/k norms: 256 elements per authored head, with 24 and 4 heads respectively;
- KV cache: BF16 `[context, 4, 256]`, exercised at context position 0;
- attention output: 24 query heads × 256 = `[6144]` before output projection;
- MLP: hidden 5120, intermediate 17408, three NVFP4 projections.

## 5. Core invariants

- Artifact payload offsets are bounded by the validated payload section.
- Every physical operand has an explicit dtype, logical shape, layout, alignment, and encoding.
- FP32-only kernels never reinterpret BF16 storage; conversions/materialization are explicit.
- q/gate interleaving is represented by `split_last` metadata, not inferred from a weight shape.
- KV state is BF16 storage with FP32 compute and persists through the Physical Plan.
- The executor remains model-blind and performs no artifact lookup or allocation in the hot path.

## 6. Performance model

This is correctness evidence, not a performance claim. The layer is dominated by six NVFP4
matrix-vector projections and attention/MLP elementwise work. Artifact reads are bounded host
copies during setup; steady-state execution uses prebound device buffers. The 20 command launches
are intentionally explicit for diagnosis and will only be fused or tuned in S04 after correctness.

## 7. Likely failure modes

- Wrong payload range or sidecar pairing: artifact validation or tensor-table binding fails.
- BF16/FP32 contract mismatch: provider eligibility or Physical Plan validation rejects the plan.
- q/gate layout error: output gate statistics diverge before output projection; inspect ID26 split.
- RoPE/cache axis error: attention output diverges at the cached GQA command.
- Quantization/reference mismatch: compare the external Python output and retained intermediate
  tensors before changing tolerances.

## 8. Exactly three files to read

1. `include/superinfer/artifact/tensor_table.hpp`
2. `tests/gpu/sm120/qwen38_layer_artifact_test.cu`
3. `backends/sm120/runtime/cuda_plan_executor.cuh`

## 9. One hands-on experiment

Run the artifact differential with `CUDA_VISIBLE_DEVICES=0` as documented in the S03-02 progress
checkpoint, then change only the `split_last` metadata from `{outer=24, first=256, second=256}`
to a flat split and predict a large output-gate divergence. The correct interleaved metadata
should reproduce `max_abs≈2e-4`; the flat split should fail the numerical contract.

## 10. Five questions

1. Why must the tensor-table parser preserve both logical shape and physical storage encoding?
2. Which layer operation proves that q/gate layout is semantic rather than inferred by CUDA?
3. What data is stored in the KV cache, and why is its compute dtype different from its storage dtype?
4. Trace one NVFP4 q projection from `.sinf` payload bytes to the CUDA command operands.
5. What symptom would distinguish a bad artifact sidecar binding from a wrong q/gate split?
