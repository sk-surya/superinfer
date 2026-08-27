# S03-02 Qwen operation coverage

This matrix is the compile-time status for the pinned text path. A capability is executable only
when the provider returns a deterministic candidate and the CUDA resolver accepts the same kernel
ID. The payload artifact records the same status in its manifest.

| Semantic operation | Lowered capability | Baseline status | Evidence / remaining boundary |
| --- | --- | --- | --- |
| embedding | `embedding` | executable baseline | BF16 provider ID 8 passes RTX 5090 differential; complete lowered graph resolves the pinned embedding payload; token execution remains S03-03 |
| RMSNorm | `rms_norm` | generic executable baseline; Qwen contract qualified by oracle | FP32 scale ID 5 and BF16 scale ID 12; Qwen `epsilon=1e-6`, `one_plus_weight` is preserved through semantic/lowered/physical metadata and the checkpoint-backed Transformers layer oracle |
| Gated Delta attention | `gated_delta_attention` | generic executable baseline; Qwen composition lowered | Provider ID 15 plus IDs 24/25 derive gates and update causal BF16 convolution state; complete layer-0 artifact differential passes across two segments with state boundary max abs 7.66158e-4 |
| gated grouped full attention | `attention_bf16_cache` | generic executable baseline; Qwen composition lowered | IDs 14, 20-23 and explicit `split_last` ID 26 cover projection-adjacent split, per-head norm, partial RoPE, BF16 cache append, cached GQA, and sigmoid gate; complete layer-3 artifact differential passes with max abs 1.98513e-4 |
| residual | `residual` | executable baseline | Kernel ID 4; CUDA and CPU fixture coverage exists |
| gated dense FFN | `gated_dense_ffn` | executable baseline | Layer-level NVFP4 IDs 13/18 and complete full-graph sidecar lowering resolve all pinned FFN payloads; token differential remains S03-03 |
| LM head | `lm_head` | executable baseline | NVFP4 LM-head lowering uses ID 13 with typed sidecars and the full artifact binder resolves the payload; logits differential remains S03-03 |

The current `Physical Plan` status is `complete-qwen-artifact-graph-compilable-and-bound; token-execution-pending`. `Specializer` queries
the injected `KernelProvider` for every lowered requirement and fails closed at the first missing
candidate; it never substitutes a model-specific or hard-coded kernel.

The checkpoint-backed Transformers layer oracle is available at `tools/qwen38_layer_differential.py`.
It is intentionally separate from the acceptance row above: the target-side physical projection
commands must consume the same typed operands and match this oracle before Qwen promotion.

## Acceptance condition for promotion

Every row must have a deterministic provider candidate, a matching CUDA resolver entry, an
independent reference/differential fixture, and an artifact-to-token trace before the plan can be
marked executable or before S03-03 correctness acceptance begins.
