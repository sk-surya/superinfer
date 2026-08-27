# S03-02 Qwen operation coverage

This matrix is the compile-time status for the pinned text path. A capability is executable only
when the provider returns a deterministic candidate and the CUDA resolver accepts the same kernel
ID. The payload artifact records the same status in its manifest.

| Semantic operation | Lowered capability | Baseline status | Evidence / remaining boundary |
| --- | --- | --- | --- |
| embedding | `embedding` | unavailable | Generic FP32 CUDA provider ID 7 and BF16 provider ID 8 pass CPU/CUDA tests; pinned artifact-to-token binding remains open |
| RMSNorm | `rms_norm` | generic executable baseline; Qwen contract qualified by oracle | FP32 scale ID 5 and BF16 scale ID 12; Qwen `epsilon=1e-6`, `one_plus_weight` is preserved through semantic/lowered/physical metadata and the checkpoint-backed Transformers layer oracle |
| Gated Delta attention | `gated_delta_attention` | generic executable baseline; Qwen composition lowered | Provider ID 15 and CUDA in-place FP32 recurrent state command match split-step oracle fixtures; generic IDs 24/25 derive gates and update causal BF16 convolution state; artifact-backed Qwen layer differential remains open |
| gated grouped full attention | `attention_bf16_cache` | generic executable baseline; Qwen composition lowered | IDs 14, 20-23 cover projection-adjacent split, per-head norm, partial RoPE, BF16 cache append, cached GQA, and sigmoid gate; artifact-backed Qwen layer differential remains open |
| residual | `residual` | executable baseline | Kernel ID 4; CUDA and CPU fixture coverage exists |
| gated dense FFN | `gated_dense_ffn` | unavailable | Pinned gate/up/down weights now bind; generic FP32 provider ID 11, NVFP4 materialization ID 9, and generic NVFP4 linear ID 13 pass, but quantized composition/artifact binding is absent |
| LM head | `lm_head` | unavailable | Weight-connected CPU oracle and generic FP32 projection ID 10 pass; NVFP4 materialization/linear IDs 9/13 exist, but no quantized `lm_head` composition/artifact binding path |

The current `Physical Plan` status is `synthetic-qwen-graph-compilable; artifact-materialization-pending`. `Specializer` queries
the injected `KernelProvider` for every lowered requirement and fails closed at the first missing
candidate; it never substitutes a model-specific or hard-coded kernel.

The checkpoint-backed Transformers layer oracle is available at `tools/qwen38_layer_differential.py`.
It is intentionally separate from the acceptance row above: the target-side physical projection
commands must consume the same typed operands and match this oracle before Qwen promotion.

## Acceptance condition for promotion

Every row must have a deterministic provider candidate, a matching CUDA resolver entry, an
independent reference/differential fixture, and an artifact-to-token trace before the plan can be
marked executable or before S03-03 correctness acceptance begins.
