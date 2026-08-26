# S03-02 Qwen operation coverage

This matrix is the compile-time status for the pinned text path. A capability is executable only
when the provider returns a deterministic candidate and the CUDA resolver accepts the same kernel
ID. The payload artifact records the same status in its manifest.

| Semantic operation | Lowered capability | Baseline status | Evidence / remaining boundary |
| --- | --- | --- | --- |
| embedding | `embedding` | unavailable | Generic FP32 CUDA provider ID 7 and BF16 provider ID 8 pass CPU/CUDA tests; pinned artifact-to-token binding remains open |
| RMSNorm | `rms_norm` | executable baseline | FP32 scale ID 5 and BF16 scale ID 12; semantic-to-physical operand ordering and CUDA differential fixtures pass |
| Gated Delta attention | `gated_delta_attention` | generic executable baseline | Provider ID 15 and CUDA in-place FP32 recurrent state command match split-step oracle fixtures; provider operand-count checks prevent accidental selection for Qwen's richer unresolved contract |
| grouped full attention | `attention` | generic executable baseline | Provider ID 14 and CUDA GQA command cover contiguous FP32 `[position, kv_head, feature]` cache windows; Qwen projection/RoPE/state composition remains open |
| residual | `residual` | executable baseline | Kernel ID 4; CUDA and CPU fixture coverage exists |
| gated dense FFN | `gated_dense_ffn` | unavailable | Pinned gate/up/down weights now bind; generic FP32 provider ID 11, NVFP4 materialization ID 9, and generic NVFP4 linear ID 13 pass, but quantized composition/artifact binding is absent |
| LM head | `lm_head` | unavailable | Weight-connected CPU oracle and generic FP32 projection ID 10 pass; NVFP4 materialization/linear IDs 9/13 exist, but no quantized `lm_head` composition/artifact binding path |

The current `Physical Plan` status is `pending-baseline-provider-coverage`. `Specializer` queries
the injected `KernelProvider` for every lowered requirement and fails closed at the first missing
candidate; it never substitutes a model-specific or hard-coded kernel.

## Acceptance condition for promotion

Every row must have a deterministic provider candidate, a matching CUDA resolver entry, an
independent reference/differential fixture, and an artifact-to-token trace before the plan can be
marked executable or before S03-03 correctness acceptance begins.
