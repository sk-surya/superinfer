# S03-02 Qwen operation coverage

This matrix is the compile-time status for the pinned text path. A capability is executable only
when the provider returns a deterministic candidate and the CUDA resolver accepts the same kernel
ID. The payload artifact records the same status in its manifest.

| Semantic operation | Lowered capability | Baseline status | Evidence / remaining boundary |
| --- | --- | --- | --- |
| embedding | `embedding` | unavailable | Generic FP32 CUDA provider ID 7 and BF16 provider ID 8 pass CPU/CUDA tests; pinned artifact-to-token binding remains open |
| RMSNorm | `rms_norm` | executable baseline | Kernel ID 5; differential/reference fixture exists |
| Gated Delta attention | `gated_delta_attention` | reference-only | Independent recurrent CPU oracle exists; CUDA provider/state binding absent |
| grouped full attention | `attention` | unavailable | Pinned q/k/v/o and q/k norm weights now bind in Semantic IR; KV/RoPE provider path absent |
| residual | `residual` | executable baseline | Kernel ID 4; CUDA and CPU fixture coverage exists |
| gated dense FFN | `gated_dense_ffn` | unavailable | Pinned gate/up/down weights now bind and the independent CPU graph oracle composes the primitive; NVFP4 decode/provider absent |
| LM head | `lm_head` | unavailable | Weight-connected independent CPU graph oracle now exists; no NVFP4 artifact/provider path |

The current `Physical Plan` status is `pending-baseline-provider-coverage`. `Specializer` queries
the injected `KernelProvider` for every lowered requirement and fails closed at the first missing
candidate; it never substitutes a model-specific or hard-coded kernel.

## Acceptance condition for promotion

Every row must have a deterministic provider candidate, a matching CUDA resolver entry, an
independent reference/differential fixture, and an artifact-to-token trace before the plan can be
marked executable or before S03-03 correctness acceptance begins.
