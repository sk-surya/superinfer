> **SUPERSEDED by `S04-P8R-CONTRACT.md`.** The classification-D conclusion below rested on a malformed
> PTX probe (missing `.ue4m3` scale type and `{byte-id,thread-id}` operands). `sm_120a` **does** expose
> warp-level block-scaled NVFP4 `mma.sync` (`m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3`),
> proven to assemble and execute. Preserved as history; do not cite as current hardware fact.

# S04-P8 Summary — Native NVFP4 Tensor-Core Feasibility Spike

**Classification: D — native block-scaled NVFP4 MMA is not available on `sm_120a`.**
(Plus a quality flag that would apply even if it were: activation quantization to E2M1 is a
first-order model perturbation.) No production promotion. This is a spike result, not a commit of a
tensor-core path.

## P8-A — hardware contract (CUDA 13.1.115, RTX 5090 `sm_120a`)

Probed directly with `ptxas -arch=sm_120a`:

- **Native fp4 MMA exists, but not block-scaled.** `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32`
  assembles with A = 4×`.b32`, B = 2×`.b32`, C = 4×`.f32` (E2M1×E2M1, FP32 accumulator).
- **Block-scale is tcgen05-only.** `.block_scale` is rejected for `mma.sync` ("Illegal modifier
  '.block_scale'"); the only block-scaled NVFP4 instruction in CUDA 13.1 is
  `tcgen05.mma...kind::mxf8f6f4.block_scale.scale_vec::N`, whose CCCL guard states it is
  "only supported on SM_100a_103a_110a". `tcgen05.alloc` is rejected on `sm_120a`.
- Scale vector size 16 / UE4M3 / FP32 accumulate are therefore **available only on sm_100a-class
  parts**, not the RTX 5090 target.
- CUTLASS is not installed on the host; the reference path was the PTX contract itself. No CUTLASS
  dependency was added.

**Consequence:** the per-16-block UE4M3 weight scales that define NVFP4 cannot be applied by any MMA
available on `sm_120a`. The `mma.sync` k32 tile sums two 16-value blocks together, so even a
software-applied scale cannot separate them inside the reduction.

## P8-B — activation quantization error (real Qwen hidden vectors, block-16 E2M1 + UE4M3)

`tools/qwen38_nvfp4_activation_quant_error.py`; deterministic `amax/6` rule (also `amax/7`, `amax/4`).

| capture | max_abs | mean_abs | RMSE | relative L2 | cosine | clipped |
|---|---:|---:|---:|---:|---:|---:|
| chat hidden (step 29) | 0.9375 | 0.1315 | 0.1802 | **9.29%** | 0.99567 | 0 |
| chat hidden (reference) | 1.0568 | 0.1310 | 0.1802 | 9.30% | 0.99567 | 0 |
| layer-3 input (step 28) | 0.1845 | 0.0061 | 0.0097 | 2.44% | 0.99970 | 0 |

A ~2–9% relative-L2 per-activation error is a **different class** from the accepted P6
reduction-order perturbation (~1e-7); it would require its own model-level quality justification.

## P8-C/D — three arms and the shape matrix

Arms A (effective N=1 padded native MMA), B (N=8 utilization upper bound) and C (minimal warp-level
MMA) are **moot for the production contract**: the native instruction that would run them cannot
represent per-16 UE4M3 block scales on `sm_120a`. Benchmarking them on the non-block-scaled
`kind::f8f6f4` path would measure a different algorithm (global-scale fp4), not SuperInfer's NVFP4
projection semantics, so it cannot answer the production feasibility question. No shape matrix was
therefore run.

## Model quality (P8-E)

Not run: there is no candidate to gate. The activation-quantization error above is recorded as the
quality cost a future native path would have to justify.

## Fixture-debt status

**Fixed.** The layer/GDN C++ differentials previously SKIPped because their reference captures were
unwired and `tools/qwen38_nvfp4_gdn_reference.py` still used the pre-5.16 Transformers GDN API. Both
are repaired and a repeatable driver exists:
`python tools/run_qwen38_layer_gdn_fixtures.py`.
Both now execute and pass:
- **layer-3:** `max_abs=0.00108337`, `mean_abs=3.55456e-05` (thresholds `0.02` / `0.0002`), `attention_max_abs=0`.
- **GDN layer-0 (2 segments):** `max_abs=0.000312805`, `mean_abs=3.04788e-06`, with `qkv/conv/core/gated`
  all `0`.

Two root causes fixed: (1) `tools/qwen38_nvfp4_gdn_reference.py` still used the pre-5.16 Transformers
GDN API (`layer.linear_attn.causal_conv1d_update`, dict-valued `recurrent_states`/`conv_states`), and
(2) the C++ GDN test derives its `.attn`/`.state` companions by inserting before a `.bin` suffix, so the
reference must be named `*.bin` — the driver now does both.

## Next lane (from the P7 fresh profile)

NVFP4 is 68.6% and now at its software-decode limit (P7 proved decode/scale/coalescing are not the
binding cost, and tensor cores are unavailable). The remaining decode-side item is `linear_f32`
(15.3%); the larger user-visible item is the **~25–30 s startup/artifact-materialization** path, which
currently dominates time-to-first-token. Do not resume NVFP4 micro-tuning without new evidence.
