# Donor Map — Reuse Before Reinvention

## Policy

The objective is not to vendor random external engines. It is to use mature source as executable design evidence and, where licenses permit, transplant/wrap the best components behind SuperInfer's existing specialization surfaces.

Preserve copyright/license notices for copied code and check dependency-local licenses before vendoring.

## NInfer — primary C++/CUDA decode donor

Repository: Neroued/ninfer  
License: Apache-2.0 at repository level; verify file/dependency notices.

### NVFP4 streaming GEMV

- src/ops/linear/nvfp4/nvfp4_gemv.cuh
- src/ops/linear/nvfp4/nvfp4_config.h
- src/ops/linear/nvfp4/shapes/

Take:

- Geometry/Schedule split;
- multi-row-per-warp mapping;
- accumulator chains;
- vector packed-code loads;
- staged/raw scale strategy;
- exact prepared scale layout;
- BF16-input / FP32-accumulate arithmetic;
- launch bounds.

Do not copy blindly:

- NInfer tensor ownership;
- server abstractions;
- unrelated quant formats.

### Fused FFN

- src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
- src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu

Take:

- interleaved gate/up pairing;
- SwiGLU epilogue;
- residual-add epilogue;
- shape-specific scheduling.

### GDN control

- src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu

Take:

- concurrent norm + A/B dot products;
- algebra for applying inverse norm after control dots;
- direct normalized-h write;
- head-parallel organization.

### Attention/GDN input projections

- src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
- src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu

Take:

- direct output routing epilogues;
- model-specific exact shape schedules.

### Graph/control

- src/core/decode_graph.cpp
- src/core/pdl.cuh
- small_t operator variants

Use after E0, not before.

## SparkInfer — primary same-lineage frontier comparator and fused hybrid donor

Repository: gittensor-ai-lab/sparkinfer  
Verify exact file licenses when copying.

SuperInfer's pinned derivative lineage is the same gittensor-model-hub Qwen3.8 NVFP4 RTX5090 family, making SparkInfer the primary performance comparator.

### GDN/recurrent path

- kernels/csrc/cuda/fused/qwen36.cu

Study:

- convolution/SiLU fusion;
- Q/K normalization;
- recurrent state organization;
- warp-level recurrence;
- state transpose/layout.

### Attention

- kernels/csrc/cuda/attention/flash_decode_split.cu

Study:

- split-KV online softmax;
- GQA K/V sharing;
- partial combine;
- compressed KV modes.

### Sampling

- kernels/csrc/cuda/fused/sample_ops.cu

Take after E0:

- GPU argmax/top-k/sampling pattern;
- device feedback.

### Evaluation infrastructure

- eval/pr_modelopt_bot.py
- eval/pr_qwen38_bot.py
- related pinned 5090 evaluation scripts

Learn from:

- same-box scoring;
- automatic evidence;
- benchmark table refresh.

Do not spend the sprint rebuilding their bot infrastructure.

## FlashInfer — independent kernel benchmark / dependency candidate

Relevant:

- SM12x BF16×FP4 GEMV CuTe DSL kernel;
- FP4 preparation/packing routines;
- split-KV attention;
- GDN support;
- sampling primitives.

Use:

- as independent comparator for E0a;
- as direct dependency where integration is simpler/stronger than a custom port.

Do not integrate both FlashInfer and NInfer for the same role unless one is demonstrably needed as fallback.

## CUTLASS / CuTe — multi-token SM120 foundation

Use for:

- native block-scaled GEMM;
- prefill;
- larger small-T verification;
- fragment/layout reference.

Do not hand-write another SM120 MMA mainloop while an adequate CUTLASS/CuTe composition exists.

## Mature engine ideas, not immediate dependencies

vLLM / SGLang:
- graph buckets;
- device input buffers;
- custom P2P all-reduce;
- serving/scheduler ideas later.

llama.cpp:
- W4A8/W4A16 decode ideas;
- KLD/perplexity qualification methodology;
- simple external baseline where useful.

Marlin:
- W4A16/HMMA small-M verification regime.

qwentin / specialist persistent projects:
- persistent scheduling;
- recurrent speculative commit;
- N-column verification concepts.
Study only after ordinary decode is healthy.

## Import rule

For every donor-derived component, record:

    source repository
    source commit
    source path
    license
    modifications
    numerical recipe
    exact supported shapes
    benchmark against donor/reference
    fallback path

This can live in the implementation summary; no new bureaucracy is required before coding.
