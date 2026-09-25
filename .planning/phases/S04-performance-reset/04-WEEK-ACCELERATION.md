# Week Acceleration — From E0 to Competitive Decode

## Entry condition

E0b is integrated and either:

- <=40 ms/token; or
- the bounded diagnosis found an immediate fix and it is being executed.

Do not enter this plan with a failed E0b and no causal explanation.

## Objective

Reach <=15 ms/token / >=67 tok/s ordinary decode within the week, then pursue <=12.5 ms/token / >=80 tok/s.

## Order of work

The exact residual ms/token profile after E0b controls local ordering, but the default sequence is below because these are known deficiencies.

### W1 — mature GDN recurrence + convolution/state layout

Current recurrence and convolution together are several ms/token and use weak launch/state organization.

Study/port:

- SparkInfer kernels/csrc/cuda/fused/qwen36.cu
- NInfer GDN execution/state code
- FLA recurrent gated-delta design as semantic reference where necessary

Goals:

- state traversed minimally;
- coalesced/transposed state layout where donor requires it;
- convolution + nearby pointwise work fused where proven;
- FP32 recurrent state preserved where numerically justified;
- no redundant state materialization.

Do not change state precision merely for speed without a recipe-quality decision.

### W2 — real GPU token feedback and sampling

The current E2E test:

- uploads predetermined continuation tokens;
- synchronizes;
- downloads the full vocabulary logits;
- performs CPU argmax.

That is a correctness harness, not the final decode loop.

Implement:

- LM-head reduction/top-k or argmax epilogue;
- device token scalar;
- device position/KV-length scalar as needed;
- asynchronous host-visible ring/result buffer;
- EOS/done state;
- no full-vocabulary readback per token.

Study:

- SparkInfer fused/sample_ops.cu
- NInfer decode program/control flow

Keep the old harness for diagnostics.

### W3 — CUDA graph capture/replay

After kernels/fusion are fast enough, launch topology becomes material.

Study:

- NInfer src/core/decode_graph.cpp
- SparkInfer graph replay path

Requirements:

- stable preallocated pointers;
- no hot-path getenv/device-property query;
- no heap allocation;
- graph captures the stable decode command set;
- position/token state changes without rebuilding the model graph.

Do not build a megakernel just to reduce launches before measuring the graph result.

### W4 — mature attention + compressed cache path

Current short-context attention is not today's largest bottleneck but is not a credible 4K–16K solution.

Use mature split-KV/flash-decode design rather than extending the current hand kernel.

Candidates:

- SparkInfer kernels/csrc/cuda/attention/flash_decode_split.cu
- FlashInfer decode attention if integration is clean

KV representation follows the selected model recipe and quality contract. Do not silently change BF16/FP8/INT8 cache semantics.

### W5 — long-context decode qualification

Do not build chunked prefill just to run the week benchmark.

Current frontend unrolls positions, so large semantic prefill plans are impractical.

Temporary measurement method:

- initialize long-context state by decode/forced-token replay;
- measure steady-state decode after the target context is reached;
- report fill method explicitly.

One 4K and one 16K point are enough for the week.

Chunked prefill becomes the next separate production milestone after ordinary decode parity.

## One-week scorecard

Required:

| Metric | Gate |
|---|---:|
| Ordinary short-context decode | >=67 tok/s |
| Relative to fastest same-machine SparkInfer | >=70% |
| Device time | <=15 ms/token |
| Stretch | <=12.5 ms/token / >=80 tok/s |
| Full-vocab CPU readback | zero in production loop |
| Per-token cudaDeviceSynchronize | zero in production loop |
| D-021 | pass |
| Fresh-process reproducibility | pass |
| Whole-model nsys | captured |

## After >=60–70 tok/s

Only then unlock:

- W4A4 repair/qualification for prefill/verify;
- small-T crossover study;
- MTP/DSpark;
- P2P/TP2 investigation;
- chunked prefill;
- compiler retarget test.

## Small-T crossover experiment

This is the only planned research experiment after ordinary performance becomes healthy.

Benchmark T = 1, 2, 4, 8, 16:

- optimized W4A16 donor path;
- native SM120 W4A4 MMA path.

Measure full projection census with realistic weight streaming.

Purpose:

Determine where native MMA becomes superior.

If T=4 W4A16 is still around 1.1–1.3x T=1, do not invent an N=8 speculative architecture merely to occupy Tensor Core columns.

If native MMA wins materially at small T, use the result to design verification.

## Compiler-thesis falsification gate

Speed parity is not enough to validate SuperInfer.

After the engine reaches parity, choose a second target with a known external frontier.

First target should hold the graph fixed and change precision/layout, for example:

- uniform NVFP4 Qwen profile -> upstream mixed NVFP4/FP8 Qwen profile.

Measure:

- engineer-days;
- generated vs hand-written specialization;
- performance as % of frontier;
- amount of target-specific code.

Pass:

- >=90% of frontier;
- <=1/3 the engineer-days of a manual hand port.

If this fails, "specialization compiler" is not yet a moat.

Only after passing this gate move to a graph-level retarget such as a hybrid MoE.
