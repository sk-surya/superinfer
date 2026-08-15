# SuperInfer

## Vision

SuperInfer is an ahead-of-time model compiler and deliberately small inference runtime for NVIDIA RTX 5090. A compatible open-weight Hugging Face model is converted into a portable `.sinf` artifact, semantically lowered, physically specialized for the exact model and target GPU, autotuned on the target machine, and executed without generic model logic in the token loop.

The product thesis is **composable at the control plane, ruthlessly specialized at the data plane**.

Project execution follows a second governance thesis: **agents preserve velocity while the user stays no more than one major L2 conceptual gate behind**. `.planning/UNDERSTANDING-GATES.md` defines the protocol; `.planning/UNDERSTANDING.md` records durable user knowledge and unknowns.

## Problem

General inference runtimes carry dynamic graph machinery, broad hardware policies, runtime dispatch, and compatibility overhead into a deployment where the model, shapes, quantization, GPU, and decode strategy are known. Hand-specialized engines can be fast but are difficult to extend or research. SuperInfer separates research composition from execution specialization so new ideas can be tried rapidly without compromising the hot path.

## Initial Audience

- kernel and inference researchers working on consumer Blackwell GPUs;
- model authors who need a reproducible single-GPU deployment artifact;
- performance engineers comparing kernels and decoding strategies under controlled protocols;
- contributors adding model families without editing the physical executor.

## V0 Milestone

V0 ends when all of the following are true:

1. Qwen3.8-27B converts from pinned Hugging Face inputs to a deterministic `.sinf` artifact.
2. The artifact validates, loads, and generates correct tokens on an RTX 5090 through the `sm120` backend.
3. The runtime token path has no heap allocation or model-family branching.
4. At least one specialized kernel portfolio beats the correctness baseline on a declared workload without numerical regression.
5. A clean-machine benchmark recipe produces the first checked, machine-readable RTX 5090 performance graph.
6. Gemma 4 26B-A4B proves the architecture is extensible through the five intended extension surfaces.

## Critical Path

The narrow critical path is:

`HF Qwen3.8-27B -> frontend -> Semantic IR -> Lowered IR -> Physical Plan -> .sinf -> sm120 runtime -> correct generation -> reproducible RTX 5090 graph`

Everything not necessary for that proof is either parallel research or explicitly deferred.

## Non-Goals for V0

- multi-GPU, distributed serving, or datacenter scheduling;
- non-NVIDIA backends or pre-Blackwell optimization;
- a vLLM-compatible server or broad OpenAI API surface;
- training, fine-tuning, or checkpoint authoring beyond format conversion;
- every Hugging Face architecture or arbitrary PyTorch graphs;
- dynamic model selection inside one running executor;
- production multi-tenant isolation;
- treating a research result as a benchmark claim without reproducible evidence.

## Success Measures

- correctness: reference-aligned logits/tokens and repeatable long-context generation;
- architecture: a second model family lands without changing the executor;
- research velocity: a kernel or decode experiment can be expressed, gated, reproduced, and promoted without hand-editing the runtime;
- performance: transparent prefill/decode results with raw data, environment metadata, and comparisons against declared baselines;
- readability: contributors can trace ownership and data transformations from source tensor to launched kernel.
- understanding: the user can own each thesis-critical mechanism through explanation, prediction, execution tracing, and a small hands-on change without gating mechanical work.

## Constraints

- primary target: one RTX 5090, CUDA toolchain capable of `sm_120a` code generation;
- C++20/CUDA for compiler and runtime; Python for model conversion, experiment orchestration, and reporting;
- artifacts and generated plans must be deterministic and versioned;
- performance work never bypasses correctness gates;
- implementation never crosses a second unpassed L2 understanding gate; phase-transition state and required packets are retained as project evidence;
- legal/license metadata from source models is preserved in `.sinf` manifests.
