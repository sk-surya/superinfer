# SuperInfer

## Vision

SuperInfer is an ahead-of-time model compiler and deliberately small inference runtime for NVIDIA RTX 5090. A compatible open-weight Hugging Face model is converted into a `.sinf` artifact, semantically lowered, physically specialized for the exact model and target hardware, optionally autotuned on the target machine, and executed without generic model logic in the token loop.

The product thesis is **composable at the control plane, ruthlessly specialized at the data plane**.

The first proof is intentionally single-device Qwen3.8-27B. The second flagship architecture proof is text-only Flash-Next on 2x RTX 5090, because it exercises multi-device placement, heterogeneous PLE/N-gram residency, sparse attention, MoE, gated residual execution, and stateful Gated DeltaNet without changing the core thesis.

Project execution follows a second governance thesis: agents preserve velocity while understanding packets remain durable checkpoints. `.planning/UNDERSTANDING-GATES.md` defines the protocol; `.planning/UNDERSTANDING.md` records durable user knowledge and unknowns. D-014 permits autonomous execution past unpassed gates while retaining those packets for study.

## Problem

General inference runtimes carry dynamic graph machinery, broad hardware policies, runtime dispatch, and compatibility overhead into deployments where the model, shapes, quantization, hardware and decode strategy are mostly known. Hand-specialized engines can be fast but are difficult to extend or research. SuperInfer separates research composition from execution specialization so new ideas can be tried rapidly without compromising the hot path.

Flash-Next adds a second problem the architecture must solve cleanly: not every useful model tensor belongs permanently in one GPU arena. Some state belongs on a specific device, PLE/N-gram tables are naturally host-resident, and dual-GPU execution needs explicit placement and transfer schedules. Those facts belong in compilation/storage/physical planning rather than ad hoc runtime offload logic.

## Initial Audience

- kernel and inference researchers working on consumer Blackwell GPUs;
- model authors who need reproducible `.sinf` deployment artifacts;
- performance engineers comparing kernels, residency policies and decoding strategies under controlled protocols;
- contributors adding model families or unusual storage/execution mechanisms without introducing model-name runtime branches.

## V0 Milestone

V0 ends when all of the following are true:

1. Qwen3.8-27B converts from pinned Hugging Face inputs to a deterministic `.sinf` artifact.
2. That artifact validates, loads, and generates correct tokens on an RTX 5090 through the `sm120` backend.
3. Text-only Flash-Next produces reference-equivalent greedy generation on 2x RTX 5090 through a deterministic multi-device Physical Plan with PLE/N-gram tables host-resident.
4. The runtime token path has no heap allocation, model-family branching, implicit checkpoint parsing, or unplanned device transfers.
5. At least one specialized kernel portfolio beats its correctness baseline on a declared workload without numerical regression.
6. A clean-machine benchmark recipe produces the first checked, machine-readable RTX 5090 performance graph.
7. Gemma 4 26B-A4B later audits model-family extensibility through the intended extension surfaces and challenges accidental coupling introduced by Qwen/Flash-Next work.

## Critical Path

The critical path is now two correctness proofs followed by optimization:

```text
Qwen3.8-27B
  -> frontend -> Semantic IR -> Lowered IR -> Physical Plan -> .sinf
  -> single-5090 correct generation

then

Flash-Next text path
  -> exact capacity proof
  -> multi-device placement + host-resident PLE + MoE + QSA + gated residual
  -> dual-5090 correct generation

then

kernel portfolio -> autoresearch -> reproducible performance proof
```

S03F-01 capacity/model-contract research may run in parallel with the tail of S03. Runtime changes for S03F remain blocked until S03's existing acceptance closes.

## Non-Goals for V0

- tensor parallelism or general distributed serving;
- arbitrary model tensor CPU offload or silent expert-weight paging;
- Flash-Next vision or MTP/speculative support;
- non-NVIDIA backends or pre-Blackwell optimization;
- a vLLM-compatible server or broad OpenAI API surface;
- training, fine-tuning, or checkpoint authoring beyond format/quantization conversion needed for `.sinf`;
- every Hugging Face architecture or arbitrary PyTorch graphs;
- dynamic model selection inside one running executor;
- production multi-tenant isolation;
- treating a research result as a benchmark claim without reproducible evidence.

## Success Measures

- correctness: reference-aligned intermediates/logits/tokens and repeatable state continuation;
- architecture: unusual Flash-Next requirements fit through generic IR/storage/placement contracts, and later Gemma model-family support does not require model-name runtime branching;
- residency: placement decisions and transfer commands are explicit, auditable and derived from packed artifact bytes rather than hidden runtime heuristics;
- research velocity: a kernel, storage or decode experiment can be expressed, gated, reproduced and promoted without hand-editing the runtime;
- performance: transparent prefill/decode results with raw data, environment metadata and comparisons against declared baselines;
- readability: contributors can trace ownership and data transformations from source tensor or host-resident table to launched kernel/transfer command;
- understanding: thesis-critical mechanisms remain captured in small evidence-based packets even when autonomous execution is prioritized.

## Constraints

- first hardware target: RTX 5090 / `sm_120a`; Qwen proof uses one card, Flash-Next architecture proof uses two cards with contiguous pipeline/layer placement;
- C++20/CUDA for compiler and runtime; Python for model conversion, research orchestration and reporting;
- artifacts and generated plans must be deterministic and versioned;
- performance work never bypasses correctness gates;
- PLE host residency is a first-class StoragePolicy/operator path, not generic CPU offload;
- Flash-Next expert residency policy is decided from S03F-01 exact capacity/quality evidence; no silent expert paging is allowed;
- legal/license/provenance metadata from source models is preserved in `.sinf` manifests.
