# Research Agenda and Technical Findings

This document distinguishes approved design direction from facts that must be measured or pinned during implementation. It is not a substitute for phase-specific experiments.

## High-Confidence Design Findings

1. **Specialize known facts early.** Model dimensions, layer topology, quantization, KV layout, target capability, and decode strategy should be compiler inputs, not token-loop decisions.
2. **Separate semantic correctness from physical optimization.** A canonical Semantic IR plus a reference oracle enables aggressive lowering without coupling model parsing to kernels.
3. **Keep a portfolio, not a megakernel monoculture.** Prefill/decode, short/long contexts, dense/MoE, and memory/compute-bound regions have different optima. A capability-selected portfolio allows measured specialization while preserving fallbacks.
4. **Artifact validation is part of runtime safety.** Offsets, sizes, alignment, integer overflow, capability IDs, and dependency graphs must be validated before device allocation or launch.
5. **Reproducible performance requires environment control.** RTX 5090 results are sensitive to thermals, power/clocks, driver/toolchain, background load, prompt shapes, and sampling semantics. The benchmark system must capture and reject invalid runs.
6. **Automated optimization needs stronger correctness gates than manual tuning.** Search will discover undefined behavior and benchmark loopholes unless schemas, oracles, and promotion rules fail closed.
7. **Speculative decoding is stateful decode policy.** Proposal, verification, acceptance, rollback, and KV consistency belong above attention kernel choice. DSpark therefore belongs behind `DecodeStrategy`.

## Questions to Resolve by Phase

### S00/S01 — Tooling and Schema

- Which build/package combination provides reproducible C++20/CUDA and Python developer setup without obscuring compiler flags?
- Which stable schema encoding best balances mmap-friendly binary data, CPU fuzzability, forwards compatibility, and zero-copy views?
- What exact CUDA toolkit/driver versions are required for the desired `sm_120a` features?

Decision method: prototype only enough to compare compile time, inspectability, binary stability, and fuzz ergonomics. Record the selection in `DECISIONS.md`.

### S02 — Backend Mechanics

- Which CUDA features and library components are usable on the target toolchain and redistributable under project goals?
- What persistent-weight/KV/workspace layout fits Qwen3.8-27B under realistic context budgets on a 5090?
- Which operations need a C++ CPU oracle versus a Python/PyTorch reference bridge?

Decision method: target probe plus memory ledger and tiny differential graphs before full model integration.

### S03 — Qwen3.8 Semantics

- Pin the exact upstream repository/revision and confirm architecture fields, tensor names/shapes, MoE parameters, tokenizer/template behavior, RoPE variant, attention/local-attention policy, and optional auxiliary heads.
- Determine the feasible weight/KV quantization necessary to fit the intended context/workload.
- Establish reference logits/tokens at short and long contexts with fixed seeds.

Decision method: immutable source manifest and generated tensor/config fixtures. Never infer semantics from the marketing name alone.

### S04 — Kernel Portfolio

Candidate research axes:

- GEMM/GEMV: datatype, tiling, warp specialization, persistent scheduling, epilogue fusion, weight layout;
- attention: prefill vs decode split, GQA head grouping, KV layout, online softmax, split-K/reduction, local attention;
- MoE: router/top-k fusion, expert ordering, token permutation, grouped GEMM/GEMV, low-token expert imbalance;
- fusion: residual/norm, RoPE/QKV, activation/gate, KV update/attention, sampling;
- memory: alignment, prefetch/double buffering, shared-memory budget, register pressure, cache residency.

Every axis needs a hypothesis, applicability envelope, numerical contract, and counterfactual baseline.

### S05 — Autoresearch and DSpark

- Choose search granularity: compile constants, layouts, kernel variants, fusion boundaries, and schedule parameters; do not allow arbitrary benchmark/test edits.
- Define promotion statistics: minimum samples, steady-state detection, robust center/spread, practical speedup threshold, and repeat confirmation.
- Reconstruct DSpark from primary material or pinned implementation before specifying proposal/verification details. Treat all current DSpark mechanics beyond its classification as unverified.

### S06 — Competitive Benchmarking

- Pin comparison engines, commits, flags, quantization, tokenizer, prompts, decode sampling, batch/concurrency, and output length.
- Decide which claims are single-request latency versus aggregate throughput; never blend them.
- Capture power policy and thermal validity; decide whether fixed clocks or stock behavior is the public default.

## Candidate Kernel Experiment Template

Each candidate record should include:

- hypothesis and expected bottleneck;
- eligible operation/shapes/layouts/dtypes;
- source/reference implementation;
- tunable domain and bounded budget;
- numerical tolerance justified by dtype/accumulation;
- correctness corpus and sanitizer requirements;
- benchmark manifest and baseline;
- accept/reject rule and rollback path.

## Model Support Research Contract

Before implementing any frontend:

1. pin repository and immutable revision;
2. capture config/tokenizer/chat-template files and their hashes;
3. enumerate every tensor with name, shape, dtype, byte size, and role;
4. derive semantic graph and state transitions from source/reference code;
5. create tiny synthetic fixtures that exercise each semantic variant;
6. capture trusted logits/tokens and tolerance rationale;
7. record license and redistribution constraints.

## Research Guardrails

- Do not copy an external kernel without preserving license/provenance and writing an independent correctness test.
- Do not optimize only one prompt and claim model-wide speedup.
- Do not compare different quantization, context, concurrency, sampling, speculative settings, or output lengths without labeling the difference.
- Do not make hard assertions about unreleased/rapidly changing model families until immutable upstream sources are pinned.
- Do not expand model coverage before the S06 proof unless it directly unblocks the critical path.

## Expected Research Outputs

Research outputs live as machine-readable experiment/evidence bundles plus concise Markdown conclusions. Every conclusion links to raw inputs, results, source revision, and promotion decision. Negative results are retained when they close a meaningful branch of the search space.
