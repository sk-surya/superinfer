# SuperInfer Agent Guide

This repository builds a single-GPU inference compiler and runtime specialized for NVIDIA RTX 5090 (`sm_120a`). The first public proof is correct Qwen3.8-27B inference from a `.sinf` artifact plus a reproducible performance graph captured on an RTX 5090.

## Read First

Before implementation, read:

1. `.planning/PROJECT.md`
2. `.planning/ARCHITECTURE.md`
3. `.planning/REQUIREMENTS.md`
4. the active phase `CONTEXT.md` and `PLAN.md`
5. `.planning/QUALITY.md` and `.planning/BENCHMARKS.md` when changing numerics or performance

Decisions in `.planning/DECISIONS.md` are binding until replaced by a recorded decision.

## Architectural Invariants

- Preserve the pipeline: semantic composition -> compile-time specialization -> minimal physical runtime.
- Preserve exactly three representations: Semantic IR, Lowered IR, and Physical Plan.
- Extend the system only through `ModelFrontend`, `GraphPass`, `KernelProvider`, `DecodeStrategy`, or `StoragePolicy`. A new extension point requires an architecture decision.
- Runtime execution consumes a validated Physical Plan. It must not inspect model families, choose graph rewrites, or allocate in the token hot path.
- Model-specific behavior belongs in frontends and compile-time passes, never in the physical executor.
- DSpark is a speculative decoding technique and belongs behind `DecodeStrategy`; it is not an attention provider.
- `.sinf` is a versioned, inspectable, checksummed artifact. Unsupported versions or capabilities fail closed with useful diagnostics.

## Languages and Style

- Runtime/compiler: modern C++20 and CUDA C++ targeting `sm_120a` first.
- Conversion, benchmarking, and autoresearch orchestration: typed Python 3.12+.
- Prefer explicit names, small cohesive modules, value types, RAII, and deterministic ownership.
- Public interfaces require API comments describing lifetime, ownership, thread safety, error behavior, and invariants.
- Avoid clever template machinery unless measured evidence shows it improves the hot path.
- No exceptions across CUDA/runtime boundaries. Use a typed status/result mechanism and attach context at layer boundaries.
- No raw owning pointers. Device/host buffers have one visible owner and non-owning views are explicitly named.

## Correctness and Performance Rules

- Correctness gates precede performance claims. Every optimized kernel retains a trusted reference path and differential tests.
- Never loosen tolerances merely to land an optimization. Explain error growth and record the numerical contract.
- No hot-path heap allocation, model-family branching, filesystem access, logging, implicit synchronization, or JIT compilation.
- Keep prefill and decode measurement separate. Report time-to-first-token, inter-token latency, throughput, memory, power policy, and exact workload.
- Benchmark changes require machine-readable raw results and environment manifests; screenshots alone are not evidence.
- An autoresearch candidate may be promoted only after build, unit, differential, sanitizer where applicable, deterministic replay, and benchmark gates pass.

## Tests and Evidence

- Put unit tests beside the owning module or in the corresponding `tests/` subtree.
- Every parser and serialized structure needs golden, round-trip, truncation, corruption, version-skew, and fuzz/property coverage.
- Every kernel needs shape coverage, boundary coverage, randomized differential tests, and representative model-shape tests.
- Every requirement referenced by a plan must name the evidence produced to prove it.
- CI must be useful without a GPU; GPU correctness and benchmark lanes are additive and clearly labeled.

## Ownership and Change Discipline

- Each plan declares the files it owns. Do not modify files owned by another active plan without coordinating the dependency.
- Keep commits atomic and scoped to one plan outcome. Do not mix formatting sweeps with functional work.
- Preserve user changes and unrelated worktree edits. Do not rewrite history or use destructive Git operations.
- When deviating from a plan, record the reason, affected requirements, verification performed, and follow-up in the phase summary or `.planning/DECISIONS.md`.
- Generated code and generated `.sinf` fixtures must be reproducible from checked-in inputs; do not hand-edit them.

## Definition of Done

A plan is done only when its required artifacts exist, all named verification commands pass, evidence is stored in the documented location, requirement coverage is updated, and no blocker is hidden behind a TODO.
