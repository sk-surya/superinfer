# Architecture

## Thesis

SuperInfer is **composable at the control plane and ruthlessly specialized at the data plane**. Researchers compose semantic model structure, passes, providers, storage, and decoding policies through stable interfaces. The compiler resolves those choices into an immutable Physical Plan. The runtime validates and executes that plan with minimal policy.

```text
Hugging Face model + pinned revision
          |
          v
    ModelFrontend -----------------------------+
          |                                    |
          v                                    |
     Semantic IR <---- semantic GraphPasses    | control plane
          |                                    |
          v                                    |
      Lowered IR <---- target/layout/fusion ---+
          |
          v
    Physical Plan <---- KernelProvider + DecodeStrategy + StoragePolicy
          |
          +---------> .sinf artifact
          |
          v
  validated sm120 executor                     data plane
          |
          v
       tokens + evidence
```

## The Three Representations

### 1. Semantic IR

Purpose: represent model meaning independent of CUDA and byte layout.

Contains typed tensors, symbolic/static dimensions, dtypes/quantization intent, model operations, state edges such as KV cache, and named entry points for prefill/decode/optional auxiliary heads. Operations include embedding, RMS/layer normalization, RoPE, MHA/GQA/local attention semantics, dense gated FFN, MoE routing/experts, residual composition, LM head, and sampling inputs.

It must not contain CUDA grid/block sizes, function pointers, file offsets, device addresses, or provider-specific opaque blobs.

### 2. Lowered IR

Purpose: make target-aware decisions while preserving inspectability and verifier-friendly structure.

Contains physical tensor layouts, padded dimensions, quantized storage types/scales, fused regions, workspace requirements, KV layout, memory spaces, scheduling dependencies, candidate kernel capability requirements, and explicit prefill/decode graphs. Lowered IR dumps are deterministic inputs to autotuning and bug reports.

### 3. Physical Plan

Purpose: be the immutable, validated execution contract.

Contains ordered command streams/DAGs, resolved kernel handles or stable kernel IDs, launch parameters, buffer arena offsets, stream/event dependencies, constants, decode-state layout, selected strategy, capability fingerprints, and resource bounds. It contains no unresolved model semantics. Runtime validation checks all indexes, offsets, sizes, capability fingerprints, dependencies, and workspace bounds before allocating/executing.

Temporary analyses—dominance, liveness, shape maps, cost models—are pass-local data, not additional IRs.

## Five Extension Surfaces

### `ModelFrontend`

Owns source config/tensor/tokenizer validation and emission of canonical Semantic IR. It never selects CUDA kernels. Frontends are pure/deterministic for pinned inputs and emit actionable schema diagnostics.

### `GraphPass`

Owns verified transformations over Semantic or Lowered IR. Every pass declares input representation, preconditions, preserved/invalidated analyses, deterministic configuration, and postconditions. Pass ordering is explicit; pass pipelines are serialized in compilation provenance.

### `KernelProvider`

Owns target implementations and capability descriptions. Providers enumerate candidates based on operation/layout/dtype/shape/target constraints. Selection is capability- and measurement-driven, never a model-name switch. A correct baseline provider remains available.

### `DecodeStrategy`

Owns token-level policy: greedy, sampling, multi-token/speculative proposal, verification, acceptance, rollback, and KV state transitions. It declares required graphs, buffers, and provider capabilities during compilation. DSpark lives here.

### `StoragePolicy`

Owns artifact tensor layout, alignment, compression/quantization payload conventions, host mapping, staging, and device materialization. Kernels see typed device views, not file-format details.

## Compiler Pipeline

1. Pin and validate source identity and licenses.
2. Frontend parses config/tensor inventory/tokenizer and emits Semantic IR.
3. Semantic verifier checks shapes, dtypes, state edges, operation invariants, and entry points.
4. Semantic passes canonicalize, fold constants, and expose fusion opportunities.
5. Target lowering applies quantization, layouts, padding, KV representation, and fused regions for an `sm120` target profile.
6. Kernel providers enumerate valid candidates; autotuning may benchmark candidates under the same correctness contract.
7. DecodeStrategy contributes state and graphs; memory planner performs liveness-based arena allocation.
8. Compiler emits and validates a Physical Plan.
9. StoragePolicy packages plan, tensors, metadata, integrity, and provenance into `.sinf`.

Each step is deterministic given the same inputs, compiler build, target profile, and tuning database.

## `.sinf` Artifact

Logical sections (binary layout finalized in S01):

1. fixed magic/header and format compatibility range;
2. section directory with checked offsets, sizes, alignments, flags, and hashes;
3. canonical manifest containing model identity/revision, licenses, tokenizer/template identity, converter/compiler/toolchain versions, target capability fingerprint, quantization and pass pipeline;
4. tensor metadata table and aligned payload regions;
5. Physical Plan schema and provider/kernel identifiers;
6. optional tuning/evidence references and auxiliary heads;
7. integrity table/signature slot.

Unknown optional sections may be skipped; unknown required capabilities fail closed. Full structural validation occurs before device allocation. Format parsing is CPU-only and fuzzable.

## `sm120` Backend

The target profile captures compute capability, supported instruction/features, shared memory/register constraints, alignment, memory budget, and compiler ABI/kernel catalog fingerprint. It does not assume every RTX 5090 machine has identical clocks or power policy; these are benchmark-environment facts.

Backend responsibilities:

- lower semantic dtypes/layouts into supported physical forms;
- plan persistent weights, KV cache, scratch arenas, streams, and events;
- select provider candidates and materialize stable kernel IDs/launch data;
- validate plan hardware compatibility;
- expose trace hooks outside the hot path for correctness and profiling.

## Minimal Runtime

Runtime construction performs artifact validation, capability matching, device materialization, kernel ID resolution, workspace allocation, and command binding. Generation then executes prebuilt prefill/decode schedules.

Forbidden in the token hot path:

- heap or CUDA allocation/free;
- model-name/family checks;
- graph/pass/kernel selection;
- artifact parsing, filesystem I/O, logging formatting;
- JIT compilation;
- implicit device-wide synchronization.

Permitted bounded decisions include sequence-length/index checks, plan-authored command conditionals, and DecodeStrategy state transitions that were validated/materialized at construction.

## Kernel Portfolio

The portfolio evolves from reference-correct baselines to specialized families:

- dense GEMM/GEMV and quantized variants;
- RMSNorm, residual, fused residual-norm;
- RoPE application;
- embedding lookup and LM-head projection;
- prefill attention;
- decode attention with fused KV update and explicit paged/contiguous layout choices;
- gated FFN activation/fusions;
- MoE routing, top-k, permutation, grouped expert compute, and combine;
- sampling primitives;
- strategy-specific verification kernels where justified.

Provider capabilities describe shapes, layouts, dtypes, alignment, workspace, determinism, and numerical contract. Selection order is: applicable -> correct -> resource-valid -> measured -> stable. Baseline fallback is always explicit.

## Autoresearch Subsystem

An experiment is a declarative manifest plus an isolated patch/worktree and immutable inputs. The runner captures build identity, toolchain, GPU/driver, clocks/power policy, thermal state, seeds, workload, correctness oracle, raw timing samples, profiler references, and resulting patch.

Promotion pipeline:

```text
schema/budget check
  -> build + static/unit tests
  -> differential correctness
  -> sanitizer/bounds checks where applicable
  -> deterministic replay
  -> controlled warmup and measurement
  -> statistical/practical significance decision
  -> audit bundle + promotable commit
```

Any correctness, determinism, environment-validity, or sample-quality failure rejects the candidate before performance ranking. Experiments cannot rewrite workload semantics or tolerances.

## Dependency and Ownership Boundaries

- `runtime` may depend on physical-plan schema, artifact reader interfaces, device abstractions, and kernel registry; never frontends or semantic passes.
- `backends/sm120` may implement compiler/provider interfaces but not import model frontends.
- frontends depend on semantic IR and source schema utilities; never runtime.
- Python tooling communicates through versioned artifact/evidence schemas or a deliberately narrow binding.
- tests may depend on reference implementations; production providers may not.

Expected source layout:

```text
include/superinfer/{base,ir,compiler,artifact,runtime,kernels,decode}/
src/{base,ir,compiler,artifact,runtime}/
backends/sm120/{compiler,kernels,runtime}/
frontends/{qwen38,gemma4}/
python/superinfer/{convert,research,bench,report}/
schemas/
tests/{unit,property,golden,integration,gpu,fuzz}/
benchmarks/{manifests,baselines,reports}/
```

## Failure Model

Errors are typed by layer and accumulate context without leaking sensitive paths or dumping tensor data. Parser/verifier errors are recoverable and returned before device state changes. CUDA failures poison the active session, stop further launches, capture bounded diagnostics, and require explicit reconstruction. Partial artifact outputs use atomic replace and are never treated as valid.

## Architecture Fitness Tests

- dependency test prevents runtime -> frontend/compiler semantic imports;
- executor source contains no model identifiers;
- IR schemas round-trip and verify deterministically;
- pass pipeline order and invalidation are tested;
- provider selection uses capabilities rather than model identity;
- hot-path allocation and unexpected synchronization are traced;
- adding Gemma leaves executor behavior/source unchanged.
