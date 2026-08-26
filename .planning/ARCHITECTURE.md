# Architecture

## Thesis

SuperInfer is **composable at the control plane and ruthlessly specialized at the data plane**. Researchers compose semantic model structure, passes, providers, storage, decoding policies and target placement through stable interfaces. The compiler resolves those choices into an immutable Physical Plan. The runtime validates and executes that plan with minimal policy.

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
          |               + placement/transfer specialization
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

Purpose: represent model meaning independent of CUDA, byte layout and device placement.

Contains typed tensors, symbolic/static dimensions, dtypes/quantization intent, model operations, state edges such as KV/recurrent state, and named entry points for prefill/decode/optional auxiliary heads. Operations include embedding, normalization, RoPE, dense/local/grouped/sparse attention semantics, Gated DeltaNet, dense gated FFN, MoE routing/dispatch/experts/combine, residual/gated-residual composition, PLE/N-gram indexing/lookup/combine, LM head and sampling inputs.

It must not contain CUDA grid/block sizes, function pointers, file offsets, device addresses, provider-specific opaque blobs, GPU ordinals, or transfer policy.

### 2. Lowered IR

Purpose: make target-aware decisions while preserving inspectability and verifier-friendly structure.

Contains physical tensor layouts, padded dimensions, quantized storage types/scales, fused regions, workspace requirements, KV/recurrent layouts, memory-space intent, state slots/transitions, scheduling dependencies, candidate kernel capability requirements, and explicit prefill/decode graphs. Lowered IR may express residency constraints but does not own executable CUDA handles.

### 3. Physical Plan

Purpose: be the immutable, validated execution contract.

Contains ordered command streams/DAGs, resolved kernel IDs, launch parameters, buffer arena offsets, **placement domains/device ordinals**, stream/event dependencies, **explicit transfer/barrier commands**, constants, decode-state layout, selected strategy, capability fingerprints and per-domain resource bounds. It contains no unresolved model semantics.

Runtime validation checks all indexes, offsets, sizes, placements, topology requirements, capability fingerprints, dependencies, transfer endpoints and workspace bounds before allocating/executing.

Temporary analyses—dominance, liveness, shape maps, cost models, layer byte ledgers—are pass-local data, not additional IRs.

## Five Extension Surfaces

### `ModelFrontend`

Owns source config/tensor/tokenizer validation and emission of canonical Semantic IR. It never selects CUDA kernels or devices. Frontends are pure/deterministic for pinned inputs and emit actionable schema diagnostics.

### `GraphPass`

Owns verified transformations over Semantic or Lowered IR. Every pass declares input representation, preconditions, preserved/invalidated analyses, deterministic configuration and postconditions. Pass ordering is explicit; pass pipelines are serialized in compilation provenance.

### `KernelProvider`

Owns target implementations and capability descriptions. Providers enumerate candidates based on operation/layout/dtype/shape/target constraints. Selection is capability- and measurement-driven, never a model-name switch. A correct baseline provider remains available.

### `DecodeStrategy`

Owns token-level policy: greedy, sampling, multi-token/speculative proposal, verification, acceptance, rollback and state transitions. It declares required graphs, buffers and provider capabilities during compilation. DSpark lives here.

### `StoragePolicy`

Owns artifact tensor layout, alignment, compression/quantization payload conventions, host/mmap mapping, staging, residency class and device materialization. Kernels see typed physical views, not file-format details. PLE/N-gram memory is a first-class StoragePolicy consumer rather than generic CPU offload.

## Placement and Residency

S03F extends physical specialization without adding a sixth extension surface.

Conceptual contracts:

```text
PlacementDomain
  Device(ordinal)
  PinnedHost
  Host
  Mmap

ResidencyClass
  persistent
  staged
  cached

PhysicalCommand
  KernelCommand
  TransferCommand
  BarrierCommand
```

Placement is compiler-owned. Frontends may author semantic locality/state requirements but never GPU IDs.

The first multi-device policy is deterministic contiguous layer partitioning selected from **actual packed artifact bytes** under independent per-device budgets and configured workspace/headroom. A hard-coded 24/24 split and tensor-parallel collectives are forbidden in S03F.

Transfers are explicit physical commands. Preferred device-to-device execution uses asynchronous peer copies when supported; a plan/runtime may use an explicit pinned-host staged fallback when peer access is unavailable. Hidden copies inside kernel launchers are not allowed.

## PLE / N-gram Memory

PLE is modeled as model semantics plus StoragePolicy, not arbitrary CPU offload:

```text
TokenHistory
  -> NgramIndex
  -> NgramEmbeddingLookup
  -> NgramProjection/Combine
  -> model injection
```

Exact hashing/index semantics come from the pinned reference implementation and are differentially tested. The table can use an independent quantization recipe and remain read-only in host/mmap storage. Runtime gathers only requested entries into preallocated pinned staging, transfers the resulting small vectors asynchronously, and never materializes the whole table on the GPU unless a future explicit policy says otherwise.

## MoE

Generic MoE semantics remain decomposed:

```text
route -> top_k -> dispatch -> routed experts
                         +-> shared expert
                         -> combine
```

Initial Flash-Next placement keeps all experts for a layer on the same device as that layer when S03F-01 proves a fitting quality/residency recipe. Dynamic expert caching, expert prediction and cross-device expert parallelism are research topics, not correctness-path behavior.

If full expert residency is not feasible at acceptable quality, implementation must stop for an explicit capacity/residency ADR rather than silently paging weights.

## Sparse Attention / QSA

Sparse selection and sparse attention are distinct semantics:

```text
QSAIndexer -> SparseBlockSelection
Q/K/V + SparseBlockSelection -> SparseAttention
```

This preserves the ability to research indexers/retrieval policies without rewriting the attention executor or conflating selection with compute.

## Gated Residual

Gated residual behavior is represented generically from the pinned reference equations. Residual branch state is explicit across lowering and physical execution. The IR must not encode Flash-Next model identity; it should expose only the read/write gating semantics and dimensions required for correctness.

## Compiler Pipeline

1. Pin and validate source identity and licenses.
2. Frontend parses config/tensor inventory/tokenizer and emits Semantic IR.
3. Semantic verifier checks shapes, dtypes, state edges, operation invariants and entry points.
4. Semantic passes canonicalize, fold constants and expose fusion opportunities.
5. Target lowering applies quantization, layouts, padding, state representation and fused regions for an `sm120` target profile.
6. Storage/capacity analysis determines packed-byte residency constraints and available placement domains.
7. Kernel providers enumerate valid candidates; correctness is established before any measured selection.
8. DecodeStrategy contributes state/graphs; memory planner performs liveness-based per-domain arena allocation.
9. Placement specialization assigns layers/resources to devices and emits explicit transfer/barrier commands.
10. Compiler emits and validates a Physical Plan.
11. StoragePolicy packages plan, tensors, metadata, integrity and provenance into `.sinf`.

Each step is deterministic given the same inputs, compiler build, target profile, placement policy and tuning database.

## `.sinf` Artifact

Logical sections include:

1. fixed magic/header and format compatibility range;
2. section directory with checked offsets, sizes, alignments, flags and hashes;
3. canonical manifest containing model identity/revision, licenses, tokenizer/template identity, converter/compiler/toolchain versions, target capability fingerprint, quantization, storage/residency policy and pass pipeline;
4. tensor metadata table and aligned payload regions, including host-resident sections where applicable;
5. Physical Plan schema and provider/kernel/placement identifiers;
6. optional tuning/evidence references and auxiliary heads;
7. integrity table/signature slot.

Unknown optional sections may be skipped; unknown required capabilities fail closed. Full structural validation occurs before device allocation. Format parsing is CPU-only and fuzzable.

## `sm120` Backend

The target profile captures compute capability, supported instruction/features, shared memory/register constraints, alignment, device topology/capabilities, memory budgets and compiler ABI/kernel catalog fingerprint. Clock/power/thermal facts remain benchmark-environment facts.

Backend responsibilities:

- lower semantic dtypes/layouts into supported physical forms;
- plan persistent weights, host-resident data, KV/recurrent state, scratch arenas, streams and events;
- select provider candidates and materialize stable kernel IDs/launch data;
- assign placement domains and explicit transfers;
- validate hardware/topology compatibility;
- expose trace hooks outside the hot path for correctness and profiling.

## Minimal Runtime

Runtime construction performs artifact validation, capability/topology matching, host mapping, device materialization, kernel ID resolution, per-device workspace allocation, transfer-stream/event construction and command binding. Generation then executes prebuilt schedules.

Forbidden in the token hot path:

- heap or CUDA allocation/free;
- model-name/family checks;
- graph/pass/kernel/device-placement selection;
- artifact parsing or filesystem metadata discovery;
- JIT compilation;
- implicit device-wide synchronization;
- unplanned expert-weight paging or hidden cross-device copies.

Permitted bounded decisions include sequence-length/index checks, plan-authored conditionals, sparse PLE gathers, explicit transfer commands and DecodeStrategy state transitions that were validated/materialized at construction.

## Kernel Portfolio

The portfolio evolves from reference-correct baselines to specialized families:

- dense GEMM/GEMV and quantized variants;
- RMSNorm, residual, gated residual, fused residual-norm;
- RoPE application;
- embedding lookup and LM-head projection;
- prefill/decode grouped attention and explicit KV update/layout choices;
- QSA index/select and sparse attention;
- Gated DeltaNet/recurrent state kernels;
- gated FFN activation/fusions;
- MoE routing, top-k, permutation, grouped expert compute, shared expert and combine;
- PLE gather/dequant/combine where GPU work is appropriate;
- sampling and strategy-specific verification kernels where justified.

Provider capabilities describe shapes, layouts, dtypes, alignment, workspace, determinism and numerical contract. Selection order is: applicable -> correct -> resource-valid -> measured -> stable. Baseline fallback is always explicit.

## Autoresearch Subsystem

An experiment is a declarative manifest plus an isolated patch/worktree and immutable inputs. The runner captures build identity, toolchain, GPU/driver/topology, clocks/power/thermal state, seeds, workload, correctness oracle, placement/residency, raw timings/transfers, profiler references and resulting patch.

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

Any correctness, determinism, environment-validity or sample-quality failure rejects the candidate before performance ranking. Experiments cannot rewrite workload semantics, residency requirements or tolerances.

## Dependency and Ownership Boundaries

- `runtime` may depend on Physical Plan schema, artifact reader interfaces, device abstractions and kernel registry; never frontends or semantic passes.
- `backends/sm120` may implement compiler/provider/placement interfaces but not import model frontends.
- frontends depend on semantic IR and source schema utilities; never runtime or device ordinals.
- StoragePolicy may expose host/mmap/device materialization contracts but not choose model semantics.
- Python tooling communicates through versioned artifact/evidence schemas or a deliberately narrow binding.
- tests may depend on reference implementations; production providers may not.

Expected source layout remains focused; S03F additions should fit under existing compiler/runtime/backend/frontends rather than creating a parallel architecture.

## Failure Model

Errors are typed by layer and accumulate context without dumping tensor data. Parser/verifier errors are recoverable and returned before device state changes. Placement/topology validation fails before partial execution. CUDA failure on any device poisons the active session, stops dependent launches, captures bounded diagnostics and requires explicit reconstruction. Partial artifact outputs use atomic replace and are never treated as valid.

## Architecture Fitness Tests

- dependency test prevents runtime -> frontend/compiler semantic imports;
- executor source contains no model identifiers;
- IR schemas round-trip and verify deterministically;
- pass pipeline order and invalidation are tested;
- provider selection uses capabilities rather than model identity;
- device placement uses compiler-owned physical facts rather than frontend/model names;
- all inter-domain movement appears as explicit transfer commands/traces;
- hot-path allocation and unexpected synchronization are traced;
- Qwen single-device correctness is unchanged by S03F infrastructure;
- Flash-Next fits through generic PLE/MoE/QSA/gated-residual/state/placement contracts;
- later Gemma support audits that those additions did not turn into hidden Qwen-family assumptions.
