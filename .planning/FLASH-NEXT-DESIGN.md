# Flash-Next Architecture Amendment

**Status:** Approved
**Date:** 2026-08-26
**Target phase:** S03F, inserted after S03 and before S04

## Purpose

Add Qwen3.8-Flash-Next as SuperInfer's second flagship correctness target after Qwen3.8-27B. The phase exists to exercise architectural extension points that were intentionally deferred from v0: multi-device placement, heterogeneous host/device residency, PLE/N-gram lookup, MoE, sparse attention, and gated residual execution.

This amendment does **not** change S03 acceptance. Qwen3.8-27B remains the first end-to-end proof of `.sinf -> compiler -> Physical Plan -> CUDA -> reference-equivalent generation`.

## Execution policy

- S03 remains the primary implementation lane until its existing acceptance criteria are met.
- S03F-01 may begin immediately as a parallel **research-only** lane.
- S03F-02 through S03F-06 may not modify runtime architecture until S03 closes.
- Vision and MTP are explicitly out of scope for initial Flash-Next support.
- Correctness precedes performance. S04 remains the first broad optimization phase.

## Success criterion

Text-only Flash-Next produces reference-equivalent greedy generation on 2x RTX 5090 using a deterministic compile-time device-placement plan, with PLE/N-gram tables host-resident and no accidental model-specific branching in the executor.

A stronger preferred target is that expert weights remain resident during steady-state decode, but S03F-01 must first prove whether a quality-preserving quantization recipe fits the available VRAM budget with useful workspace headroom.

## Architectural additions

### Multi-device placement

Physical buffers and commands gain explicit placement. Placement is a compiler decision, not a frontend concern.

Conceptual contract:

```text
PlacementDomain = Device(ordinal) | PinnedHost | Host | Mmap

BufferPlacement
  domain
  residency = persistent | staged | cached

ExecutionPlacement
  device_id
  stream

PhysicalCommand
  KernelCommand
  TransferCommand
  BarrierCommand
```

Transfers are explicit commands. The first policy is contiguous layer partitioning selected from actual packed artifact sizes under per-device memory/headroom constraints. Tensor parallelism is out of scope.

### PLE / N-gram lookup

PLE is modeled as a first-class semantic/storage path, not generic CPU offload:

```text
TokenHistory
  -> NgramIndex
  -> NgramEmbeddingLookup
  -> NgramProjection/Combine
  -> model injection
```

StoragePolicy owns host mapping, independent PLE quantization, sparse gathers, pinned staging, and asynchronous H2D transfer. Initial PLE storage is read-only.

### MoE

The existing semantic MoE concepts become executable through a pipeline equivalent to:

```text
moe_route
  -> moe_top_k
  -> moe_dispatch
  -> grouped_expert_compute
  -> shared_expert
  -> moe_combine
```

The first implementation uses layer-local expert residency. Expert prediction, expert parallelism, hot/cold caches, and routing-aware optimization are deferred to S04+.

### QSA

Sparse block selection remains semantically distinct from sparse attention:

```text
QSAIndexer -> SparseBlockSelection
Q/K/V + selected blocks -> SparseAttention
```

This permits future indexer research without coupling retrieval policy to attention execution.

### Gated Residual

The exact semantic boundary must follow the pinned reference equations. The intended design separates residual read/write gating from the surrounding layer rather than hiding it inside a model-named kernel. Candidate semantic concepts are `GatedResidualRead` and `GatedResidualWrite` or one parameterized generic equivalent.

## Capacity policy

S03F-01 must produce an exact memory ledger from the pinned artifact before implementation assumptions are made. It must separately account for routed experts, shared experts, PLE, non-expert text weights, KV/recurrent state, router/indexer, embedding/LM head, workspace, optional MTP, and vision.

The compiler must optimize placement from packed `.sinf` bytes rather than hard-code a 24/24 layer split.

If full expert residency cannot be achieved at an acceptable quality point, an explicit architecture decision is required before adding expert staging/caching. The runtime must not silently page expert weights during decode.

## Correctness ladder

S03F qualification proceeds in this order:

1. PLE/N-gram lookup contract.
2. Router/top-k contract.
3. Single expert projection.
4. Complete MoE layer.
5. QSA indexer and block selection.
6. Sparse attention.
7. Gated residual.
8. One GDN layer.
9. One QSA layer.
10. Cross-device transfer boundary.
11. Multi-token state continuation.
12. 48-layer greedy generation.

Every layer-level milestone compares selected intermediate tensors against the pinned reference implementation, not only the final output.

## Non-goals

- Tensor parallel execution.
- General distributed serving.
- Vision support.
- MTP/speculative execution.
- Performance heroics before correctness.
- Generic arbitrary CPU offload.
- Model-name dispatch in runtime or kernels.
