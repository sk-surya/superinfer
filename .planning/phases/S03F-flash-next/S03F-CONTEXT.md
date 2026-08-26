# S03F — Flash-Next Bring-up Context

## Position in roadmap

S03F is inserted between S03 Qwen3.8 correctness and S04 kernel optimization.

S03F-01 is research-only and may run in parallel with the remaining S03 work. S03F-02 through S03F-06 are blocked until S03's existing end-to-end correctness gate closes.

## Goal

Demonstrate that SuperInfer's compiler/runtime architecture can express and correctly execute a text-only heterogeneous, stateful, sparse MoE model across 2x RTX 5090 without model-specific runtime branching.

## Canonical spec

Read `.planning/FLASH-NEXT-DESIGN.md` before executing any S03F plan.

## Global constraints

- Preserve semantic composition -> compile-time specialization -> minimal physical runtime.
- Frontends author model meaning only; they never select kernels or devices.
- Placement and transfer decisions belong to compilation/physical planning.
- Runtime consumes immutable physical schedules.
- Correctness is a hard gate before timing.
- S03 acceptance is unchanged.
- Vision and MTP are excluded from S03F.
- No tensor parallelism in initial dual-GPU support.
- No arbitrary expert-weight paging may be introduced without an explicit capacity decision.

## Understanding gate

S03F is L2 at the architecture boundary. The user is not required to block S03F-01 research. Before implementation progresses beyond the first complete multi-device Flash-Next layer path, prepare an Understanding Packet covering placement, transfer commands, heterogeneous PLE residency, MoE execution, QSA selection/attention separation, and state continuation.
