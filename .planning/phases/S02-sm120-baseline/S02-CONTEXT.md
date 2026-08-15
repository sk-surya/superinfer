# S02: `sm120` Correctness Backend — Context

**Status:** Planned
**Depends on:** S01
**Critical-path role:** Establishes real RTX 5090 execution and memory/lifecycle correctness before full-model integration.

<domain>
## Phase Boundary

Implement target probing/profile, `sm120` lowering decisions, memory/workspace planning, a minimal Physical Plan executor, baseline KernelProvider implementations sufficient for small synthetic graphs, and GPU correctness/lifecycle tests. Performance specialization waits for S04.
</domain>

<decisions>
## Locked Decisions

- [D-004] Target RTX 5090 `sm_120a` first and reject incompatible targets.
- [D-006] Baseline correctness exists before optimized providers.
- [D-009] No allocation, model branching, JIT, filesystem, or implicit global sync in the token hot path.
- [D-012] Buffers, streams, events, mappings, sessions, and plans have explicit owners.
- Runtime validates the complete Physical Plan before device allocation/launch.
- Provider selection is capability/shape/layout based, never model-name based.

### Executor Discretion

- Exact CUDA wrappers and baseline implementation technique.
- Single or small fixed stream topology for V0, provided the plan encodes dependencies explicitly.
- CPU oracle versus external trusted reference bridge per operation.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — `sm120` backend, minimal runtime, portfolio contract.
- `.planning/REQUIREMENTS.md` — BCK-001–006 and KER-001/002/005.
- `.planning/QUALITY.md` — differential, lifecycle, sanitizer and hot-path tests.
- `.planning/RISKS.md` — R-02, R-05, R-12, R-17.
</canonical_refs>

<deferred>
## Deferred

Full Qwen graphs/tensors, advanced fusion, tuned kernels, autoresearch, public comparison, continuous batching and multi-GPU.
</deferred>
