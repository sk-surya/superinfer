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

<understanding>
## Understanding Gate B

**Level:** L2 — own one-token transformer execution.

At S02 completion, produce and present Gate B's packet. Trace one token through embedding, normalization, attention/KV, residual, FFN/MoE where applicable, LM head, and sampling, with important tensor shapes derived from the Qwen config pinned during S03 preparation. Explain which dimensions change between prefill and decode and where memory/workspace and launches are fixed. The hands-on exercise runs a tiny reference transformer, captures selected intermediates, changes one shape/config value, and predicts the resulting tensor/plan changes. Pinning config evidence is preparation, not Qwen implementation in S02.
</understanding>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — `sm120` backend, minimal runtime, portfolio contract.
- `.planning/REQUIREMENTS.md` — BCK-001–006 and KER-001/002/005.
- `.planning/QUALITY.md` — differential, lifecycle, sanitizer and hot-path tests.
- `.planning/RISKS.md` — R-02, R-05, R-12, R-17.
- `.planning/UNDERSTANDING-GATES.md` — Gate B packet and debt protocol.
</canonical_refs>

<deferred>
## Deferred

Full Qwen graphs/tensors, advanced fusion, tuned kernels, autoresearch, public comparison, continuous batching and multi-GPU.
</deferred>
