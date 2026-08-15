# S00: Foundation and Contracts — Context

**Status:** Ready for execution
**Depends on:** None
**Critical-path role:** Makes every later phase buildable, testable, and architecture-conformant.

<domain>
## Phase Boundary

Create the repository skeleton, build/toolchain policy, five extension interfaces, three representation shells, typed base utilities, reference-test scaffolding, and CPU CI. Do not implement model semantics, artifact binary details, GPU kernels, or performance claims.
</domain>

<decisions>
## Locked Decisions

- [D-001] Control-plane composition is separate from minimal runtime execution.
- [D-002] The repository exposes Semantic IR, Lowered IR, and Physical Plan—not a growing dialect stack.
- [D-003] The only extension surfaces are `ModelFrontend`, `GraphPass`, `KernelProvider`, `DecodeStrategy`, and `StoragePolicy`.
- [D-011] C++20/CUDA own compiler/runtime; typed Python owns converter/research/report orchestration.
- [D-012] Ownership, lifetime, thread-safety, and error behavior are explicit at public APIs.
- Build and tests must work without a GPU. CUDA discovery may be optional in the default developer path.
- The runtime target remains a small library/executable; avoid framework/plugin registries with dynamic loading in V0.

### Executor Discretion

- Exact build system and dependency manager, provided they are reproducible and transparent.
- Exact testing, linting, and schema libraries after a short documented comparison.
- Header/source subdivision consistent with `.planning/ARCHITECTURE.md`.
</decisions>

<understanding>
## Understanding Gate

**Level:** L0/L1 — non-blocking.

Agents continue autonomously. Teach only the basic CUDA execution model: host/device memory, asynchronous launches, streams/events, synchronization, kernel launch, and thread/warp/block/SM hierarchy. Ask: “Why can CPU wall time report 0.01 ms while the GPU is still working?” The hands-on exercise compares unsynchronized CPU timing, synchronized CPU timing, and CUDA-event timing for one small kernel.
</understanding>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — layers, interfaces, dependency rules, source layout.
- `.planning/REQUIREMENTS.md` — ARCH-001/004/005/008 and QUA-001/003.
- `.planning/QUALITY.md` — test pyramid and CPU CI lanes.
- `.planning/UNDERSTANDING-GATES.md` — non-blocking S00 behavior and transition output.
- `AGENTS.md` — coding, ownership, evidence, and hot-path rules.
</canonical_refs>

<deferred>
## Deferred

Concrete IR operations, `.sinf` layout, `sm120` device code, model frontends, Python/HF dependencies, benchmark infrastructure, and GPU CI are later phases.
</deferred>
