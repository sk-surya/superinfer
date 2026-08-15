# S04: Kernel Portfolio and Specialization — Context

**Status:** Planned
**Depends on:** S03
**Critical-path role:** Turns the correct Qwen baseline into a fast, shape-aware `sm120` engine without sacrificing fallbacks or evidence.

<domain>
## Phase Boundary

Implement and tune a capability-described portfolio for Qwen-critical dense, attention/KV, FFN/MoE, normalization/RoPE, embedding/LM-head and sampling regions. Add fusion and selector policies only when correctness and measured end-to-end benefit support them. Autoresearch automation and public claims remain S05/S06.
</domain>

<decisions>
## Locked Decisions

- [D-006] A specialized candidate cannot replace its baseline before differential/boundary/determinism gates.
- Prefer a portfolio with explicit applicability envelopes over one universal megakernel.
- Provider selection uses operation/layout/dtype/shape/target/resource capabilities, never model identity.
- Prefill and decode are distinct optimization regimes and benchmarks.
- Baseline fallback stays tested and selectable.
- No new hot-path allocation, registry mutation, policy selection or device-wide synchronization.

### Executor Discretion

- CUDA implementation technique, libraries, inline PTX and code generation subject to license/toolchain/readability review.
- Fusion boundaries and tuning parameters based on evidence.
- Which lower-value candidate variants are deferred after operation-matrix coverage and profiling.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — KernelProvider/portfolio/hot path.
- `.planning/RESEARCH.md` — candidate research axes and experiment template.
- `.planning/BENCHMARKS.md` — micro/component/E2E measurement rules.
- `.planning/REQUIREMENTS.md` — KER-002–006, BCK-002/004, BEN-002.
- `.planning/RISKS.md` — R-06, R-08, R-12, R-17.
</canonical_refs>

<deferred>
## Deferred

Unbounded automated search, DSpark/speculation, external baseline claims, non-Qwen shapes without generic reuse value, other architectures and multi-GPU.
</deferred>
