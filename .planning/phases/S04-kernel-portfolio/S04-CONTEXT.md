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

<understanding>
## Understanding Gate C

**Level:** L2, repeated by major mechanism.

- **C.1:** GEMV/GEMM to NVFP4/Tensor Core mechanisms and dense-support kernels (`S04-01`).
- **C.2:** prefill/decode attention and KV traffic (`S04-02`).
- **C.3:** fusion, grouped/MoE work, and persistent/megakernel tradeoffs (`S04-03`).

Each mini-gate produces a packet and counts as an L2 boundary. The user should apply a roofline model, distinguish memory hierarchy and Tensor Core/CUDA-core work, reason about occupancy/resources and fusion costs, predict an Nsight result before profiling, and diagnose a likely bottleneck. Agents continue between mechanisms only while debt remains at most one.
</understanding>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — KernelProvider/portfolio/hot path.
- `.planning/RESEARCH.md` — candidate research axes and experiment template.
- `.planning/BENCHMARKS.md` — micro/component/E2E measurement rules.
- `.planning/REQUIREMENTS.md` — KER-002–006, BCK-002/004, BEN-002.
- `.planning/RISKS.md` — R-06, R-08, R-12, R-17.
- `.planning/UNDERSTANDING-GATES.md` — Gate C mini-gates and debt protocol.
</canonical_refs>

<deferred>
## Deferred

Unbounded automated search, DSpark/speculation, external baseline claims, non-Qwen shapes without generic reuse value, other architectures and multi-GPU.
</deferred>
