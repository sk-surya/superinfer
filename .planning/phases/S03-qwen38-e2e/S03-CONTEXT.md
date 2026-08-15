# S03: Qwen3.8-27B End to End — Context

**Status:** Planned
**Depends on:** S02
**Critical-path role:** Produces the first real model proof: correct Qwen3.8-27B generation from `.sinf` on RTX 5090.

<domain>
## Phase Boundary

Pin the exact Qwen3.8-27B source/revision; implement its ModelFrontend, config/tensor/tokenizer mapping, required generic semantics/lowering, baseline execution, conversion, and rigorous end-to-end correctness. Optimize only enough to make the correctness proof usable; kernel performance work belongs to S04.
</domain>

<decisions>
## Locked Decisions

- [D-007] Qwen3.8-27B is the first supported model and owns the V0 critical path.
- Source repository/revision, tensor inventory, config, tokenizer, chat template, license, and reference implementation must be immutable inputs before code is written.
- No Qwen-named branch or type may enter the physical executor or KernelProvider selection.
- Frontend failures are actionable and strict; unknown semantics do not silently fall back.
- Token/logit correctness and long-context KV invariants precede performance claims.
- If fitting the model requires quantization/context constraints, they are explicit in the artifact and acceptance scope.

### Executor Discretion

- Exact trusted reference harness and intermediate capture mechanism.
- Exact initial quantization/storage configuration after a checked RTX 5090 memory ledger.
- Small generic IR/pass/provider additions needed by the pinned source, provided names/semantics are model-independent.
</decisions>

<understanding>
## Understanding Gate

**Level:** L1 — non-blocking.

Trace one real weight from its pinned Hugging Face tensor name through validation, quantization/packing, `.sinf` offset and metadata, host/device materialization, plan binding, and the GPU operation that consumes it. Record the shape/layout/dtype changes and the first diagnostic at each boundary. No code-reading-completeness or hard stop is required.
</understanding>

<canonical_refs>
## Canonical References

- `.planning/PROJECT.md` — critical-path definition and V0 success.
- `.planning/RESEARCH.md` — model support research contract.
- `.planning/REQUIREMENTS.md` — MOD-001–004, DEC-001, KER-006, FMT-003, BCK-004.
- `.planning/QUALITY.md` — full-model differential and numerical contracts.
- `.planning/RISKS.md` — R-01, R-03, R-11, R-12, R-15.
- `.planning/UNDERSTANDING-GATES.md` — S03 L1 behavior.
</canonical_refs>

<deferred>
## Deferred

Aggressive specialized kernels, speculative decoding/DSpark, competitive graphs, other Qwen variants, server APIs, continuous batching, and additional model families.
</deferred>
