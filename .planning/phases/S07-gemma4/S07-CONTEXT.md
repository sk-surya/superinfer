# S07: Gemma 4 26B-A4B Extension — Context

**Status:** Planned
**Depends on:** S06
**Critical-path role:** Tests whether the approved abstractions are real by adding the second model family after the Qwen proof.

<domain>
## Phase Boundary

Pin and implement Gemma 4 26B-A4B through the existing ModelFrontend, GraphPass, KernelProvider, DecodeStrategy and StoragePolicy surfaces; add generic semantic operations/providers only where required; convert to `.sinf` and prove correct RTX 5090 generation. Performance is diagnostic, not a new public race.
</domain>

<decisions>
## Locked Decisions

- [D-007] Gemma 4 26B-A4B is second, after the Qwen performance proof.
- The physical executor must not receive model-specific edits. Its source/dependency diff is an exit gate.
- New semantics are generic and mathematically documented; “Gemma” names remain in the frontend/fixtures/docs.
- Exact source revision/config/tensor/tokenizer/license details are pinned before implementation.
- Reuse is not forced when semantics differ; represent real distinctions explicitly in Semantic IR.

### Executor Discretion

- New generic operations, passes, layout/provider variants, and StoragePolicy modes required by pinned semantics.
- Initial quantization/context envelope based on a transparent RTX 5090 memory ledger.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/ARCHITECTURE.md` — extension contracts and fitness tests.
- `.planning/RESEARCH.md` — model support research contract.
- `.planning/REQUIREMENTS.md` — MOD-005, ARCH-005/006, KER-002, FMT-004, QUA-004.
- `.planning/RISKS.md` — R-03, R-11, R-12, R-15, R-18.
</canonical_refs>

<deferred>
## Deferred

Gemma competitive performance claims, broad Gemma variants, architecture-specific executor forks, non-5090 backends, server/multi-GPU work.
</deferred>
