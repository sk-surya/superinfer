---
phase: "S05-autoresearch"
plan: "S05-03"
type: "research"
wave: 3
depends_on: [S05-02]
files_modified:
  - include/superinfer/decode/**
  - src/decode/**
  - backends/sm120/compiler/decode/**
  - backends/sm120/kernels/verification/**
  - python/superinfer/research/strategies/**
  - tests/{unit,property,integration,gpu}/decode/**
  - experiments/dspark/**
autonomous: false
requirements_addressed: [DEC-001, DEC-002, DEC-003, DEC-004, RES-001, RES-002]
must_haves:
  truths:
    - "Decode strategies declare all graphs/state/workspace before runtime execution."
    - "Speculation preserves token semantics and KV state under accept/reject/rollback boundaries."
    - "DSpark is researched as a DecodeStrategy and does not alter attention interfaces."
  artifacts:
    - "DecodeStrategy conformance suite and speculative state machine"
    - "Pinned DSpark research note/manifest with unknowns made explicit"
    - "Correctness-gated DSpark experiment(s), whether positive or negative"
---

# S05-03 — Speculative Decode Framework and DSpark Track

## Objective

Make decoding policy genuinely composable, establish a correct speculative state machine, and run DSpark as a bounded research hypothesis without contaminating attention kernels or public claims.

## Tasks

1. **Harden DecodeStrategy contract and conformance suite**
   - Formalize compile-time graph/resource declarations, runtime inputs/outputs, deterministic state, reset, error, stop and capacity behavior.
   - Validate greedy and sampling strategies as reference implementations over the same executor contract.
   - Assert no dynamic allocation or provider/model selection during strategy transitions.

2. **Specify the speculative state machine**
   - Model proposal tokens/state, target verification, acceptance count, rejection replacement, stop conditions, rollback/checkpoint and KV commit/restore.
   - Define invariants for committed prefix, proposed suffix, token positions, RNG consumption and KV visibility.
   - Use property/state-machine testing across accept-none/some/all, EOS, capacity boundary, errors and repeated resets.

3. **Materialize speculation at compile time**
   - Extend strategy compilation to request proposal/verification graphs, bounded buffers/workspaces and provider capabilities.
   - Physical Plan owns fixed maximum proposal width and commands; runtime only follows validated transitions.
   - Any attention or verification kernel remains capability-described computation without proposal-policy logic.

4. **Pin DSpark primary semantics before implementation**
   - Locate and pin authoritative paper/code/version/license; document algorithm, assumptions, proposal source, verification/acceptance behavior, state/KV requirements and claimed workload.
   - Map each behavior to DecodeStrategy or a generic requested kernel capability.
   - Mark unresolved/unsupported parts and stop rather than inventing semantics.

5. **Create bounded DSpark experiment manifest**
   - Define exact Qwen artifact, prompts, output lengths, reference strategy, proposal width/domain, correctness corpus, resource budget and accept rule.
   - Keep attention provider and benchmark semantics immutable; record any new generic verification kernel separately.

6. **Implement and gate the minimal faithful candidate**
   - Implement the strategy/state logic and only required generic kernels/passes.
   - Run unit/property/differential/KV rollback/determinism/hot-path gates before performance.
   - If correct, measure acceptance, overhead, TTFT/TPOT and memory under controlled manifests; if not beneficial, retain a negative result.

7. **Run architecture conformance review**
   - Prove no DSpark references or proposal/rollback decisions appear in attention provider interfaces/implementations.
   - Prove strategy resources appear in Lowered IR/Physical Plan and `.sinf` compatibility metadata.

## Verification

- State-machine property tests cover accept/reject/rollback/EOS/capacity/error transitions.
- Speculative output matches the non-speculative target strategy under the approved semantic contract.
- Hot-path trace and memory bounds pass at maximum proposal width.
- DSpark experiment is replayable and either correctly rejected or eligible for reviewed promotion.

## Completion Evidence

- DecodeStrategy/speculative state specification and conformance results.
- DSpark source/provenance mapping and experiment bundle.
- Architecture diff/test proving DSpark remains out of attention interfaces.
