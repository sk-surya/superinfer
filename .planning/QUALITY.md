# Quality Strategy

## Principles

- Tests prove contracts and failure behavior, not just line coverage.
- CPU-only development remains productive; GPU lanes add device truth.
- Optimized paths are never the oracle for themselves.
- Evidence and reproduction are part of definition of done.
- Flaky tests are defects and are quarantined only with an owner, issue, and expiration.

## Test Pyramid

### Static and Build

- C++/CUDA formatting, warnings-as-errors for owned code, static analysis on supported subset;
- Python formatting, lint, type checking, package/build validation;
- dependency-boundary and forbidden-import tests;
- generated-schema freshness and deterministic-codegen checks.

### Unit and Property Tests

- IR constructors/verifiers, pass preconditions/invalidation, shape/dtype algebra;
- artifact offset/overflow/alignment/version validation;
- memory planner liveness/non-overlap/resource bounds;
- kernel capability matching and fallback;
- decode state machines including rollback/KV consistency;
- experiment schemas, statistics, and promotion rules.

Property tests use persisted failure seeds and bounded runtime.

### Golden and Compatibility Tests

- canonical Semantic/Lowered/Physical dumps;
- `.sinf` byte/manifest fixtures for supported versions;
- tokenizer/config/tensor mappings;
- benchmark/report schemas and generated tables;
- supported reader/writer compatibility matrix.

Golden updates require an explanation and semantic diff review.

### Differential Tests

- small CPU/reference executor vs trusted tensor/reference implementation;
- each GPU kernel vs independent reference across random, boundary, and model shapes;
- full-model logits at selected layers/positions and final token sequences;
- baseline vs specialized plan equivalence within documented numerical contracts.

### Fuzz and Negative Tests

- artifact and schema parsers: truncation, corrupt section tables, overflow, aliasing, unknown capability/version, checksum failure;
- frontend config/tensor schemas: missing/extra/wrong-shaped tensors and unsafe values;
- Physical Plan verifier: cycles, invalid dependencies, overlapping buffers, bad launches, unsupported kernels;
- CLI inputs and experiment manifests.

### GPU Lifecycle and Sanitizer Tests

- compute-sanitizer/memcheck/racecheck where supported;
- stream/event ordering, session teardown after error, repeated load/unload;
- OOM/resource-bound rejection and no partial launch;
- hot-path allocation/synchronization trace;
- long-context KV bounds and wrap/rollback cases.

## CI Lanes

| Lane | Trigger | Hardware | Content | Blocking |
|---|---|---|---|---|
| Fast | every PR | CPU | format, lint, type, build, focused unit | yes |
| Full CPU | every PR/main | CPU | all unit/property/golden/fuzz-smoke/integration | yes |
| GPU smoke | eligible PR | RTX 5090 | load, tiny plan, small kernel set | yes once infrastructure is stable |
| GPU correctness | main/nightly | RTX 5090 | kernel matrix, Qwen/Gemma differentials | yes for release |
| GPU sanitizer | scheduled | RTX 5090 | lifecycle/bounds/race suites | yes for release |
| Benchmark | controlled/manual + scheduled | dedicated RTX 5090 | fixed manifests, regression evaluation | claim/release gate |

Benchmark jobs never share the GPU and record environment validity before accepting samples.

## Numerical Contracts

Each operation documents input/output dtype, accumulation dtype, deterministic guarantees, exceptional-value behavior, and absolute/relative/ULP or task-level tolerance. Tolerances derive from reference analysis and representative error distributions. Token agreement is required for deterministic acceptance prompts unless an explicitly approved logits-based criterion explains divergence.

## Coverage Expectations

- 100% of public parsers/verifiers and error branches have tests;
- 100% of registered kernel capability envelopes have at least one positive and negative case;
- every requirement has at least one named automated or manual evidence producer;
- line coverage is reported for trend/debugging but never substitutes for the above contract coverage.

## Understanding-Gate Evidence

Understanding governance is a project quality control, not optional mentoring:

- `.planning/STATE.md` and `.planning/UNDERSTANDING.md` agree at every phase transition;
- reached-but-unpassed L2 debt is `0` or `1`, never greater;
- every reached L2 gate has a concise packet containing all fields required by `.planning/UNDERSTANDING-GATES.md`, exactly three reading files, one hands-on experiment, and exactly five user questions;
- L2 passage is supported by the user's explanation, prediction, execution trace, diagnostic reasoning, and experiment result;
- gate packets reference retained implementation evidence; code-reading completeness and agent assertions are not evidence of user ownership;
- L0/L1 and mechanical work remain non-blocking unless a documented conditional L2 trigger occurs.

## Review Checklist

Reviewers verify:

- architectural layer and extension surface are correct;
- ownership/lifetime and error behavior are explicit;
- hot-path impact and allocation/synchronization behavior are known;
- independent oracle and boundary cases exist;
- feature flags/fallbacks fail safely;
- schema/ABI/artifact compatibility impact is declared;
- benchmark methodology is unchanged or versioned;
- docs/evidence are sufficient to reproduce the conclusion.
- understanding status is current, debt remains within the allowed window, and any newly reached L2 packet has been presented proactively.

## Release Gate

V0 requires all blocking CI green, zero unresolved correctness/security-critical findings, supported `.sinf` compatibility tests, Qwen and Gemma clean-machine demos, a verified benchmark evidence bundle, coherent understanding-gate evidence, install/build documentation, license inventory, known limitations, and rollback/recovery guidance.
