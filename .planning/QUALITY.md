# Quality Strategy

## Principles

- Tests prove contracts and failure behavior, not just line coverage.
- CPU-only development remains productive; GPU lanes add device truth.
- Optimized paths are never the oracle for themselves.
- Evidence and reproduction are part of definition of done.
- Flaky tests are defects and are quarantined only with an owner, issue, and expiration.
- Multi-device and heterogeneous-residency behavior must be explicit enough to test without relying on timing luck or hidden runtime policy.

## Test Pyramid

### Static and Build

- C++/CUDA formatting, warnings-as-errors for owned code, static analysis on supported subset;
- Python formatting, lint, type checking, package/build validation;
- dependency-boundary and forbidden-import tests;
- generated-schema freshness and deterministic-codegen checks.

### Unit and Property Tests

- IR constructors/verifiers, pass preconditions/invalidation, shape/dtype algebra;
- artifact offset/overflow/alignment/version validation;
- memory planner liveness/non-overlap/resource bounds including independent placement-domain budgets;
- kernel capability matching and fallback;
- transfer-command endpoint/topology/resource validation;
- PLE index/storage metadata contracts;
- MoE routing/top-k/dispatch/combine semantics;
- sparse-attention selection/index contracts;
- gated-residual and recurrent/KV state transitions;
- decode state machines including rollback/KV consistency;
- experiment schemas, statistics, and promotion rules.

Property tests use persisted failure seeds and bounded runtime.

### Golden and Compatibility Tests

- canonical Semantic/Lowered/Physical dumps, including placement/transfer records;
- `.sinf` byte/manifest fixtures for supported versions and host-resident sections;
- tokenizer/config/tensor mappings;
- deterministic device-placement/residency manifests;
- benchmark/report schemas and generated tables;
- supported reader/writer compatibility matrix.

Golden updates require an explanation and semantic diff review.

### Differential Tests

- small CPU/reference executor vs trusted tensor/reference implementation;
- each GPU kernel vs independent reference across random, boundary, and model shapes;
- Qwen full-attention and Gated-Delta layers through real artifact-bound CUDA paths;
- Flash-Next PLE indices/vectors, MoE route/expert/combine, QSA selection/attention, gated residual, one GDN layer, one QSA layer, cross-device boundary and state continuation;
- full-model logits at selected layers/positions and final token sequences;
- baseline vs specialized plan equivalence within documented numerical contracts.

A project-local reimplementation matching another project-local implementation is not sufficient model evidence. Model qualification must ultimately compare against the pinned external/reference semantics at selected intermediates.

### Fuzz and Negative Tests

- artifact and schema parsers: truncation, corrupt section tables, overflow, aliasing, unknown capability/version, checksum failure;
- frontend config/tensor schemas: missing/extra/wrong-shaped tensors and unsafe values;
- Physical Plan verifier: cycles, invalid dependencies, bad placements/device ordinals, illegal transfer endpoints, unsupported kernels, resource-bound violations;
- PLE metadata: invalid offsets/quantization/host mappings;
- MoE/QSA/gated-residual authored attributes and shape failures;
- CLI inputs and experiment manifests.

### GPU Lifecycle and Sanitizer Tests

- compute-sanitizer/memcheck/racecheck where supported;
- stream/event ordering, session teardown after error, repeated load/unload;
- per-device OOM/resource-bound rejection and no partial launch;
- explicit peer-transfer and pinned-host fallback ordering;
- hot-path allocation/synchronization/unplanned-transfer trace;
- PLE prefetch producer/consumer ordering;
- long-context KV/recurrent state bounds and wrap/rollback/continuation cases.

## CI Lanes

| Lane | Trigger | Hardware | Content | Blocking |
|---|---|---|---|---|
| Fast | every PR | CPU | format, lint, type, build, focused unit | yes |
| Full CPU | every PR/main | CPU | all unit/property/golden/fuzz-smoke/integration | yes |
| GPU smoke | eligible PR | 1x RTX 5090 | load, tiny plan, small kernel set | yes once infrastructure is stable |
| GPU correctness | main/nightly | 1x RTX 5090 | kernel matrix and Qwen differentials | yes for release |
| Dual-GPU correctness | S03F changes + main/nightly | 2x RTX 5090 | placement/transfers, PLE staging, Flash-Next layer/state/generation differentials | yes once S03F support lands |
| GPU sanitizer | scheduled | 1x/2x RTX 5090 as applicable | lifecycle/bounds/race/transfer suites | yes for release |
| Benchmark | controlled/manual + scheduled | dedicated RTX 5090 topology declared by manifest | fixed manifests, regression evaluation | claim/release gate |

Benchmark jobs never share the measured GPU(s) and record environment/topology validity before accepting samples.

## Numerical Contracts

Each operation documents input/output dtype, accumulation dtype, deterministic guarantees, exceptional-value behavior, and absolute/relative/ULP or task-level tolerance. Tolerances derive from reference analysis and representative error distributions. Token agreement is required for deterministic acceptance prompts unless an explicitly approved logits-based criterion explains divergence.

Quantized Flash-Next differentials must distinguish source/reference quantization differences from runtime arithmetic differences. S03F-01 residency recipes cannot claim quality equivalence from byte-size projections alone.

## S03F Correctness Ladder

S03F cannot be declared complete by a single final-token match. Evidence is staged:

1. pinned model/reference + exact capacity ledger;
2. multi-device placement/transfer fixture;
3. PLE index/vector/injection differential;
4. MoE route/top-k/expert/combine differential;
5. QSA index/select + sparse-attention differential;
6. gated-residual continuation differential;
7. one complete GDN layer and one complete QSA layer through artifact-bound CUDA;
8. cross-device multi-token state continuation;
9. full 48-layer fixed-prompt greedy generation.

Each later gate retains enough selected intermediates to localize a mismatch without bisecting the entire model blindly.

## Coverage Expectations

- 100% of public parsers/verifiers and error branches have tests;
- 100% of registered kernel capability envelopes have at least one positive and negative case;
- every placement/transfer command variant has positive, rejection and lifecycle coverage;
- every requirement has at least one named automated or manual evidence producer;
- line coverage is reported for trend/debugging but never substitutes for the above contract coverage.

## Understanding-Gate Evidence

Understanding governance is a project quality control, not optional mentoring:

- `.planning/STATE.md` and `.planning/UNDERSTANDING.md` agree at phase transitions;
- the normal debt policy remains defined in `.planning/UNDERSTANDING-GATES.md`; D-014 may explicitly override blocking without implying user passage;
- every reached L2 gate has a concise packet containing all fields required by `.planning/UNDERSTANDING-GATES.md`, exactly three reading files, one hands-on experiment, and exactly five user questions;
- S03F's architecture packet covers placement/transfer, heterogeneous PLE residency, MoE, QSA/gated residual and state continuation;
- gate packets reference retained implementation evidence; code-reading completeness and agent assertions are not evidence of user ownership.

## Review Checklist

Reviewers verify:

- architectural layer and extension surface are correct;
- ownership/lifetime and error behavior are explicit;
- placement/residency/transfer behavior is compiler-authored and inspectable;
- hot-path impact and allocation/synchronization behavior are known;
- independent oracle and boundary cases exist;
- feature flags/fallbacks fail safely;
- schema/ABI/artifact compatibility impact is declared;
- model frontends do not choose kernel IDs or device ordinals;
- no expert or full PLE table movement is hidden inside runtime/kernel helpers;
- benchmark methodology is unchanged or versioned;
- docs/evidence are sufficient to reproduce the conclusion;
- understanding status is current and no gate is marked passed on the user's behalf.

## Release Gate

V0 requires all blocking CI green, zero unresolved correctness/security-critical findings, supported `.sinf` compatibility tests, a clean-machine Qwen single-5090 demo, the declared Flash-Next dual-5090 text correctness demo, the later model-family architecture audit, a verified benchmark evidence bundle, coherent understanding-gate evidence, install/build documentation, license inventory, known limitations, and rollback/recovery guidance.
