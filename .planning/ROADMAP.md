# Roadmap

## Milestone V0: Qwen Proof, Research Loop, Second-Model Validation

The milestone deliberately narrows early work to a correct Qwen3.8 artifact/runtime path, then optimizes and proves repeatability before using Gemma 4 to validate extensibility.

| Phase | Name | Depends on | Understanding | Outcome |
|---|---|---|---|---|
| S00 | Foundation and contracts | — | L0/L1 | Buildable skeleton, interfaces, ownership rules, CI baseline |
| S01 | Artifact and IR | S00 | L2 Gate A | Deterministic `.sinf`, three IRs, converter/inspection tooling |
| S02 | `sm120` correctness backend | S01 | L2 Gate B | Validated Physical Plan executes reference-correct small graphs |
| S03 | Qwen3.8-27B end to end | S02 | L1 | Correct Qwen generation from `.sinf` on RTX 5090 |
| S04 | Kernel portfolio and specialization | S03 | L2 Gate C.1–C.3 | Specialized provider portfolio with correct fallbacks |
| S05 | Autoresearch and decode experiments | S04 | L1 / conditional L2 | Correctness-gated experiment/promotion loop; DSpark placed correctly |
| S06 | Reproducible performance proof | S05 | L2 Gate D at entry | First checked RTX 5090 graph and evidence bundle |
| S07 | Gemma 4 26B-A4B extension | S06 | L1 / conditional L2 | Second model added without executor edits |
| S08 | Hardening and V0 release | S07 | L0/L1 | Compatibility, docs, release rehearsal, public V0 packet |

### S00 — Foundation and Contracts

**Goal:** Establish a readable, testable C++20/CUDA/Python repository and freeze the architectural contracts before model code lands.

**Understanding:** L0/L1, non-blocking. The user should explain host/device execution, streams/events, synchronization, GPU hierarchy, and why asynchronous launch invalidates naive CPU timing.

**Plans:**

- `S00-01` — repository skeleton, dependency rules, interfaces, typed errors, build tooling;
- `S00-02` — CPU CI, test harness, reference-oracle scaffolding, developer workflows.

**Requirements:** ARCH-001, ARCH-004, ARCH-005, ARCH-008, QUA-001, QUA-003, GOV-001, GOV-004

**Exit evidence:** clean build/test on CPU; interface conformance tests; dependency-boundary checks; documented ownership.

### S01 — Artifact and IR

**Goal:** Convert pinned model inputs into a deterministic, inspectable `.sinf` that contains verified semantic/lowered/physical representations and tensor payload metadata.

**Understanding:** L2 Gate A fires at S01 completion. Own `HF -> ModelFrontend -> Semantic IR -> GraphPasses -> Lowered IR -> kernel selection/memory planning -> Physical Plan -> runtime`, including which architecture decisions belong at each boundary.

**Plans:**

- `S01-01` — Semantic IR and verifier;
- `S01-02` — Lowered IR, pass manager, Physical Plan schema;
- `S01-03` — `.sinf` writer/reader, StoragePolicy, converter and inspector CLI.

**Requirements:** ARCH-001, ARCH-002, ARCH-003, ARCH-007, FMT-001–FMT-006, MOD-002, MOD-003, GOV-002–GOV-004

**Exit evidence:** golden IR dumps; deterministic artifact hash; corrupt-artifact/fuzz tests; CPU-only inspect/validate command.

### S02 — `sm120` Correctness Backend

**Goal:** Produce and execute validated `sm_120a` Physical Plans with baseline kernels and explicit memory/stream ownership.

**Understanding:** L2 Gate B fires at S02 completion. Own a one-token transformer execution, Qwen-derived tensor shapes, and the difference between prefill and decode; the pinned Qwen config may be prepared for this packet without pulling model implementation into S02.

**Plans:**

- `S02-01` — target profile, memory/workspace planner, compiler specialization;
- `S02-02` — minimal executor, baseline KernelProvider, CPU/reference differential path;
- `S02-03` — GPU validation, lifecycle, allocation and sanitizer tests.

**Requirements:** BCK-001–BCK-006, KER-001, KER-002, KER-005, ARCH-004, GOV-002–GOV-004

**Exit evidence:** small graph executes on GPU and matches reference; adversarial plans rejected; decode trace shows zero hot-path allocation/model branching.

### S03 — Qwen3.8-27B End to End

**Goal:** Make Qwen3.8-27B the first complete model: convert, load, prefill, decode, and match the pinned reference.

**Understanding:** L1, non-blocking. Trace one weight end to end: Hugging Face tensor -> quantize/pack -> `.sinf` -> load/materialize -> selected GPU operation.

**Plans:**

- `S03-01` — Qwen frontend, tensor/config/tokenizer mapping;
- `S03-02` — Qwen lowering, MoE/attention/KV/sampling baseline integration;
- `S03-03` — end-to-end correctness corpus and RTX 5090 acceptance run.

**Requirements:** MOD-001–MOD-004, DEC-001, KER-006, FMT-003, BCK-004, GOV-001

**Exit evidence:** `.sinf` conversion manifest; logits/token differential reports; deterministic and long-context generation; acceptance transcript on RTX 5090.

### S04 — Kernel Portfolio and Specialization

**Goal:** Replace correctness baselines with capability-selected, measured `sm120` candidates while retaining reliable fallback paths.

**Understanding:** L2 Gate C repeats at mechanism transitions: C.1 dense GEMV/NVFP4/Tensor Cores, C.2 attention/KV, C.3 fusion/persistent mechanisms. Each packet covers roofline, memory hierarchy, Tensor versus CUDA cores, occupancy, fusion tradeoffs, likely failures, and a profiler prediction.

**Plans:**

- `S04-01` — GEMM/GEMV, norm/residual, RoPE, embedding/LM-head portfolio;
- `S04-02` — prefill/decode attention, KV update/layout portfolio;
- `S04-03` — gated FFN, MoE routing/grouped experts, fusion and selector tuning.

**Requirements:** KER-002–KER-006, BCK-002, BCK-004, BEN-002, GOV-002–GOV-004

**Exit evidence:** operation matrix complete; promotion records; representative Qwen correctness unchanged; workload-specific speedups with raw data.

### S05 — Autoresearch and Decode Experiments

**Goal:** Turn kernel and decode research into a repeatable, correctness-first experiment and promotion system.

**Understanding:** L1 for individual experiments; conditional L2 when experimental-validity/noise/statistical methodology changes. The user owns why a result is eligible, not every candidate mutation. S05-03 also produces Gate D's speculative-decoding packet for the S06 transition.

**Plans:**

- `S05-01` — declarative experiment schema, isolation, capture and replay;
- `S05-02` — correctness/statistical gates, promotion and rollback;
- `S05-03` — DecodeStrategy framework, speculative state machine, DSpark experiment track.

**Requirements:** RES-001–RES-005, DEC-001–DEC-004, KER-004, BEN-001, GOV-002–GOV-004

**Exit evidence:** known-good and known-bad experiments gate correctly; replay matches; promoted patch has full evidence; DSpark does not modify attention interfaces.

### S06 — Reproducible Performance Proof

**Goal:** Publish the first honest, reproducible RTX 5090 performance graph for Qwen3.8-27B.

**Understanding:** L2 Gate D is enforced at S06 entry using the S05-03 packet. Own vanilla speculative decoding, draft/verify/accept economics, why larger `k` can hurt, and why acceptance is workload-dependent. S06 benchmark mechanics proceed only within the recorded debt window.

**Plans:**

- `S06-01` — benchmark runner, environment control, raw schema and baseline adapters;
- `S06-02` — graph/report generation, clean-checkout reproduction, claim audit.

**Requirements:** BEN-001–BEN-005, QUA-002, QUA-003, QUA-005, GOV-001–GOV-004

**Exit evidence:** immutable raw bundle; generated chart; matched-semantics comparison; second-run/clean-checkout reproduction record.

### S07 — Gemma 4 26B-A4B Extension

**Goal:** Prove model-family composability by adding Gemma 4 26B-A4B without changing the minimal executor.

**Understanding:** L1 architecture audit: inspect every core modification and challenge model-specific branching. Elevate to L2 before implementing any proposed executor or core-boundary change.

**Plans:**

- `S07-01` — Gemma frontend and semantic fixtures;
- `S07-02` — required generic passes/providers and artifact conversion;
- `S07-03` — end-to-end correctness and architecture conformance proof.

**Requirements:** MOD-005, ARCH-005, ARCH-006, KER-002, FMT-004, QUA-004, GOV-002, GOV-004

**Exit evidence:** correct Gemma generation; executor tree hash/diff proves no model-specific edits; reusable components documented.

### S08 — Hardening and V0 Release

**Goal:** Convert the research prototype into a trustworthy V0 release with compatibility, documentation, and clean-machine verification.

**Understanding:** L0/L1 and generally non-blocking. Release evidence audits packet links, ledger/state consistency, and remaining unknowns without requiring mastery of productization mechanics.

**Plans:**

- `S08-01` — negative/security/compatibility hardening and GPU CI tiers;
- `S08-02` — user/developer docs, packaging, support matrix, release rehearsal;
- `S08-03` — milestone evidence audit and V0 publication packet.

**Requirements:** QUA-001–QUA-005, REL-001, REL-002, FMT-002, BCK-003, GOV-001–GOV-004

**Exit evidence:** compatibility matrix; required CI green; clean-machine artifact-to-generation demo; audited benchmark and release bundle.

## Explicit Deferrals

Multi-GPU, server scheduling, continuous batching at production scale, non-`sm120` targets, arbitrary model import, training, and production network APIs require a later milestone and must not delay S00–S06.
