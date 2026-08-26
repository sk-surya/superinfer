# Roadmap

## Milestone V0: Qwen Proof, Flash-Next Architecture Proof, Research Loop, Model-Family Validation

The milestone deliberately narrows early work to a correct Qwen3.8 artifact/runtime path. Flash-Next then exercises multi-device placement, heterogeneous PLE residency, MoE, sparse attention, and gated residuals before broad kernel optimization. Performance research follows only after both correctness ladders are explicit; Gemma remains the later model-family portability audit.

| Phase | Name | Depends on | Understanding | Outcome |
|---|---|---|---|---|
| S00 | Foundation and contracts | — | L0/L1 | Buildable skeleton, interfaces, ownership rules, CI baseline |
| S01 | Artifact and IR | S00 | L2 Gate A | Deterministic `.sinf`, three IRs, converter/inspection tooling |
| S02 | `sm120` correctness backend | S01 | L2 Gate B | Validated Physical Plan executes reference-correct small graphs |
| S03 | Qwen3.8-27B end to end | S02 | L1 | Correct Qwen generation from `.sinf` on RTX 5090 |
| S03F | Flash-Next architecture bring-up | S03; S03F-01 research may overlap S03 | L2 architecture packet, deferred under D-014 | Text-only reference-equivalent Flash-Next on 2x RTX 5090 with host-resident PLE |
| S04 | Kernel portfolio and specialization | S03F | L2 Gate C.1–C.3 | Specialized provider portfolio with correct fallbacks |
| S05 | Autoresearch and decode experiments | S04 | L1 / conditional L2 | Correctness-gated experiment/promotion loop; DSpark placed correctly |
| S06 | Reproducible performance proof | S05 | L2 Gate D at entry | First checked RTX 5090 graph and evidence bundle |
| S07 | Gemma 4 26B-A4B extension | S06 | L1 / conditional L2 | Model-family extension without executor model branches |
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

**Understanding:** L2 Gate B fires at S02 completion. Own a one-token transformer execution, Qwen-derived tensor shapes, and the difference between prefill and decode.

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
- `S03-02` — Qwen lowering, artifact-bound quantized projection, attention/GDN/state integration;
- `S03-03` — end-to-end correctness corpus and RTX 5090 acceptance run.

**Requirements:** MOD-001–MOD-004, DEC-001, KER-006, FMT-003, BCK-004, GOV-001

**Exit evidence:** `.sinf` conversion manifest; layer/logit/token differential reports; deterministic generation; acceptance transcript on RTX 5090.

**Invariant:** S03 acceptance is unchanged by S03F. Do not delay the current layer-level and end-to-end Qwen correctness closure to modify multi-device runtime code.

### S03F — Flash-Next Architecture Bring-up

**Goal:** Prove SuperInfer can correctly execute the text path of Flash-Next across 2x RTX 5090 using compiler-owned placement, host-resident PLE/N-gram memory, persistent layer-local MoE experts when capacity permits, QSA, Gated DeltaNet, and gated residual semantics.

**Spec:** [`FLASH-NEXT-DESIGN.md`](FLASH-NEXT-DESIGN.md)

**Understanding:** L2 architecture ownership. Under D-014 this remains a retained study checkpoint and does not automatically block autonomous execution. The packet covers multi-device placement/transfer commands, heterogeneous storage, PLE sparse prefetch, MoE routing/residency, QSA selection vs attention, gated residual, and state continuation.

**Plans:**

- `S03F-01` — **research-only and allowed during S03:** pin model/reference revisions, exact tensor inventory, capacity ledger, quantization/residency options;
- `S03F-02` — multi-device Physical Plan, per-domain memory arenas, contiguous compiler-selected layer partition, explicit peer/staged transfers;
- `S03F-03` — host/mmap-resident quantized PLE lookup with sparse gather and asynchronous pinned-staging H2D;
- `S03F-04` — router/top-k/dispatch/shared+routed grouped experts/combine with typed resident expert bindings;
- `S03F-05` — separate QSA indexing/block selection and sparse attention plus reference-derived gated residual semantics;
- `S03F-06` — complete text-only dual-GPU Physical Plan, state continuation, and reference-equivalent greedy generation.

**Requirements:** MD-001–MD-006, PLE-001–PLE-005, MOE-001–MOE-005, QSA-001–QSA-004, GR-001–GR-003, FN-001–FN-005

**Exit evidence:** pinned architecture/capacity artifacts; deterministic placement plan; peer/staged transfer tests; PLE differential; MoE differential; one GDN and one QSA layer differential; multi-token state continuation; fixed-prompt greedy token golden on 2x RTX 5090.

**Non-goals:** vision, MTP/speculative decode, tensor parallelism, performance-specific fusion, generic arbitrary CPU offload, or silent expert paging.

### S04 — Kernel Portfolio and Specialization

**Goal:** Replace correctness baselines with capability-selected, measured `sm120` candidates while retaining reliable fallback paths for the now-qualified model semantics.

**Understanding:** L2 Gate C repeats at mechanism transitions: C.1 dense GEMV/NVFP4/Tensor Cores, C.2 attention/KV/sparse attention, C.3 fusion/persistent/MoE mechanisms. Each packet covers roofline, memory hierarchy, Tensor versus CUDA cores, occupancy, fusion tradeoffs, likely failures, and a profiler prediction.

**Plans:**

- `S04-01` — GEMM/GEMV, norm/residual, RoPE, embedding/LM-head portfolio;
- `S04-02` — prefill/decode attention, KV update/layout and sparse-attention portfolio;
- `S04-03` — gated FFN, MoE routing/grouped experts, fusion and selector tuning.

**Requirements:** KER-002–KER-006, BCK-002, BCK-004, BEN-002, GOV-002–GOV-004

**Exit evidence:** operation matrix complete; promotion records; representative model correctness unchanged; workload-specific speedups with raw data.

### S05 — Autoresearch and Decode Experiments

**Goal:** Turn kernel and decode research into a repeatable, correctness-first experiment and promotion system.

**Understanding:** L1 for individual experiments; conditional L2 when experimental-validity/noise/statistical methodology changes. S05-03 also produces Gate D's speculative-decoding packet for the S06 transition.

**Plans:**

- `S05-01` — declarative experiment schema, isolation, capture and replay;
- `S05-02` — correctness/statistical gates, promotion and rollback;
- `S05-03` — DecodeStrategy framework, speculative state machine, DSpark experiment track.

**Requirements:** RES-001–RES-005, DEC-001–DEC-004, KER-004, BEN-001, GOV-002–GOV-004

**Exit evidence:** known-good and known-bad experiments gate correctly; replay matches; promoted patch has full evidence; DSpark does not modify attention interfaces.

### S06 — Reproducible Performance Proof

**Goal:** Publish the first honest, reproducible RTX 5090 performance graph, beginning with Qwen3.8-27B and adding Flash-Next only when comparison semantics are defensible.

**Understanding:** L2 Gate D is enforced at S06 entry using the S05-03 packet. Own vanilla speculative decoding, draft/verify/accept economics, why larger `k` can hurt, and why acceptance is workload-dependent.

**Plans:**

- `S06-01` — benchmark runner, environment control, raw schema and baseline adapters;
- `S06-02` — graph/report generation, clean-checkout reproduction, claim audit.

**Requirements:** BEN-001–BEN-005, QUA-002, QUA-003, QUA-005, GOV-001–GOV-004

**Exit evidence:** immutable raw bundle; generated chart; matched-semantics comparison; second-run/clean-checkout reproduction record.

### S07 — Gemma 4 26B-A4B Extension

**Goal:** Prove model-family composability by adding Gemma 4 26B-A4B without model-specific runtime branching.

**Understanding:** L1 architecture audit: inspect every core modification and challenge model-specific branching. Elevate to L2 before implementing any proposed executor/core-boundary change.

**Plans:**

- `S07-01` — Gemma frontend and semantic fixtures;
- `S07-02` — required generic passes/providers and artifact conversion;
- `S07-03` — end-to-end correctness and architecture conformance proof.

**Requirements:** MOD-005, ARCH-005, ARCH-006, KER-002, FMT-004, QUA-004, GOV-002, GOV-004

**Exit evidence:** correct Gemma generation; architecture audit proves no model-name branching; reusable components documented.

### S08 — Hardening and V0 Release

**Goal:** Convert the research prototype into a trustworthy V0 release with compatibility, documentation, and clean-machine verification.

**Understanding:** L0/L1 and generally non-blocking.

**Plans:**

- `S08-01` — negative/security/compatibility hardening and GPU CI tiers;
- `S08-02` — user/developer docs, packaging, support matrix, release rehearsal;
- `S08-03` — milestone evidence audit and V0 publication packet.

**Requirements:** QUA-001–QUA-005, REL-001, REL-002, FMT-002, BCK-003, GOV-001–GOV-004

**Exit evidence:** compatibility matrix; required CI green; clean-machine artifact-to-generation demos; audited benchmark and release bundle.

## Explicit Deferrals

Tensor parallelism, general distributed serving, arbitrary expert paging, production continuous batching, non-`sm120` targets, arbitrary model import, training, vision/MTP Flash-Next support, and production network APIs remain deferred. Dual-RTX-5090 contiguous pipeline placement and PLE-specific host residency are no longer deferred because S03F explicitly owns them.
