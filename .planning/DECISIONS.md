# Architecture and Product Decisions

## D-001 — AOT Compiler and Minimal Runtime

**Status:** Accepted
**Decision:** SuperInfer is an ahead-of-time model compiler plus a minimal physical executor, not a general dynamic graph runtime. Composition happens before runtime; runtime executes a validated Physical Plan.
**Why:** The model, dimensions, quantization, target GPU, and decoding policy are known and should become compile-time facts.
**Consequence:** Broad compatibility and runtime dynamism are intentionally deferred.

## D-002 — Three Representations

**Status:** Accepted
**Decision:** Use Semantic IR, Lowered IR, and Physical Plan as the only durable representations.
**Why:** They separate model meaning, target-aware transformation, and direct execution without an unbounded ladder of intermediate dialects.
**Consequence:** Temporary pass-local analysis structures are allowed but cannot become implicit fourth IRs.

## D-003 — Five Extension Surfaces

**Status:** Accepted
**Decision:** Extensibility is limited to `ModelFrontend`, `GraphPass`, `KernelProvider`, `DecodeStrategy`, and `StoragePolicy`.
**Why:** These map to independent axes of model semantics, compilation, device implementation, token policy, and persistence.
**Consequence:** Adding an extension surface requires a superseding ADR and evidence that existing surfaces cannot express the need.

## D-004 — Target `sm_120a` First

**Status:** Accepted
**Decision:** The first and optimized backend targets RTX 5090 (`sm_120a`).
**Why:** Hardware specificity is the project advantage and keeps the initial performance problem bounded.
**Consequence:** Backend interfaces must not claim portability that is untested; other targets are later milestones.

## D-005 — `.sinf` Is the Deployment Contract

**Status:** Accepted
**Decision:** `.sinf` stores a versioned manifest, tensors, tokenizer/config metadata, compiled Physical Plan data, integrity metadata, and provenance.
**Why:** Deployment should not depend on reconstructing Hugging Face/Python state.
**Consequence:** Readers validate completely before allocation/execution and schema compatibility is tested.

## D-006 — Correctness Before Speed

**Status:** Accepted
**Decision:** Every optimized kernel and autoresearch candidate must pass a trusted differential path before performance is considered.
**Why:** Fast wrong tokens are not progress, and research automation amplifies mistakes.
**Consequence:** Promotion gates fail closed; baselines remain available as fallbacks.

## D-007 — Qwen First, Gemma Second

**Status:** Superseded in ordering by D-017
**Decision:** Qwen3.8-27B is the first vertical proof. Gemma 4 26B-A4B was originally designated the second-model architecture test.
**Why:** One model creates focus; the second reveals accidental model-specific coupling.
**Consequence:** D-017 preserves Qwen as the first proof but inserts Flash-Next as the second flagship architecture proof. Gemma remains the later model-family portability audit.

## D-008 — DSpark Is Speculative Decoding

**Status:** Accepted
**Decision:** DSpark is explored behind `DecodeStrategy` as speculative decoding, not registered as attention.
**Why:** Its proposal/verification/acceptance and rollback semantics operate at decoding-policy level; attention kernels remain reusable computation providers.
**Consequence:** DSpark experiments may request kernels or workspace but cannot redefine the attention extension contract.

## D-009 — No Hot-Path Dynamism

**Status:** Accepted
**Decision:** Token execution has no heap allocation, model-name or family branch, filesystem access, JIT, or compile-time policy choice.
**Why:** Predictability and specialization are central to latency and readability.
**Consequence:** All buffers, dispatch tables, schedules, placement and transfer topology are materialized before generation begins; sparse PLE gathers may use predeclared host/pinned resources without allocation.

## D-010 — Evidence Is a Product Artifact

**Status:** Accepted
**Decision:** Raw benchmark results, environment manifests, correctness reports, experiment patches, placement/residency manifests and promotion decisions are versioned artifacts with schemas.
**Why:** Public performance credibility and autoresearch reproducibility depend on auditable evidence.
**Consequence:** A graph or performance claim without its evidence bundle is incomplete.

## D-011 — C++20/CUDA and Typed Python

**Status:** Accepted
**Decision:** Compiler/runtime code uses modern C++20/CUDA; conversion, research, and reporting use typed Python 3.12+.
**Why:** This keeps device code close to the hardware while using Python where orchestration velocity matters.
**Consequence:** Language crossings use explicit schemas/files or narrow bindings, never hidden shared state.

## D-012 — Explicit Ownership

**Status:** Accepted
**Decision:** Buffers, streams, events, plans, artifact mappings, placement domains and staging buffers each have one visible owner; views are non-owning and named as such.
**Why:** GPU lifecycle ambiguity creates correctness and synchronization defects that are difficult to reproduce.
**Consequence:** APIs document lifetime and thread-safety contracts and are tested with sanitizers.

## D-013 — Fixed Sectioned `.sinf` Encoding for V0

**Status:** Accepted
**Decision:** V0 uses a fixed little-endian section directory with canonical JSON/text metadata, 8-byte alignment, FNV-1a per-section checksums, and an explicit integrity table. C++ and Python implementations share the byte layout without a generated schema dependency.
**Why:** The format must be deterministic, inspectable, CPU-only, fuzzable, and easy to validate before any storage/device materialization. A small explicit encoding keeps those properties visible while the semantic and physical schemas are still evolving.
**Consequence:** Major/minor compatibility rules and bounds are part of the checked-in format document; FNV-1a is integrity protection, not a cryptographic authenticity claim.

## D-014 — User-directed autonomous understanding-gate execution

**Status:** Accepted
**Decision:** Retain every understanding gate, packet, exercise, and branch checkpoint as durable study evidence, but do not pause implementation for user passage of an L2 gate. Continue through the approved roadmap autonomously and present each gate for later study.
**Why:** The user explicitly requested implementation velocity while preserving the gates as an independent learning trail and checkpoint history.
**Consequence:** Phase and gate state records may show reached-but-unpassed gates beyond the original one-gate debt window. This is an explicit workflow override, not evidence that the user has passed a gate; no gate is marked passed without the user's own answers and experiment evidence.

## D-015 — Canonical gated-delta attention semantic operation

**Status:** Accepted
**Decision:** Represent Qwen3.8’s Gated DeltaNet/linear-attention block with the generic Semantic IR operation `gated_delta_attention`, carrying key/value head dimensions and convolution-kernel dimension as semantic attributes. It is not a CUDA kernel or provider selection.
**Why:** Qwen3.8 alternates linear-attention and full-attention layers; mapping the former to ordinary multi-head attention would erase state-transition and projection semantics before lowering.
**Consequence:** Lowering and baseline execution must add an independent reference contract before this operation can be selected for a physical plan. The executor remains model-agnostic.

## D-016 — V0 `.sinf` artifact bound covers the pinned Qwen payload

**Status:** Accepted
**Decision:** Set the defensive V0 artifact-size limit to 32 GiB in both Python and C++ readers. Large artifacts must be inspected section-by-section without materializing the payload in host memory.
**Why:** The authenticated pinned Qwen3.8 payload artifact exceeds the previous 16-GiB reader bound. The larger bound remains bounded while fitting the declared Qwen V0 envelope.
**Consequence:** Flash-Next may require a separately justified artifact/sectioning update during S03F-01; do not silently increase global bounds merely because the source repository is larger.

## D-017 — Flash-Next is the second flagship architecture proof

**Status:** Accepted
**Supersedes:** D-007 ordering only
**Decision:** Finish S03 Qwen3.8 correctness unchanged, then execute S03F Flash-Next bring-up before S04 kernel optimization. S03F-01 contract/capacity research may run in parallel with the tail of S03; S03F-02 through S03F-06 runtime work is blocked until S03 closes.
**Why:** Flash-Next directly exercises extension points SuperInfer was designed to support—heterogeneous storage, stateful execution, MoE, sparse attention and multi-device specialization—while Qwen provides the simpler debugging ladder needed to validate the compiler/runtime first.
**Consequence:** Gemma moves from “second model” to later model-family portability audit. Vision and MTP remain excluded from initial Flash-Next correctness.

## D-018 — Initial Flash-Next placement uses contiguous layers and first-class PLE residency

**Status:** Accepted
**Decision:** Initial 2x5090 support uses compiler-selected contiguous layer partitioning from actual packed bytes under independent device budgets/headroom. Cross-domain movement is represented as explicit Physical Plan transfer commands. PLE/N-gram tables are a first-class read-only host/mmap StoragePolicy path with sparse gather and preallocated asynchronous staging, not generic CPU offload.
**Why:** Pipeline partitioning minimizes PCIe communication relative to expert weights and keeps execution deterministic. PLE naturally benefits from host residency because only selected vectors need transfer.
**Consequence:** Tensor parallelism, expert parallelism and generic arbitrary offload remain deferred. A hard-coded 24/24 split is forbidden. Expert residency is decided separately by S03F-01 exact capacity/quality evidence; no silent expert paging is permitted.

## D-019 — Flash-Next residency remains undecided pending source evidence

**Status:** Accepted
**Decision:** S03F-01 cannot select a full-expert residency or quantization recipe because the exact Flash-Next artifact and pinned reference revision are unavailable. Until that evidence is supplied, `full_expert_residency_feasible` is unknown, no quality-preserving quantization claim is made, and S03F implementation may not add expert staging, caching, or paging.
**Why:** The checked-in evidence contains no exact Flash-Next safetensors headers, packed byte ranges, or reference evaluation. Parameter-count estimates and near-name model artifacts do not satisfy FN-001/FN-002.
**Consequence:** S03F-02 remains engineering-blocked after the existing S03 dependency; the next research action is to provide the exact source/reference inputs and regenerate the canonical ledgers.
