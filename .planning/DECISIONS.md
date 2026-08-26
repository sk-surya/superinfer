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

**Status:** Accepted
**Decision:** Qwen3.8-27B is the first vertical proof. Gemma 4 26B-A4B is the second-model architecture test.
**Why:** One model creates focus; the second reveals accidental model-specific coupling.
**Consequence:** Other models remain backlog until the first reproducible Qwen performance graph exists.

## D-008 — DSpark Is Speculative Decoding

**Status:** Accepted
**Decision:** DSpark is explored behind `DecodeStrategy` as speculative decoding, not registered as attention.
**Why:** Its proposal/verification/acceptance and rollback semantics operate at decoding-policy level; attention kernels remain reusable computation providers.
**Consequence:** DSpark experiments may request kernels or workspace but cannot redefine the attention extension contract.

## D-009 — No Hot-Path Dynamism

**Status:** Accepted
**Decision:** Token execution has no heap allocation, model-name or family branch, filesystem access, JIT, or compile-time policy choice.
**Why:** Predictability and specialization are central to latency and readability.
**Consequence:** All buffers, dispatch tables, and schedules are materialized before generation begins.

## D-010 — Evidence Is a Product Artifact

**Status:** Accepted
**Decision:** Raw benchmark results, environment manifests, correctness reports, experiment patches, and promotion decisions are versioned artifacts with schemas.
**Why:** Public performance credibility and autoresearch reproducibility depend on auditable evidence.
**Consequence:** A graph or performance claim without its evidence bundle is incomplete.

## D-011 — C++20/CUDA and Typed Python

**Status:** Accepted
**Decision:** Compiler/runtime code uses modern C++20/CUDA; conversion, research, and reporting use typed Python 3.12+.
**Why:** This keeps device code close to the hardware while using Python where orchestration velocity matters.
**Consequence:** Language crossings use explicit schemas/files or narrow bindings, never hidden shared state.

## D-012 — Explicit Ownership

**Status:** Accepted
**Decision:** Buffers, streams, events, plans, and artifact mappings each have one visible owner; views are non-owning and named as such.
**Why:** GPU lifecycle ambiguity creates correctness and synchronization defects that are difficult to reproduce.
**Consequence:** APIs document lifetime and thread-safety contracts and are tested with sanitizers.

## D-013 — Fixed Sectioned `.sinf` Encoding for V0

**Status:** Accepted
**Decision:** V0 uses a fixed little-endian section directory with canonical JSON/text metadata,
8-byte alignment, FNV-1a per-section checksums, and an explicit integrity table. C++ and Python
implementations share the byte layout without a generated schema dependency.
**Why:** The format must be deterministic, inspectable, CPU-only, fuzzable, and easy to validate
before any storage/device materialization. A small explicit encoding keeps those properties visible
while the semantic and physical schemas are still evolving.
**Consequence:** Major/minor compatibility rules and bounds are part of the checked-in format
document; FNV-1a is integrity protection, not a cryptographic authenticity claim.

## D-014 — User-directed autonomous understanding-gate execution

**Status:** Accepted
**Decision:** Retain every understanding gate, packet, exercise, and branch checkpoint as durable
study evidence, but do not pause implementation for user passage of an L2 gate. Continue through the
approved roadmap autonomously and present each gate for later study.
**Why:** The user explicitly requested implementation velocity while preserving the gates as an
independent learning trail and checkpoint history.
**Consequence:** Phase and gate state records may show reached-but-unpassed gates beyond the original
one-gate debt window. This is an explicit workflow override, not evidence that the user has passed a
gate; no gate is marked passed without the user's own answers and experiment evidence.
