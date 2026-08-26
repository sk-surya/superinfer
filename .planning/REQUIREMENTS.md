# Requirements

Requirements are evidence-oriented. A requirement is complete only when its named behavior exists and the specified evidence is retained.

## Architecture and Extensibility

- **ARCH-001** — The pipeline has exactly three durable representations: Semantic IR, Lowered IR, and Physical Plan. Evidence: type/API tests and architecture conformance test.
- **ARCH-002** — Semantic IR expresses model meaning without CUDA launch details or storage offsets. Evidence: dependency-boundary test.
- **ARCH-003** — Lowered IR contains target-aware tensor/layout/fusion decisions while remaining inspectable and serializable. Evidence: golden lowering dumps.
- **ARCH-004** — Physical Plan is immutable after validation and directly executable without model-family inspection. Evidence: executor dependency test and runtime trace.
- **ARCH-005** — Supported extension surfaces are `ModelFrontend`, `GraphPass`, `KernelProvider`, `DecodeStrategy`, and `StoragePolicy`. Evidence: interface contract tests and example plugins/providers.
- **ARCH-006** — New model support does not require model-name dispatch in the physical executor. Evidence: architecture diff/conformance tests.
- **ARCH-007** — Graph pass ordering, preconditions, effects, and invalidation are explicit and validated. Evidence: pass-manager unit/property tests.
- **ARCH-008** — Components expose explicit ownership, lifetimes, typed errors, and deterministic behavior. Evidence: API review checklist and sanitizers.

## `.sinf` Artifact and Conversion

- **FMT-001** — `.sinf` has a versioned header, manifest, aligned tensor payloads, tokenizer/config metadata, Physical Plan section, and integrity table. Evidence: binary format spec plus golden files.
- **FMT-002** — Reader rejects truncation, corruption, unknown required capabilities, incompatible versions, and invalid offsets before device allocation. Evidence: negative/fuzz tests.
- **FMT-003** — Same pinned inputs and converter version produce byte-identical artifacts. Evidence: reproducibility CI test.
- **FMT-004** — Source model identity, revision, licenses, conversion options, quantization, tensor hashes, and toolchain identity are recorded. Evidence: manifest inspection test.
- **FMT-005** — StoragePolicy supports aligned/memory-mapped host access and planned device materialization without leaking format details to kernels. Evidence: storage contract tests.
- **FMT-006** — A CLI can inspect, validate, and explain a `.sinf` artifact without a GPU. Evidence: CLI integration tests.

## Model Frontends and Semantics

- **MOD-001** — Qwen3.8-27B is the first complete frontend and first correctness proof. Evidence: config/tensor mapping fixtures and reference comparisons.
- **MOD-002** — Frontends validate configs and tensor schemas with actionable errors; no best-effort silent defaults. Evidence: malformed-input tests.
- **MOD-003** — Semantic operations cover embeddings, normalization, RoPE, attention variants, gated dense FFN, MoE routing/experts, residuals, LM head, and optional auxiliary heads needed by target models. Evidence: IR verifier tests.
- **MOD-004** — Tokenizer and chat-template identity are preserved and reference-compatible. Evidence: tokenizer golden tests.
- **MOD-005** — Gemma 4 26B-A4B remains the later model-family audit after the Flash-Next architecture proof. Evidence: end-to-end artifact/generation tests and architecture audit.

## `sm120` Backend and Runtime

- **BCK-001** — The first backend targets RTX 5090 `sm_120a` and rejects incompatible hardware/artifacts clearly. Evidence: capability probe tests.
- **BCK-002** — Compilation specializes layouts, workspace, kernel choices, launch parameters, placement and decode schedule for the exact model and target profile. Evidence: Physical Plan dumps.
- **BCK-003** — Runtime validates the complete plan and resource bounds before execution. Evidence: adversarial-plan tests.
- **BCK-004** — Decode hot path performs no heap allocation, model-family branching, filesystem access, JIT compilation, or implicit global synchronization. Evidence: allocation/trace regression test.
- **BCK-005** — Buffer/workspace ownership and CUDA stream/event lifetimes are explicit and deterministic. Evidence: compute-sanitizer and lifecycle tests.
- **BCK-006** — CPU/reference executor or oracle path can execute small graphs for differential testing. Evidence: randomized IR tests.

## Multi-Device Placement

- **MD-001** — Physical buffers and executable commands carry explicit placement domains/device ordinals; model frontends remain placement-agnostic. Evidence: Physical Plan schema/verifier tests and dependency checks.
- **MD-002** — Multi-device memory planning enforces independent per-device budgets and keeps host/pinned/mmap allocations outside device arena accounting. Evidence: unit/property tests.
- **MD-003** — Inter-device and host/device data movement is represented by explicit validated transfer commands, never hidden inside model kernels. Evidence: plan dumps and invalid-transfer tests.
- **MD-004** — Initial dual-5090 placement uses a deterministic contiguous layer partition derived from actual packed bytes and configured headroom, not a hard-coded 24/24 split. Evidence: uneven-layer partition fixtures and generated placement manifest.
- **MD-005** — CUDA runtime executes peer copies asynchronously when supported and has an explicit pinned-host staged fallback; both preserve dependencies and numerical results. Evidence: dual-GPU fixture and CUDA trace.
- **MD-006** — Single-device Qwen3.8 behavior remains unchanged by multi-device support. Evidence: S03 regression corpus and plan fingerprint/behavior audit.

## PLE / N-gram Memory

- **PLE-001** — PLE/N-gram indexing, lookup and combination are explicit model-independent semantic operations whose exact behavior is pinned to reference evidence. Evidence: semantic verifier tests and index differential fixtures.
- **PLE-002** — PLE tables may be independently quantized and remain read-only in host/mmap storage without device materialization at artifact load. Evidence: artifact/storage-policy tests.
- **PLE-003** — Runtime transfers only requested PLE vectors through preallocated pinned staging/device buffers; full-table transfer is forbidden in steady-state execution. Evidence: transfer-byte trace and allocation regression.
- **PLE-004** — Sparse gather/dequantization and model injection match the pinned reference for fixed and continuation token histories. Evidence: PLE differential report.
- **PLE-005** — PLE prefetch uses explicit transfer-stream/event dependencies and cannot race the consuming GPU command. Evidence: CUDA ordering test and sanitizer run.

## MoE

- **MOE-001** — Routing, top-k, dispatch, routed-expert compute, shared-expert compute and combine remain separately representable and testable generic semantics. Evidence: IR verifier/lowering fixtures.
- **MOE-002** — Flash-Next expert NVFP4 payloads and scale sidecars bind deterministically to layer/expert/device physical buffers. Evidence: artifact-to-plan binding report.
- **MOE-003** — Router logits, selected expert IDs/order, routing weights, routed/shared outputs and combine result match the pinned reference. Evidence: layer differential report.
- **MOE-004** — Initial expert placement is layer-local and persistent when the accepted S03F-01 capacity recipe proves it feasible; runtime may not silently stage/page experts. Evidence: placement plan and execution transfer trace.
- **MOE-005** — If acceptable full expert residency is not feasible, expert caching/staging requires a superseding capacity ADR with explicit correctness/performance semantics before implementation. Evidence: decision record.

## QSA / Sparse Attention

- **QSA-001** — QSA indexing/block selection is semantically distinct from sparse attention execution. Evidence: IR contracts and dependency graph fixture.
- **QSA-002** — Indexer inputs, budgets/compression/block semantics and selected blocks match the pinned reference. Evidence: indexer differential report.
- **QSA-003** — Sparse attention over selected blocks matches a trusted reference across causal/boundary/continuation cases. Evidence: CPU/GPU differential tests.
- **QSA-004** — QSA providers are selected by generic operation/type/shape/layout attributes and never by Flash-Next model name. Evidence: selector/architecture tests.

## Gated Residual

- **GR-001** — Four-branch gated residual read/write semantics are derived from and pinned to the reference implementation before the final generic IR contract is frozen. Evidence: contract artifact with revision/hash.
- **GR-002** — Residual branch state is explicit through lowering/physical execution and survives multi-step continuation. Evidence: state transition tests.
- **GR-003** — Gated residual intermediates and final layer output match the pinned reference for both GDN and QSA layer families. Evidence: layer differential reports.

## Flash-Next End-to-End

- **FN-001** — S03F-01 produces an exact packed-byte inventory and per-category memory ledger covering experts, shared experts, PLE, non-expert text weights, router/indexer, embedding/LM head, state/workspace, vision and MTP. Evidence: canonical JSON ledger.
- **FN-002** — Residency/quantization candidates report exact per-device/host bytes, configured headroom, and unresolved quality assumptions; parameter-count estimates alone cannot close capacity. Evidence: options artifact and ADR.
- **FN-003** — One GDN layer and one QSA layer execute through real `.sinf`-bound CUDA paths and match selected reference intermediates. Evidence: layer differential reports.
- **FN-004** — Multi-token continuation preserves GDN/KV/gated-residual state and validates the cross-device activation at the partition boundary. Evidence: dual-GPU continuation report.
- **FN-005** — Text-only Flash-Next produces a fixed reference-equivalent greedy token sequence on 2x RTX 5090 from a deterministic `.sinf`/Physical Plan; vision and MTP are excluded. Evidence: golden generation artifact with source/model/code/plan hashes.

## Kernel Portfolio

- **KER-001** — A correctness baseline exists for every operation required by the current qualified model phase before specialized fusion. Evidence: operation matrix.
- **KER-002** — KernelProvider selection is driven by declared capabilities and measurable constraints, never model-name switches. Evidence: selector tests.
- **KER-003** — Portfolio includes measured candidates for GEMM/GEMV, RMSNorm/residual, RoPE, prefill attention, decode attention/KV update, sparse attention, gated FFN, MoE routing/grouped experts, embeddings/LM head, and sampling. Evidence: registry report.
- **KER-004** — Every promoted kernel passes reference differential, boundary-shape, determinism, and representative-model tests. Evidence: promotion record.
- **KER-005** — Kernel selection can fall back to a correct baseline when a specialization is inapplicable. Evidence: capability/fallback tests.
- **KER-006** — KV/recurrent state layouts are explicit, versioned in the plan, bounds checked, and compatible with selected providers. Evidence: layout round-trip and continuation tests.

## Decode Strategies

- **DEC-001** — Greedy and sampling decode are independent `DecodeStrategy` implementations over a stable runtime contract. Evidence: strategy conformance tests.
- **DEC-002** — Speculative decoding owns proposal, verification, acceptance, rollback, and KV consistency; attention remains a kernel concern. Evidence: state-machine tests.
- **DEC-003** — DSpark experiments are represented as speculative `DecodeStrategy` work, not an attention implementation. Evidence: interface placement and experiment manifest.
- **DEC-004** — Decode strategies declare memory/workspace and kernel requirements at compile time. Evidence: invalid-plan tests.

## Autoresearch

- **RES-001** — Experiments are declarative, reproducible, isolated, and record source revision, patch, toolchain, GPU state, seeds, workload, and raw results. Evidence: replay test.
- **RES-002** — Candidate pipeline gates build -> static/unit -> differential correctness -> sanitizer -> determinism -> benchmark -> statistical decision. Evidence: rejected/passed fixture runs.
- **RES-003** — Correctness failure, flaky output, thermal invalidity, or insufficient samples prevents promotion. Evidence: gate unit tests.
- **RES-004** — Promotion produces an auditable patch and evidence bundle; rollback is one commit/revert away. Evidence: promotion integration test.
- **RES-005** — Search budgets and tunable domains are bounded; experiments cannot silently change benchmark semantics. Evidence: schema validation tests.

## Benchmarking and Public Proof

- **BEN-001** — Benchmark protocol pins model/artifact, prompt/output distributions, batch/concurrency, context lengths, decode settings, clocks/power policy, software versions, warmup/sample policy, and device placement/residency when multi-device. Evidence: validated run manifest.
- **BEN-002** — Prefill and decode are reported separately with TTFT, TPOT/inter-token latency, throughput, latency percentiles, memory use, correctness status and relevant transfer/residency metrics. Evidence: report schema tests.
- **BEN-003** — Raw samples and environment metadata are retained; graphs are generated from data, never manually edited. Evidence: reproducible report build.
- **BEN-004** — Comparisons use matched semantics and clearly name baselines and unsupported/unmatched dimensions. Evidence: comparison audit checklist.
- **BEN-005** — First RTX 5090 graph can be regenerated on a clean checkout from a documented command and evidence bundle. Evidence: release CI/manual attestation.

## Quality, CI, and Release

- **QUA-001** — CPU-only CI covers formatting, linting, type checks, build, unit, property, parser fuzz-smoke, and artifact golden tests. Evidence: required checks.
- **QUA-002** — GPU CI tiers cover smoke, per-commit correctness, scheduled exhaustive/sanitizer, controlled benchmark lanes, and dual-GPU correctness when S03F is supported. Evidence: CI matrix.
- **QUA-003** — Tests have explicit owners, deterministic seeds, timeouts, and actionable failure artifacts. Evidence: test metadata audit.
- **QUA-004** — ABI/artifact schema compatibility is tested across supported versions. Evidence: compatibility matrix.
- **QUA-005** — Documentation maps supported models, features, constraints, placement/residency behavior, and benchmark reproduction. Evidence: release checklist.
- **REL-001** — V0 ships install/build instructions, converter/runtime CLIs, sample manifests, changelog, security/reporting policy, and known limitations. Evidence: clean-machine release rehearsal.
- **REL-002** — Unsupported hardware, device topology, model configs, kernels, placements and artifacts fail safely with diagnostic context. Evidence: negative integration suite.

## Understanding Governance

- **GOV-001** — Phase transitions record implementation phase, current user gate/status, debt distance/override, allowed autonomous work, blocked boundary, and next user action in both state and user-facing output. Evidence: transition commit and `UNDERSTANDING STATUS` transcript.
- **GOV-002** — The normal one-L2 debt rule remains canonical; any autonomous override such as D-014 must remain explicit and cannot mark gates passed on the user's behalf. Evidence: state/ledger history.
- **GOV-003** — Every reached L2 gate has an evidence-based Understanding Packet with the canonical ten contents, exactly three reading files, one hands-on experiment, and exactly five user questions. Evidence: packet schema review and linked phase evidence.
- **GOV-004** — An L2 gate passes only with recorded evidence that the user can explain, predict, trace, diagnose, and complete a small predicted change; L0/L1 work remains non-blocking unless explicitly elevated. Evidence: `.planning/UNDERSTANDING.md` gate record and agent assessment.

## Coverage Policy

Every phase in `ROADMAP.md` lists the requirement IDs it owns. Cross-cutting requirements may appear in multiple phases; the last owning phase must supply the final milestone evidence. No requirement—including understanding governance—may be marked complete based only on an implementation commit.
