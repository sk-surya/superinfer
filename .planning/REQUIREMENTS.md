# Requirements

Requirements are evidence-oriented. A requirement is complete only when its named behavior exists and the specified evidence is retained.

## Architecture and Extensibility

- **ARCH-001** — The pipeline has exactly three durable representations: Semantic IR, Lowered IR, and Physical Plan. Evidence: type/API tests and architecture conformance test.
- **ARCH-002** — Semantic IR expresses model meaning without CUDA launch details or storage offsets. Evidence: dependency-boundary test.
- **ARCH-003** — Lowered IR contains target-aware tensor/layout/fusion decisions while remaining inspectable and serializable. Evidence: golden lowering dumps.
- **ARCH-004** — Physical Plan is immutable after validation and directly executable without model-family inspection. Evidence: executor dependency test and runtime trace.
- **ARCH-005** — Supported extension surfaces are `ModelFrontend`, `GraphPass`, `KernelProvider`, `DecodeStrategy`, and `StoragePolicy`. Evidence: interface contract tests and example plugins/providers.
- **ARCH-006** — New model support does not require edits to the physical executor. Evidence: Gemma 4 phase diff/conformance test.
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

- **MOD-001** — Qwen3.8-27B is the first complete frontend and canonical V0 model. Evidence: config/tensor mapping fixtures and reference comparisons.
- **MOD-002** — Frontends validate configs and tensor schemas with actionable errors; no best-effort silent defaults. Evidence: malformed-input tests.
- **MOD-003** — Semantic operations cover embeddings, normalization, RoPE, attention variants, gated dense FFN, MoE routing/experts, residuals, LM head, and optional auxiliary heads needed by target models. Evidence: IR verifier tests.
- **MOD-004** — Tokenizer and chat-template identity are preserved and reference-compatible. Evidence: tokenizer golden tests.
- **MOD-005** — Gemma 4 26B-A4B is the second frontend and validates architectural composability. Evidence: end-to-end artifact/generation tests and executor-no-change check.

## `sm120` Backend and Runtime

- **BCK-001** — The first backend targets RTX 5090 `sm_120a` and rejects incompatible hardware/artifacts clearly. Evidence: capability probe tests.
- **BCK-002** — Compilation specializes layouts, workspace, kernel choices, launch parameters, and decode schedule for the exact model and target profile. Evidence: Physical Plan dumps.
- **BCK-003** — Runtime validates the complete plan and resource bounds before execution. Evidence: adversarial-plan tests.
- **BCK-004** — Decode hot path performs no heap allocation, model-family branching, filesystem access, JIT compilation, or implicit global synchronization. Evidence: allocation/trace regression test.
- **BCK-005** — Buffer/workspace ownership and CUDA stream/event lifetimes are explicit and deterministic. Evidence: compute-sanitizer and lifecycle tests.
- **BCK-006** — CPU/reference executor or oracle path can execute small graphs for differential testing. Evidence: randomized IR tests.

## Kernel Portfolio

- **KER-001** — A correctness baseline exists for every operation required by Qwen3.8 before specialized fusion. Evidence: operation matrix.
- **KER-002** — KernelProvider selection is driven by declared capabilities and measurable constraints, never model-name switches. Evidence: selector tests.
- **KER-003** — Portfolio includes measured candidates for GEMM/GEMV, RMSNorm/residual, RoPE, prefill attention, decode attention/KV update, gated FFN, MoE routing/grouped experts, embeddings/LM head, and sampling. Evidence: registry report.
- **KER-004** — Every promoted kernel passes reference differential, boundary-shape, determinism, and representative-model tests. Evidence: promotion record.
- **KER-005** — Kernel selection can fall back to a correct baseline when a specialization is inapplicable. Evidence: capability/fallback tests.
- **KER-006** — KV-cache layout is explicit, versioned in the plan, bounds checked, and compatible with the selected attention provider. Evidence: layout round-trip and long-context tests.

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

- **BEN-001** — Benchmark protocol pins model/artifact, prompt/output distributions, batch/concurrency, context lengths, decode settings, clocks/power policy, software versions, and warmup/sample policy. Evidence: validated run manifest.
- **BEN-002** — Prefill and decode are reported separately with TTFT, TPOT/inter-token latency, throughput, latency percentiles, memory use, and correctness status. Evidence: report schema tests.
- **BEN-003** — Raw samples and environment metadata are retained; graphs are generated from data, never manually edited. Evidence: reproducible report build.
- **BEN-004** — Comparisons use matched semantics and clearly name baselines and unsupported/unmatched dimensions. Evidence: comparison audit checklist.
- **BEN-005** — First RTX 5090 graph can be regenerated on a clean checkout from a documented command and evidence bundle. Evidence: release CI/manual attestation.

## Quality, CI, and Release

- **QUA-001** — CPU-only CI covers formatting, linting, type checks, build, unit, property, parser fuzz-smoke, and artifact golden tests. Evidence: required checks.
- **QUA-002** — GPU CI tiers cover smoke, per-commit correctness, scheduled exhaustive/sanitizer, and controlled benchmark lanes. Evidence: CI matrix.
- **QUA-003** — Tests have explicit owners, deterministic seeds, timeouts, and actionable failure artifacts. Evidence: test metadata audit.
- **QUA-004** — ABI/artifact schema compatibility is tested across supported versions. Evidence: compatibility matrix.
- **QUA-005** — Documentation maps supported models, features, constraints, and benchmark reproduction. Evidence: release checklist.
- **REL-001** — V0 ships install/build instructions, converter/runtime CLIs, sample manifests, changelog, security/reporting policy, and known limitations. Evidence: clean-machine release rehearsal.
- **REL-002** — Unsupported hardware, model configs, kernels, and artifacts fail safely with diagnostic context. Evidence: negative integration suite.

## Understanding Governance

- **GOV-001** — Phase transitions record implementation phase, current user gate/status, debt distance, allowed autonomous work, blocked boundary, and next user action in both state and user-facing output. Evidence: transition commit and `UNDERSTANDING STATUS` transcript.
- **GOV-002** — Reached-but-unpassed L2 debt never exceeds one gate; agents stop before a second L2 boundary while mechanical, planning, and reversible preparation may continue. Evidence: `.planning/STATE.md`/ledger history and phase-transition audit.
- **GOV-003** — Every reached L2 gate has an evidence-based Understanding Packet with the canonical ten contents, exactly three reading files, one hands-on experiment, and exactly five user questions. Evidence: packet schema review and linked phase evidence.
- **GOV-004** — An L2 gate passes only with recorded evidence that the user can explain, predict, trace, diagnose, and complete a small predicted change; L0/L1 work remains non-blocking unless explicitly elevated. Evidence: `.planning/UNDERSTANDING.md` gate record and agent assessment.

## Coverage Policy

Every phase in `ROADMAP.md` lists the requirement IDs it owns. Cross-cutting requirements may appear in multiple phases; the last owning phase must supply the final milestone evidence. No requirement—including understanding governance—may be marked complete based only on an implementation commit.
