# S04 Performance Reset — Master Execution Plan

## Objective

Move from about 123.6 ms/token to the competitive RTX-5090 regime by replacing low-quality commodity implementations before inventing new architecture.

The sprint optimizes:

**engineering velocity × end-to-end decode performance**

not architectural purity and not experiment count.

## Known starting accounting

From artifacts/S04/s04p7-post-nsys-profile.json over 60 continuation steps:

| Component | Approx ms/token | Action |
|---|---:|---|
| Packed NVFP4 linears | 82.87 | Replace immediately |
| linear_f32 control projections | 18.51 | Replace immediately |
| RMSNorm | 5.23 | Fuse/replace |
| causal conv + SiLU | 3.68 | Fuse/replace |
| GDN recurrence | 3.25 | Import mature recurrence/state layout |
| SiLU multiply | 2.63 | Fuse |
| BF16->FP32 casts | 1.09 | Remove where fusion/representation permits |
| remaining kernels | 3.50 | Profile after major replacement |
| device idle | about 2.84 | Not today's primary problem |
| device span | about 123.61 | Baseline |

Approximately 82% of device time is in the projection subsystem targeted by E0.

## Strategic decisions

### D1 — T=1 decode uses a weight-only streaming path first

Do not require activation FP4 for ordinary single-token decode. The first competitive route is weight-only NVFP4 with higher-precision activations and FP32 accumulation, using a mature streaming GEMV design.

The native SM120 block-scaled MMA contract remains valuable for multi-token regimes. It is not the critical path for E0.

### D2 — E0a is plan-neutral, not storage-neutral

E0a may:

- create/reuse an offline packed layout;
- add a StoragePolicy representation;
- prepare scales once at conversion/load time;
- specialize by exact shape;
- adapt donor input loads from FP32 to BF16 in registers/shared memory;
- replace kernel IDs/providers.

E0a may not:

- change command topology;
- fuse semantic operations;
- add per-token conversion kernels;
- redesign the executor.

This gives a clean causal result while allowing the donor kernel to run in the layout it was designed for.

### D3 — E0b is implementation, not an experiment

Once E0a runs correctly, proceed directly to role fusion unless there is a fatal architectural blocker.

E0b adds:

- gate + up + SwiGLU;
- down + residual;
- RMSNorm + GDN A/B control projection + gate parameter generation;
- output split/store epilogues where already demonstrated by donors.

Do not pause for a long review between E0a and E0b.

### D4 — external benchmarking is bounded

Agent A gets enough time to:

- build/run pinned SparkInfer;
- prove checkpoint lineage / local LMHead4 delta;
- capture one trustworthy short-context ordinary-decode result;
- capture one medium/long-context point if cheap;
- record the commands/environment.

Then Agent A moves to donor archaeology and integration support.

NInfer is secondary scoreboard and primary implementation donor. Do not spend a day normalizing every public engine.

## Workstreams

### Workstream A — frontier truth + donor support

Initial budget: a few hours.

Deliverables:

1. SparkInfer same-machine ordinary decode result.
2. Exact model/checkpoint identity and any relevant LM-head conversion delta.
3. A concise byte-accounting note: model streamed bytes/token and effective GB/s.
4. Exact donor layout/schedule notes for Agent B.
5. Optional cheap probes: mtp tensor inventory; P2P only if both GPUs are actually free.

After item 1 exists, implementation support outranks additional benchmarking.

### Workstream B — E0a/E0b implementation

This is the critical path.

E0a must land a full-model donor-backed projection path.

E0b must convert demonstrated donor fusion patterns into Physical Plan lowering.

No benchmark-only branch is accepted as the main output.

## 48-hour execution sequence

### 0–4 hours — bootstrap

Both agents:

- verify HEAD and worktree;
- read this packet and D-022;
- preserve user-owned processes and untracked artifacts;
- create separate branches/worktrees;
- capture current P7 command and environment;
- inspect exact source dtypes/layouts rather than assuming them.

Agent A starts SparkInfer build/run.
Agent B starts donor layout + shape mapping immediately.

### 4–24 hours — E0a

Agent B:

- implement prepared donor-compatible weight/scale layout;
- implement/adapt W4A16 streaming NVFP4 GEMV for the complete 401 projection census;
- replace the tiny control projection implementation without changing command topology;
- do not add 401 cast kernels: consume current FP32 activation and convert values inside the donor kernel path as needed;
- integrate behind KernelProvider/StoragePolicy;
- run smallest correctness differential after each shape family;
- run whole-model decode as soon as all major shapes execute.

Required hour-24 artifact:

    full_model_device_ms_per_token
    full_model_tok_per_s
    projection_ms_per_token
    linear_control_ms_per_token
    kernel_launches_per_token
    correctness_status
    donor_ratio_on_major_shapes

Targets, not blockers:

- projections <=20 ms/token;
- E2E <=55 ms/token;
- >=18 tok/s.

A worse result does not trigger research. Profile once, fix the obvious integration issue, and continue.

### 24–48 hours — E0b

Add proven role-level fusion.

Priority:

1. gate/up/SwiGLU;
2. down/residual;
3. GDN norm + A/B + gate parameters;
4. output routing/splits that can disappear into epilogues.

Full-model result is mandatory.

Architecture gate:

- <=40 ms/token: current shell survives.
- 40–49 ms/token: one bounded 12-hour diagnosis window after hour 48; proceed only if the measured cause plausibly closes the gap.
- >=50 ms/token: prepare data-plane pivot unless the run is invalid.
- after the bounded diagnosis, still >40 ms/token: pivot.

Target: 30–35 ms/token.

## What "pivot" means

Pivot does NOT mean abandon SuperInfer.

It means stop maintaining a commodity runtime implementation that cannot exploit good kernels.

Retain:

- .sinf;
- converter/provenance;
- semantic/lowered representation where useful;
- specialization;
- layout generation;
- correctness assets;
- benchmark/evidence tooling.

Use an existing high-performance execution substrate or a much thinner fused executor underneath.

SuperInfer becomes primarily a specialization compiler/generator.

## Week gate

After E0b, implementation proceeds directly into 04-WEEK-ACCELERATION.md.

End-of-week hard target:

- <=15 ms/token OR >=67 tok/s ordinary decode;
- AND >=70% of fastest qualified same-machine SparkInfer ordinary decode;
- AND real GPU feedback path;
- AND quality gates green.

If SparkInfer measures materially faster than 95 tok/s locally, the relative gate dominates.

Stretch:

- <=12.5 ms/token;
- >=80 tok/s.

## Hard stop rules

1. No custom decode GEMV gets more than one serious implementation attempt if a donor implementation is >10% faster.
2. No kernel microbenchmark gets a second day unless the kernel is already integrated end-to-end.
3. No week with zero E2E ms/token improvement.
4. No fusion project before the unfused donor implementation exists, unless the donor only exists fused.
5. No executor rewrite before E0b, unless E0a proves the executor prevents donor kernels from running.
6. No speculative decoding before ordinary target >=60 tok/s.
7. No TP2 work before single-GPU week gate.
8. No P9 work during this sprint.
9. No claim of "world class" without same-machine external comparison.
10. No performance milestone based on projected bounds when an integrated measurement is possible.

## Acceptance evidence at each milestone

E0a:
- compile/build;
- focused donor-kernel differential;
- representative layer/GDN regression;
- full-model forced-token run;
- whole-model timing;
- nsys top-kernel table.

E0b:
- same as E0a;
- fusion equivalence tests;
- D-021 smoke;
- whole-model timing.

Week gate:
- full validation;
- D-021 full corpus;
- fresh-process reproduction;
- ordinary autoregressive generation;
- ctx short + one longer decode state;
- same-machine SparkInfer comparison;
- nsys profile;
- memory/power/config manifest.

## Stop condition

This plan ends when either:

A. SuperInfer reaches the week gate and the next bottleneck is selected from the new fast profile; or

B. the <=40 ms E0b shell gate fails after one bounded diagnosis and the runtime/data plane pivot is executed.

Do not reopen the old S04 ladder after this packet.
