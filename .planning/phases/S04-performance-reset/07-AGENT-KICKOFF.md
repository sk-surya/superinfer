# Agent Kickoff — SuperInfer Performance Reset

You are the implementation lead for SuperInfer's reuse-first RTX-5090 acceleration sprint.

Repository:

    sk-surya/superinfer

Starting branch:

    sol/results-first-recovery

The branch must already contain D-022 and this S04-performance-reset packet. Do not reset the branch to an older commit.

Implementation reset baseline (the last pre-reset engineering commit):

    2983d285ca5db83a285c017291437893ea3f8958

Your mission is to make end-to-end ordinary Qwen3.8-27B decode materially fast. This is not an architecture-review assignment and not an open-ended research assignment.

## Read before touching code

1. AGENTS.md
2. .planning/DECISIONS.md — especially D-022
3. .planning/STATE.md
4. .planning/phases/S04-performance-reset/00-README.md
5. 01-MASTER-PLAN.md
6. 02-E0A-DONOR-PROJECTIONS.md
7. 03-E0B-FUSED-DECODE.md
8. 05-DONOR-MAP.md
9. 06-EVIDENCE-CONTRACT.md

Historical P8/P9 material is evidence, not your roadmap.

## Core directive

**BUILD FIRST. EXPERIMENT ONLY TO UNBLOCK BUILDING.**

Do not spend the session producing another research report.

The project already knows the main problem:

- about 82.9 ms/token in 401 packed NVFP4 projection launches;
- about 18.5 ms/token in 96 control linears;
- about 123.6 ms/token device span;
- about 8.1–8.3 tok/s.

Mature engines are dramatically faster on the same GPU/model class.

## Immediate objective — E0a

Replace the projection subsystem behind the existing command topology with mature donor-style kernels.

Primary donor: NInfer.

First inspect and adapt:

    src/ops/linear/nvfp4/nvfp4_gemv.cuh
    src/ops/linear/nvfp4/nvfp4_config.h
    src/ops/linear/nvfp4/shapes/
    src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
    src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu

Use the exact 401-command census in SuperInfer.

E0a may change offline/prepared weight and scale layout through StoragePolicy. It may NOT change Physical Plan command topology or add semantic fusion.

Do not add standalone FP32->BF16 cast launches. SuperInfer activations are currently FP32. Adapt donor loads/conversion inside the GEMV path so activation conversion does not become hundreds of new launches.

Also replace the 96 tiny GDN control projections with a proper specialized implementation while preserving command topology.

Keep P7 as fallback/oracle.

## Parallel external truth

If you are the agent owning SparkInfer/frontier work, your benchmark budget is short.

SuperInfer's pinned derivative lineage is:

    gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090

Establish a trustworthy same-machine SparkInfer ordinary-decode number and provenance. Then stop benchmarking and help the implementation agent with donor layouts/fusion/source archaeology.

Do not spend 48 hours comparing engines.

## E0b follows immediately

Once E0a runs correctly, implement:

1. fused gate/up/SwiGLU;
2. fused down/residual;
3. fused GDN norm + A/B control + gate parameters;
4. direct projection output routing where trivial.

Use donor patterns named in 03-E0B-FUSED-DECODE.md.

This is implementation, not a new research decision.

## Time/attention rules

- A microbenchmark exists to choose or validate code you are about to integrate.
- If a donor implementation is >10% faster after one serious adaptation attempt, use the donor.
- If one rare shape is difficult, retain fallback and land the high-value majority first.
- Do not optimize P7.
- Do not touch P9.
- Do not work on persistent megakernels, TP2, speculation, Flash-Next, or generic schedulers.
- Do not redesign .sinf except for the smallest deterministic prepared-layout support E0a requires.
- Do not wait for user understanding-gate answers; D-014 remains active.
- Do not disturb user-owned NInfer/GPU workloads.
- Preserve the pre-existing untracked S03 artifact.

## Performance gates

Hour-24 E0a target:

    projection subsystem <= 20 ms/token
    whole model <= 55 ms/token

Hour-48 E0b:

    survival <= 40 ms/token
    target 30–35 ms/token

If E0b is 40–49 ms/token, one bounded diagnosis/fix window is allowed.

If still >40 ms/token after that, return a runtime/data-plane pivot recommendation backed by the measured residual profile. Do not begin another incremental kernel ladder.

One-week target after E0:

    <=15 ms/token
    >=67 tok/s
    >=70% of fastest same-machine SparkInfer result

Stretch:

    <=12.5 ms/token
    >=80 tok/s

## Correctness

Use the evidence tiers in 06-EVIDENCE-CONTRACT.md.

Do not run the full expensive corpus after every edit.

For implementation-preserving kernels:
- local differential;
- representative layer/GDN smoke;
- short full-model smoke.

At E0 milestones run the stronger gates.

Do not weaken tolerances to land a kernel.

## Required working behavior

Work autonomously.

Inspect exact repository state and donor source yourself. Do not trust comments or this prompt where code disagrees.

Commit coherent implementation slices.

Keep the working tree clean except for explicitly preserved pre-existing user artifacts.

If a planned donor path is unavailable or incompatible, choose the fastest credible alternative and record the deviation. Do not stop merely because the plan named a particular file.

## Required final return

Return one compact engineering report:

    START_SHA
    FINAL_SHA
    commits
    files_changed

    SPARKINFER_SCOREBOARD
    - artifact/checkpoint identity
    - context
    - device ms/token
    - tok/s
    - command used

    E0A
    - projection backend implemented
    - prepared layout
    - supported/fallback shapes
    - projection ms/token before/after
    - control-linear ms/token before/after
    - full-model ms/token and tok/s
    - correctness

    E0B
    - fusions landed
    - launches/token before/after
    - full-model ms/token and tok/s
    - correctness

    PROFILE
    - top 10 remaining GPU kernels
    - device idle
    - effective bandwidth where measurable

    GATE
    - PASS <=40 ms/token
    - DIAGNOSE 40–49
    - PIVOT >=50 or >40 after bounded diagnosis

    NEXT
    - exactly one next implementation target

Do not return a future plan in place of code. If time ends mid-sprint, return the strongest integrated state with measurements and the exact blocker.