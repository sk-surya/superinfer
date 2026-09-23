---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: S04-RESET
status: autonomous_execution
last_updated: "2026-09-23T00:00:00Z"
progress:
  total_phases: 14
  completed_phases: 9
  total_plans: 35
completed_plans: 33
current_phase_name: reuse-first-5090-acceleration
parallel_research_phase: none
s03f_01_status: research_complete_capacity_quality_blocked
---

# Project State

**Project:** SuperInfer  
**Branch:** sol/results-first-recovery  
**Reset base:** 2983d28  
**Active authority:** D-022 and .planning/phases/S04-performance-reset/

## Current lane

The old incremental S04/P9 lane is frozen.

The active lane is the **reuse-first RTX-5090 performance reset**:

1. bounded SparkInfer same-machine truth;
2. E0a donor projection backend with unchanged Physical Plan command topology;
3. E0b role fusion/lowering;
4. <=40 ms/token architecture gate;
5. recurrence + GPU feedback + CUDA graph + mature attention;
6. >=67 tok/s one-week gate.

Default action is implementation. Experiments exist only to select or verify code that will be integrated immediately.

## Current performance truth

P7 production remains about 123.6 ms/token device span / 8.1–8.3 tok/s.

Measured P7 profile over 60 steps:

- packed NVFP4 linears: about 82.87 ms/token;
- linear_f32 control projections: about 18.51 ms/token;
- RMSNorm: about 5.23 ms/token;
- causal conv + SiLU: about 3.68 ms/token;
- GDN recurrence: about 3.25 ms/token;
- SiLU multiply: about 2.63 ms/token;
- BF16->FP32 casts: about 1.09 ms/token;
- device idle: about 2.3%.

The immediate target is therefore the projection subsystem, not launch topology research.

## P8/P9 status

P8-R remains a valid hardware result: RTX 5090 / sm_120a executes native block-scaled NVFP4 MMA and the synthetic fragment/scale differential was proven.

P8RQ remains evidence that the native path is fast but the tested activation recipe changed numerics materially.

P9 is **not an active negative verdict on canonical NVFP4 quality**. The two-level implementation at 2983d28 writes raw global amax and then consumes it as s_global; the documented global_amax/(448*6) transformation is absent. E4M3/E2M1 encoder semantics also require independent verification. Therefore P9 is frozen, not extended.

This does not block E0 because ordinary T=1 decode will first pursue a mature weight-only streaming path.

## Architecture decision

SuperInfer's near-term value is:

- .sinf/provenance;
- AOT specialization;
- exact-shape/layout selection;
- static memory planning;
- fusion/layout generation;
- correctness/evidence infrastructure.

Commodity kernels may be imported/wrapped. Ownership of CUDA code is not itself a goal.

The specialization-compiler thesis remains provisional and must later pass the retargeting-cost gate defined in 04-WEEK-ACCELERATION.md.

## Active targets

### E0a
- donor-backed complete projection family;
- specialized tiny control projections;
- no command-topology changes;
- full-model number by the first implementation milestone;
- target <=55 ms/token.

### E0b
- gate/up/SwiGLU;
- down/residual;
- GDN norm/control/gating;
- direct output routing;
- survival <=40 ms/token;
- target 30–35 ms/token.

### Week
- <=15 ms/token / >=67 tok/s;
- >=70% of fastest qualified same-machine SparkInfer result;
- stretch >=80 tok/s.

## Frozen work

Until the week gate:

- P9/activation FP4 quality work;
- persistent whole-model kernel;
- TP2;
- MTP/DSpark;
- Flash-Next implementation;
- generic scheduler/server;
- new IR layers;
- unrelated artifact redesign.

Cheap metadata/inventory probes are allowed only if they do not interrupt the critical path.

## Operational constraints

- Do not disturb user-owned NInfer or other GPU workloads.
- Do not assume a particular GPU index is free.
- Preserve the pre-existing untracked S03 artifact.
- P7 remains fallback/oracle until a replacement is qualified.
- D-006 correctness remains binding.
- D-014 autonomous understanding-gate override remains binding; no gate is falsely marked user-passed.

## Understanding Gate State

| Field | Current value |
|---|---|
| Historical gates | Gate A, Gate B, L2 Gate C.1 reached; none user-passed |
| User status | Packets retained for later study under D-014 |
| Highest passed L2 gate | None |
| Debt policy | D-014 autonomous override active |
| Next understanding event | C.2/C.3 when the relevant mechanism transition is actually reached |
| S03F | Engineering deferred; D-019 still binding |

Canonical protocol: .planning/UNDERSTANDING-GATES.md  
Durable user ledger: .planning/UNDERSTANDING.md

## Next command

Read .planning/phases/S04-performance-reset/07-AGENT-KICKOFF.md and execute E0a immediately.

Do not resume the historical profiler->5–15% optimization ladder.
