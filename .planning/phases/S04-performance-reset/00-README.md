# S04 Performance Reset — Execution Packet

**Status:** ACTIVE  
**Authority:** D-022 + this packet supersede the old incremental S04/P9 ordering. Historical P8/P9 evidence is retained; it is not the active roadmap.  
**Base:** sol/results-first-recovery @ 2983d28  
**Primary target:** Qwen3.8-27B on one RTX 5090 / sm_120a  
**Production baseline:** P7 software W4A16-style decode, about 123.6 ms/token device span, about 8.1–8.3 tok/s.  
**Frontier reference:** SparkInfer is the primary comparator because SuperInfer's pinned derivative lineage is gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090. NInfer is the primary donor/reference implementation for specialized decode kernels.

## Mission

Get SuperInfer into the competitive regime quickly by replacing weak commodity data-plane code with proven implementations and reserving custom research for places where SuperInfer can actually differentiate.

This is an **implementation sprint with bounded truth checks**, not a research sprint.

The central question is no longer "what kernel should we hand-optimize next?" It is:

> Can the existing SuperInfer shell become fast when its projection/fusion data plane is replaced with mature RTX-5090 implementations?

Answer that while building the replacement.

## Operating policy

1. **Default action is integrate, not experiment.**
2. A microbenchmark is allowed only when it selects between concrete implementations that can be integrated immediately.
3. No experiment may consume more than one bounded work block without producing code, a decisive rejection, or a blocker.
4. Prefer donor code or donor architecture when it is within 10% of the best known implementation on the same shape.
5. Do not spend time "improving" P7 once a donor path is available.
6. Every workday ends with a whole-model ms/token number or a concrete integration blocker.
7. Do not disturb user-owned NInfer or other GPU processes. GPU work is scheduled around them.
8. Preserve the pre-existing untracked S03 artifact.
9. No new generic extension surfaces. Work through KernelProvider, StoragePolicy, GraphPass/lowering, DecodeStrategy, and the existing Physical Plan.
10. No P10. P9 is frozen.

## Packet map

- **01-MASTER-PLAN.md** — critical path, targets, gates, stop conditions.
- **02-E0A-DONOR-PROJECTIONS.md** — first implementation tranche: replace projection backend without changing command topology.
- **03-E0B-FUSED-DECODE.md** — second tranche: proven role fusion and lowering changes.
- **04-WEEK-ACCELERATION.md** — after E0: recurrence, attention, GPU feedback, graphs, long-context qualification.
- **05-DONOR-MAP.md** — exact code to study/port/wrap and licensing notes.
- **06-EVIDENCE-CONTRACT.md** — minimum sufficient correctness/performance evidence.
- **07-AGENT-KICKOFF.md** — executable agent mission and return contract.

## One-screen critical path

    SparkInfer truth (hours, not days)
              |
              +------------------------+
              |                        |
              v                        v
        E0a donor layout         donor code archaeology
        + projection swap        + integration support
              |
              v
       full-model benchmark
              |
              v
        E0b fused roles
              |
              v
        <= 40 ms/token?
          /         \
        yes          no
        |             |
        v             v
   keep shell     12h diagnosis only
        |             |
        +-------> if still >40:
                  pivot data plane
              |
              v
  recurrence + attention + GPU feedback + graph
              |
              v
       >= 67 tok/s week gate
              |
              v
    >=80 tok/s frontier pursuit
              |
              v
 compiler-retarget falsification test

## What is frozen

Until the week gate closes, do not spend implementation time on:

- P9 activation-quality recovery.
- 4-over-6, residual FP4, SmoothQuant, rotations.
- persistent whole-model megakernels.
- TP2.
- speculative decoding / DSpark / MTP integration.
- Flash-Next.
- generic scheduler/serving infrastructure.
- new IR abstractions.
- artifact redesign unrelated to required offline repacking.
- startup/TTFT optimization unless it blocks measurement.

Cheap inventory probes are allowed only when they do not interrupt the critical path.

## Definition of success

The sprint succeeds when SuperInfer stops being an 8 tok/s research runtime and becomes a credible low-latency 5090 engine.

Minimum one-week acceptance:

- ordinary decode >= 67 tok/s or >=70% of the fastest qualified same-machine SparkInfer result, whichever is harder;
- real autoregressive GPU feedback path exists;
- no full-vocabulary CPU round trip per token;
- donor/fused projection path is the production path;
- correctness is qualified under 06-EVIDENCE-CONTRACT.md;
- a current nsys profile identifies the next residual bottleneck;
- no performance claim is based only on cache-hot isolated GEMV.

Stretch: <=12.5 ms/token / >=80 tok/s ordinary decode.
