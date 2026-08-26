# Understanding Gates

This is the canonical protocol for keeping implementation velocity high while preserving an explicit learning trail. [`.planning/UNDERSTANDING.md`](UNDERSTANDING.md) is the mutable ledger; phase files identify when this protocol applies.

## Operating Rule

- Agents work autonomously within the current phase and may prepare the next phase.
- Mechanical work does not wait for the user. The user is not expected to read every task, file, or implementation detail.
- The default governance model bounds conceptual debt by reached-but-user-unpassed L2 gates.
- D-014 is an explicit user-directed override: implementation may continue autonomously beyond the default debt window, but every gate/packet remains durable and **no gate is ever marked passed on the user's behalf**.
- Planning, investigation, packet preparation, and other reversible work may continue at any understanding boundary.
- Phase-specific engineering dependencies still block work even under D-014. In particular, S03F-02 runtime implementation is blocked until S03 correctness closes; this is an engineering dependency, not an understanding-gate pause.

## Gate Levels

| Level | Meaning | User obligation | Normal blocking behavior |
|---|---|---|---|
| **L0 — Observe** | Infrastructure or mechanical work | Read transition summary when useful | Never blocks implementation |
| **L1 — Understand** | Important, reversible subsystem | Spend about 20–30 minutes on the mental model, trace, or question | Does not normally block |
| **L2 — Own** | A concept central to SuperInfer's architecture, correctness, or performance thesis | Pass the ownership test below | Default policy blocks before a second unpassed L2 boundary; D-014 may override execution blocking while retaining evidence |

Understanding is not code-reading completeness. An L2 gate is passed when the user can:

1. explain the mechanism from memory;
2. predict behavior before seeing a test, trace, or benchmark;
3. trace one execution from input to the relevant hardware operation;
4. change one small thing and correctly predict its effect.

The agent records the user's answers and experiment evidence in `.planning/UNDERSTANDING.md`; unsupported self-certification or agent completion does not pass a gate.

## L2 Understanding Packet

When an L2 gate is reached, the responsible agent creates a small packet in `.planning/understanding-packets/`, links it from the ledger, and proactively presents it to the user. Every packet contains:

1. what changed;
2. why it exists and which boundary it protects;
3. one execution path from user/model input to hardware-visible work;
4. important tensor shapes and data structures;
5. core invariants;
6. a concise performance model;
7. likely failure modes and where to diagnose them;
8. **exactly three files worth reading**;
9. one bounded hands-on experiment with a predicted outcome;
10. **exactly five questions** the user should answer.

The five questions test explanation, prediction, execution tracing, diagnosis, and the hands-on result. Packet evidence should be concrete—plan dumps, traces, profiler captures, tensor tables, transfer timelines or test outputs—not a code-tour transcript.

## Reaching and Passing

1. The phase/plan named in `ROADMAP.md` produces the packet and marks the gate `reached` in the ledger.
2. The agent updates `.planning/STATE.md` before further phase-transition work.
3. The agent presents the packet and `UNDERSTANDING STATUS` block without waiting to be asked.
4. The user may perform the exercise and answer the five questions when convenient under D-014.
5. A gate becomes `passed` only when all four L2 ownership capabilities have evidence.
6. D-014 changes execution blocking only; it does not change the definition of passage.

L0 and L1 never add ownership obligations beyond the recorded learning trail. A conditional L2 gate is reached only when its documented trigger occurs.

## Required Phase-Transition Output

At every phase transition—and immediately when any L2 gate is reached—agents emit:

```text
UNDERSTANDING STATUS
Implementation phase: <phase and plan>
Current user gate: <gate/level and status>
Autonomy policy: default | D-014 override
Allowed next autonomous work: <specific scope>
Engineering-blocked boundary: <dependency boundary, or none>
User action: <one concrete optional/required exercise or question>
Packet: <path, or not required>
```

The same facts must be reflected in `.planning/STATE.md` and `.planning/UNDERSTANDING.md` at transition commits.

## Gate Map

| Phase | Level / gate | Conceptual ownership and behavior |
|---|---|---|
| S00 | L0/L1 | Basic CUDA execution model. Explain why asynchronous launch makes CPU wall timing misleading. |
| S01 | **L2 Gate A** | Semantic IR/compiler boundaries and the full frontend-to-runtime pipeline. |
| S02 | **L2 Gate B** | One-token transformer execution, Qwen-derived tensor shapes, and prefill versus decode. |
| S03 | L1 | Trace one weight from Hugging Face tensor through quantize/pack, `.sinf`, load and GPU use. |
| S03F | **L2 Flash-Next Architecture Gate** | Own compiler-selected multi-device placement, explicit transfer commands, heterogeneous PLE residency/prefetch, MoE route/residency, QSA selection vs sparse attention, gated residual, and multi-token state continuation. S03F-01 research does not by itself reach this gate; prepare/present the packet when the first complete multi-device Flash-Next layer path is qualified. |
| S04 | **L2 Gate C, repeated by mechanism** | Roofline, hierarchy, Tensor Cores versus CUDA cores, occupancy, fusion and Nsight prediction. C.1 dense/NVFP4, C.2 attention/KV/QSA, C.3 fusion/persistent/MoE mechanisms each count as a mini-gate. |
| S05 | L1 / conditional L2 methodology | Own validity, noise and statistical decisions; do not gate every mutation. Methodology changes trigger a mini-gate. |
| S06 | **L2 Gate D at entry** | Own vanilla speculative decoding draft/verify/accept economics. S05-03 produces the packet. |
| S07 | L1 / conditional L2 architecture audit | Inspect core changes needed for Gemma and challenge model-specific branching. Any proposed new core boundary triggers L2 review. |
| S08 | L0/L1 | Productization mechanics are generally non-blocking; audit that gate evidence ships coherently. |

## Flash-Next Architecture Packet

The S03F packet is deliberately one architecture packet rather than separate gates for every new operator. It must use implementation evidence from S03F-02 through the first complete multi-device layer qualification and cover:

- why placement belongs in Physical Plan rather than Semantic IR;
- why transfers are explicit commands and how peer vs pinned-host fallback is represented;
- why PLE host residency is not generic CPU offload and what actually crosses PCIe;
- how route/top-k/dispatch/expert/combine map to resident MoE execution;
- why QSA index/select is separate from sparse attention;
- how GDN/KV/gated-residual state is committed across tokens and device boundaries;
- one capacity/communication prediction the user makes before inspecting the measured trace.

Exactly three files, one exercise and five questions still apply.
