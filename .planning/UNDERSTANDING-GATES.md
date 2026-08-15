# Understanding Gates

This is the canonical protocol for keeping implementation velocity high while bounding the user's understanding debt. [`.planning/UNDERSTANDING.md`](UNDERSTANDING.md) is the mutable ledger; phase files identify when this protocol applies.

## Operating Rule

- Agents work autonomously within the current phase and may prepare the next phase.
- Mechanical work does not wait for the user. The user is not expected to read every task, file, or implementation detail.
- Conceptual debt is measured by reached but user-unpassed L2 gates, not by commits or phases. Debt distance MUST remain `0` or `1`.
- After reaching one unpassed L2 gate, agents may finish its mechanical tail and do explicitly allowed work toward the next conceptual boundary. They MUST stop before reaching another L2 gate, because that would make the debt distance `2`.
- Planning, investigation, packet preparation, and other reversible work may continue at a blocked boundary. Agents must not silently implement across it.
- Passing the oldest outstanding L2 gate reduces the debt distance and releases the next boundary.

## Gate Levels

| Level | Meaning | User obligation | Blocking behavior |
|---|---|---|---|
| **L0 — Observe** | Infrastructure or mechanical work | Read the transition summary when useful | Never blocks implementation |
| **L1 — Understand** | Important, reversible subsystem | Spend about 20–30 minutes on the mental model, trace, or question | Does not normally block; agents record unknowns and continue |
| **L2 — Own** | A concept central to SuperInfer's architecture, correctness, or performance thesis | Pass the ownership test below | Blocks only when another unpassed L2 gate would be reached |

Understanding is not code-reading completeness. An L2 gate is passed when the user can:

1. explain the mechanism from memory;
2. predict behavior before seeing a test, trace, or benchmark;
3. trace one execution from input to the relevant hardware operation;
4. change one small thing and correctly predict its effect.

The agent records the user's answers and experiment evidence in `.planning/UNDERSTANDING.md`; perfection is not required, but unsupported self-certification is not sufficient.

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

The five questions must test explanation, prediction, execution tracing, diagnosis, and the hands-on result. Packet evidence should be concrete—plan dumps, traces, profiler captures, tensor tables, or test outputs—not a code-tour transcript.

## Reaching, Passing, and Blocking

1. The phase/plan named in `ROADMAP.md` produces the packet and marks the gate `reached` in the ledger.
2. The agent calculates debt distance and updates `.planning/STATE.md` before further implementation.
3. The agent presents the packet and the `UNDERSTANDING STATUS` block below without waiting to be asked.
4. The user performs the exercise and answers the five questions. The agent records strengths, corrections, diagnostic ability, and remaining unknowns.
5. The gate becomes `passed` only when all four L2 ownership capabilities have evidence. Otherwise it remains `in progress`; agents continue only within the allowed debt window.
6. If proceeding would reach a second unpassed L2 gate, the agent marks the boundary blocked, reports the concrete user action needed, and limits work to preparation or other non-crossing tasks.

L0 and L1 never add to debt distance. A conditional L2 gate adds to debt only when its trigger occurs and the packet is presented.

## Required Phase-Transition Output

At every phase transition—and immediately when any L2 gate is reached—agents emit:

```text
UNDERSTANDING STATUS
Implementation phase: <phase and plan>
Current user gate: <gate/level and status>
Debt distance: <0 or 1>
Allowed next autonomous work: <specific scope>
Blocked boundary: <next L2 boundary, or none>
User action: <one concrete exercise or question>
Packet: <path, or not required>
```

The output is a user-facing obligation, not optional progress commentary. The same values must be reflected in `.planning/STATE.md` and `.planning/UNDERSTANDING.md` in the phase-transition commit.

## Gate Map

| Phase | Level / gate | Conceptual ownership and behavior |
|---|---|---|
| S00 | L0/L1 | Basic CUDA execution model. No hard stop; explain why asynchronous launch makes CPU wall timing misleading. |
| S01 | **L2 Gate A** | Semantic IR/compiler boundaries and the full frontend-to-runtime pipeline. |
| S02 | **L2 Gate B** | One-token transformer execution, Qwen-derived tensor shapes, and prefill versus decode. |
| S03 | L1 | Trace one weight from Hugging Face tensor through quantize/pack, `.sinf`, load, and GPU use. |
| S04 | **L2 Gate C, repeated by mechanism** | Roofline, hierarchy, Tensor Cores versus CUDA cores, occupancy, fusion, and Nsight prediction. C.1 dense/NVFP4, C.2 attention/KV, C.3 fusion/persistent mechanisms each count as an L2 mini-gate. |
| S05 | L1 / conditional L2 methodology | Own validity, noise, and statistical decisions; do not gate every mutation. Methodology changes trigger a mini-gate. |
| S06 | **L2 Gate D at entry** | Own vanilla speculative decoding draft/verify/accept economics. S05-03 produces the packet; S06 enforces the boundary. |
| S07 | L1 / conditional L2 architecture audit | Inspect every core change needed for Gemma and challenge model-specific branching. Any proposed executor/core-boundary change triggers L2 before implementation. |
| S08 | L0/L1 | Productization mechanics are generally non-blocking; audit that gate evidence ships coherently. |

Gate-specific exercises and questions live in phase context and the packet created from implementation evidence.
