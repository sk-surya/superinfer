# Understanding Ledger

This is the durable record of what the user understands, what evidence supports that assessment, and what remains unknown. Follow [`.planning/UNDERSTANDING-GATES.md`](UNDERSTANDING-GATES.md); update this ledger and `.planning/STATE.md` together at phase transitions and gate events.

## Current Status

| Field | Value |
|---|---|
| Implementation phase | S01 — Artifact and IR |
| Current user gate | Gate A — semantic IR/compiler boundaries — L2 |
| User status | Not started |
| Highest passed L2 gate | None |
| Outstanding L2 gates | None |
| Debt distance | 0 |
| Allowed autonomous work | Execute S01-01 through Gate A evidence preparation; do not cross Gate A. |
| Next user action | Explain why a CUDA launch can appear fast to CPU wall timing while GPU work is still running; answer the Gate A packet when presented. |
| Last updated | S00 transition on 2026-08-15 |

## Status Vocabulary

- **Implementation:** `not reached`, `reached`, or `superseded`.
- **User:** `not started`, `packet presented`, `in progress`, or `passed`.
- **Debt distance:** count of reached L2 gates newer than the highest passed L2 gate; only `0` or `1` is allowed.

## Gate Register

| Gate | Phase | Level | Implementation | User | Packet / evidence |
|---|---|---|---|---|---|
| S00 CUDA execution model | S00 | L0/L1 | Not reached | Not started | No packet required |
| Gate A — semantic IR/compiler boundaries | S01 | L2 | Not reached | Not started | Pending |
| Gate B — one-token transformer execution | S02 | L2 | Not reached | Not started | Pending |
| Gate C.1 — dense/NVFP4/Tensor Core mechanism | S04 | L2 | Not reached | Not started | Pending |
| Gate C.2 — attention/KV mechanism | S04 | L2 | Not reached | Not started | Pending |
| Gate C.3 — fusion/persistent mechanism | S04 | L2 | Not reached | Not started | Pending |
| Experimental-methodology mini-gate | S05 | Conditional L2 | Not reached | Not started | Trigger only for methodology changes |
| Gate D — speculative decoding economics | S05→S06 | L2 | Not reached | Not started | Produced by S05-03; enforced at S06 entry |
| Gemma architecture audit | S07 | L1 / conditional L2 | Not reached | Not started | L2 only if core boundary/executor change is proposed |

## Current Mental Model

### Mental model

To be filled from the user's own explanation. For S00, include host versus device work, asynchronous launch, streams/events, synchronization, kernel launch, and thread/warp/block/SM hierarchy.

### Can explain

- Pending: why enqueue time is not kernel execution time.
- Pending: why CUDA events on the relevant stream are needed for device timing.

### Can diagnose

- Pending: identify missing synchronization or measuring the wrong stream as likely causes of implausible timing.

### Hands-on experiment

- Pending: time the same small kernel using CPU wall time without synchronization, CPU wall time with synchronization, and CUDA events; predict the ordering before running it.

### Remaining unknowns

- Record uncertainties explicitly. Unknowns are acceptable at L0/L1 and are inputs to future packets, not hidden blockers.

## L2 Gate Record Template

Copy this section for each reached L2 gate. Keep the packet itself in `.planning/understanding-packets/` and link it here.

### <Gate name>

| Field | Value |
|---|---|
| Level | L2 |
| Implementation status | Not reached / reached |
| User status | Not started / packet presented / in progress / passed |
| Reached at | Commit, phase evidence, and date |
| Packet | Path |
| Debt distance after update | 0 or 1 |

**Mental model:** User's own concise model of the mechanism and boundary.

**Can explain:** Evidence from the explanation question.

**Can predict:** Prediction made before seeing the trace/test/benchmark, followed by the actual result.

**Can trace:** One input-to-hardware execution trace, including key shapes/data structures.

**Can diagnose:** Likely failure mode, observable symptom, and first evidence/file/tool to inspect.

**Hands-on experiment:** Small change, predicted effect, actual effect, and explanation of any mismatch.

**Five answers:** Record answers or a compact assessment linked to the packet's exactly five questions.

**Remaining unknowns:** Specific gaps, whether they block ownership, and the next learning action.

**Agent assessment:** `passed` or `in progress`, with evidence and date.

## Update Discipline

- Never replace the user's words with a generic agent summary when assessing an L2 gate.
- Preserve earlier gate records; append corrections and later refinements.
- A phase summary is incomplete until its `UNDERSTANDING STATUS` output agrees with this ledger and `.planning/STATE.md`.
