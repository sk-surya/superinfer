# Understanding Ledger

This is the durable record of what the user understands, what evidence supports that assessment, and what remains unknown. Follow [`.planning/UNDERSTANDING-GATES.md`](UNDERSTANDING-GATES.md); update this ledger and `.planning/STATE.md` together at phase transitions and gate events.

## Current Status

| Field | Value |
|---|---|
| Implementation phase | S03 — Qwen3.8 end-to-end |
| Current user gate | Gate B — one-token transformer execution — L2 |
| User status | Packet presented |
| Highest passed L2 gate | None |
| Outstanding L2 gates | Gate A — semantic IR/compiler boundaries; Gate B — one-token transformer execution |
| Debt distance | 2 under explicit D-014 autonomous-execution override |
| Allowed autonomous work | All approved roadmap plans; preserve packets and branch checkpoints for later study. |
| Next user action | Optional: study Gate B packet and answer its five questions/experiment when convenient. |
| Last updated | Autonomous execution enabled by explicit user directive on 2026-08-26 |

## Status Vocabulary

- **Implementation:** `not reached`, `reached`, or `superseded`.
- **User:** `not started`, `packet presented`, `in progress`, or `passed`.
- **Debt distance:** count of reached L2 gates newer than the highest passed L2 gate; only `0` or `1` is allowed.

## Gate Register

| Gate | Phase | Level | Implementation | User | Packet / evidence |
|---|---|---|---|---|---|
| S00 CUDA execution model | S00 | L0/L1 | Not reached | Not started | No packet required |
| Gate A — semantic IR/compiler boundaries | S01 | L2 | Reached | Packet presented | [GATE-A.md](understanding-packets/GATE-A.md) |
| Gate B — one-token transformer execution | S02 | L2 | Reached | Packet presented | [GATE-B.md](understanding-packets/GATE-B.md) |
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

### Gate A — semantic IR/compiler boundaries

| Field | Value |
|---|---|
| Level | L2 |
| Implementation status | Reached |
| User status | Packet presented |
| Reached at | `c5d23d8`, `b30777e`, `8d4ed65`; S01 evidence; 2026-08-15 |
| Packet | [understanding-packets/GATE-A.md](understanding-packets/GATE-A.md) |
| Debt distance after update | 1 |

**Mental model:** Pending user explanation; do not self-certify.

**Can explain:** Pending the five packet answers.

**Can predict:** Pending the operation-kind experiment and answer 2.

**Can trace:** Pending the `hidden [2,4]` execution trace in answer 3.

**Can diagnose:** Pending answer 4 and artifact/plan failure diagnosis.

**Hands-on experiment:** Pending the semantic operation-kind change and `ctest` result.

**Five answers:** Not yet provided.

**Remaining unknowns:** Whether the user can distinguish semantic meaning, target lowering, and physical execution boundaries. This remains a study checkpoint, not an implementation blocker under D-014.

**Agent assessment:** `in progress`, packet presented 2026-08-15; retained as a study checkpoint under D-014.

### Gate B — one-token transformer execution

| Field | Value |
|---|---|
| Level | L2 |
| Implementation status | Reached |
| User status | Packet presented |
| Reached at | `023c505`; S02-03 evidence; 2026-08-26 |
| Packet | [understanding-packets/GATE-B.md](understanding-packets/GATE-B.md) |
| Debt distance after update | 2 under explicit D-014 autonomous-execution override |

**Mental model:** Pending user explanation; do not self-certify.

**Can explain:** Pending the five packet answers.

**Can predict:** Pending the kernel-ID/scale experiment.

**Can trace:** Pending the `[4]` host-to-RMSNorm-to-output trace.

**Can diagnose:** Pending the operand/catalog/poisoning diagnosis answers.

**Hands-on experiment:** Pending the command-ID change and observed output difference.

**Five answers:** Not yet provided.

**Remaining unknowns:** Qwen-derived hidden/KV shapes and prefill/decode entry semantics are pinned
in S03-01; the current packet uses a four-element executable fixture.

**Agent assessment:** `in progress`, packet presented 2026-08-26; retained as an unpassed autonomous
study checkpoint under D-014.

## Autonomous execution override

On 2026-08-26 the user explicitly directed implementation to continue without waiting for
understanding-gate passage. Gate packets, exercises, ledger records, and phase/gate commits remain
mandatory evidence and are not to be marked passed by the agent. This override permits later gates to
be reached while earlier gates remain unpassed; the resulting debt is visible and intentional.

## Update Discipline

- Never replace the user's words with a generic agent summary when assessing an L2 gate.
- Preserve earlier gate records; append corrections and later refinements.
- A phase summary is incomplete until its `UNDERSTANDING STATUS` output agrees with this ledger and `.planning/STATE.md`.
