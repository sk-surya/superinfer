# D-021 / RMSNorm reduction audit

**Status: EVIDENCE COMPLETE. Production unchanged (serial RMSNorm retained). D-021 NOT modified.**
git_sha: 11050d5 baseline + env-gated experimental parallel path
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)

## What was tested

Two RMSNorm reductions, selected at runtime (`SUPERINFER_QWEN38_RMS_PARALLEL`), serial still default:

* **serial** (production): thread 0, single FP32 accumulator, strictly increasing index order,
  `float4` shared reads.
* **parallel** (E0b form): strided FP32 partial per thread -> warp shuffle reduction ->
  ordered warp-partial accumulation -> `sqrtf(total/K + eps)`.

Plus a diagnostic-only FP64 audit (`SUPERINFER_QWEN38_RMS_AUDIT`) that, on the *real* norm rows the
model feeds in a live run, also computes a double-precision denominator and accumulates
`|serial - fp64|` and `|parallel - fp64|`.

## A decisive implementation bug was found and fixed

The historical rejection rested on a bug, not on the reduction order. In the restored parallel path,
`denominator_slot = parallel_mode ? parallel_denominator : serial_denominator;` was executed by **all**
threads, but `parallel_denominator` is only defined on thread 0. Threads 1..255 therefore wrote `0.0`
over the shared denominator, so the norm divided by zero. Fixed by guarding the store with
`if (threadIdx.x == 0)`.

With the guard in place the parallel path is indistinguishable in output from serial on chat-60
(60/60, byte-identical tokens) and passes D-021. The earlier "3 strict-row flips" are explained by the
clobber, not by FP32 association order.

## FP64 oracle (Phase 1–2)

172,860 real norm rows captured in one 60-token continuation (2,881 rows/token across the 209 norm
launches, including the 48-wide GDN output norms):

| | mean abs(denom − fp64) | mean rel(denom − fp64) |
|---|---:|---:|
| serial | 8.777137e-08 | 8.484164e-08 |
| **parallel** | **9.275058e-09** | **3.041187e-08** |

**The parallel reduction is 9.5x closer in absolute error and 2.8x closer in relative error to a
double-precision RMSNorm than the serial order.** The serial FP32 order is not the mathematical
oracle; it is one association order with measurably larger rounding error.

## D-021

| run | verdict | strict rows | strict failures | greedy mismatch rows |
|---|---|---:|---:|---:|
| serial, `--repeat 1` (baseline) | pass | 240 exact | 0 | 11 |
| **parallel, `--repeat 1`** | **pass** | **240 exact** | **0** | 8 |
| serial, `--repeat 2` | inconclusive | 240 exact | 0 | 12 |
| parallel, `--repeat 2` | inconclusive | 240 exact | 0 | 8 |

Both implementations resolve 240/240 strict rows greedy-exact with zero strict failures, and both pass
all 8 cases. The `--repeat 2` "inconclusive" verdict is the **pre-existing whole-model session
non-determinism**, not a property of the reduction: the two paths fail repeatability on *different*
cases (parallel -> `chat-template`, serial -> `seeded-long-103`), and the serial production path — the
one that was never accused — fails it too. The historical serial baseline behaved identically.

Because there are **no strict-row flips** once the bug is fixed, the requested Phase-4
first-divergence localisation and Phase-5 flip-margin analysis have no subject: there is no flipped
row to localise.

## Performance: the premise no longer holds

| | serial | parallel |
|---|---:|---:|
| RMSNorm ms/token | **4.036** | 4.208 |
| kernel sum ms/token | **22.144** | 22.339 |
| device span ms/token | 23.34 | 23.50 |
| chat-60 | 60/60 | 60/60 (identical to serial) |

The historical `5.26 -> 2.23` was measured against the *un-optimised* serial kernel. The serial path
has since been optimised (phase-2 `float4` shared reads), and the parallel reduction is now **4.3%
slower** — its extra warp shuffles and `__syncthreads` cost more than thread 0's vectorised chain
saves at K=5120 with 256 threads. The ~1.8 ms opportunity this audit was commissioned to adjudicate
**does not exist on the current tree**.

## Verdict — OUTCOME C, with a corrected cause

**Retain serial RMSNorm. D-021 unchanged. No D-023 proposed.**

Reasoning, strictly from the evidence:

* The numerical question resolves **against** the original suspicion: the parallel reduction is
  *more* faithful to FP64, and D-021 does **not** reject it (240/240 strict rows exact, zero strict
  failures). There is therefore nothing for D-021's strict-greedy rule to over-constrain here, and no
  contract change is justified.
* The production decision nonetheless stays on serial, for a different and simpler reason: the
  parallel reduction is **slower** on the current tree, so promoting it would be a regression.
* The historical "parallel RMSNorm failed D-021" record is corrected: it was an implementation bug
  (`denominator_slot` clobber), not a statement about FP32 association order.

## Repeatability (Phase 7)

Same-configuration captures: chat-60 produced identical token sequences across fresh processes for
both paths, serial == parallel. Within-invocation `--repeat 2` payload comparison is non-deterministic
for **both** paths on one case each — a pre-existing environment/whole-model issue that must be fixed
before any future run can use payload-hash repeatability as a gate. It is not a property of the
reduction and does not affect the verdict.
