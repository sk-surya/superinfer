# S04 Autoresearch Decision — Minimum Automatable Loop (after 3 manual loops)

**Status:** decision record. Do not build a generic research platform; automate the proven loop.

## The three completed manual loops

| # | Target | Change class | Local | E2E |
|---|---|---|---|---|
| 1 (R02) | `nvfp4_linear_f32` | occupancy: single-block -> row-parallel grid | 33x | 7.6–9.4x |
| 2 (S04-P1) | `grouped_attention_bf16_cache` | algorithmic: cache scores, remove `O(head_dim²)` pass | 4,320x | 1.72–2.41x |
| 3 (S04-P2) | `nvfp4_linear_rows_f32` | instruction/latency: vectorized loads, hoist block-scale decode | 5.48x | 1.57–1.68x |

## What is actually common

Every loop had the same shape, and it was small:

1. **Fixed profile harness** — Nsight Systems on the same corpus/GPU; rank kernels by GPU-time share.
2. **Mechanical target choice** — take the #1 share unless a smaller share is provably on the critical path.
3. **One-kernel, one-launch-site change** — a faster variant added beside the incumbent, behind a launch guard.
4. **Bit-exact differential** — the new variant must equal the incumbent bit-for-bit (possible in all three because the per-row/per-position operation order was preserved). This is the single highest-value property: it makes regression detection trivial and independent of model-level tolerance.
5. **Fixed E2E benchmark** — chat-60 / long-103 wall times under the same manifest.
6. **D-021 model gate + second fresh session** — sign reproduction and byte-identity.
7. **Promote or reject; incumbent retained** — rollback is one launch-site revert.

What varied: the *hypothesis* (occupancy vs algorithmic redundancy vs instruction/latency) and the
*evidence source* (share table, roofline arithmetic, source inspection). What did not vary: steps.

## Minimum autoresearch machinery (v1) — automate only the proven loop

A single experiment-runner that, given an incumbent commit and a target kernel, produces a
promotion/rejection decision. Explicitly **not** a general search system.

Required components (minimum):

1. **Profile step:** run the fixed benchmark under Nsight Systems; emit the ranked kernel table
   (`tools/nsys_qwen_profile.py` already does most of this). Output is the target suggestion, not a
   decision.
2. **Candidate isolation:** one git worktree per candidate; one kernel + one launch-guard diff.
   Candidate declares \(\{target\ kernel, local\ microbenchmark, expected\ direction\}\).
3. **Bit-exact differential gate:** run the incumbent-vs-candidate unit differential first; a
   non-bit-exact candidate is rejected unless it declares a tolerance rationale. (All three loops
   passed bit-exact, so this is the default and cheapest gate.)
4. **Correctness gate:** D-021 corpus verdict + capture hash comparison against the incumbent.
5. **Benchmark gate:** same manifest before/after; require a positive E2E delta and a second-session
   sign reproduction; record local and E2E deltas separately.
6. **Decision record:** promote (one launch-site commit) or reject (delete worktree), with raw
   evidence, exactly as the three manual summaries did.

Deferred (do not build yet): candidate *generation* (LLM/synthesis), parameter search, statistical
promotion bands, distributed scheduling. Loop 4 will run manually again; the runner is extracted from
three concrete instances, not invented.

## Is the ladder still worth continuing manually?

Yes. The post-loop-3 profile has credible single changes well above the 15% E2E threshold:
`gated_delta_attention_f32` (54.1%), `rms_norm_f32_bf16_scale` (15.2%, single-threaded `<<<1,1>>>`),
and NVFP4 still ~19x off its remaining roofline. Checkpoint target (>=5 decode tok/s) is not yet met
(current ~1.33 tok/s). Continue the profiler-driven ladder; extract the runner after loop 4 or 5.
