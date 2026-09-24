# Whole-model nondeterminism audit (D-021 repeatability blocker)

**Status: runner defect FIXED; model nondeterminism REPRODUCED and NARROWED but NOT yet localised to
a single kernel. Production behaviour unchanged.**
git_sha: f0cb502 baseline + harness fix
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)

## 1. Harness defect fixed (was masking the real problem)

`tools/qwen38_s03r_acceptance.py` spawns a **fresh executable process per repeat**
(`subprocess.run`), so `--repeat 1` is one capture. One hash trivially satisfies
`len({hash}) == 1`, so a run that *requires* repeatability could report a verdict from a single
execution. Fixed: the decision is now extracted into `repeat_semantics(contract, d021_present,
repeat)`; when repeatability is required and `repeat < 2` the verdict is `inconclusive` with
`"D-021 repeatability requires at least two fresh process captures"`. The requested repeat count is
never silently widened. Verified end to end: `--repeat 1` now yields `inconclusive` for both the
same-artifact decision and the D-021 verdict. Unit coverage:
`tests/unit/test_qwen38_s03r_repeat_semantics.py` (7 tests).

Also added `--retain-repeat-captures DIR`: retains every per-iteration capture, stdout/stderr and, when
repeats differ, a `mismatch.json` carrying first differing byte / FP32 element / row / vocab index,
both values, absolute difference, row max-abs and differing-element count.

## 2. Reproduction (cross-process)

`seeded-long-103`, 8 fresh processes, identical inputs:

```
run1 c9d0f8ff  run2 e2a69460  run3 2ea280d8  run4 b683ce6d
run5 6d43d1e9  run6 6d43d1e9  run7 6d43d1e9  run8 6d43d1e9     (5 distinct)
```

The first four differ; the last four are byte-identical. Same shape appeared twice more
(first run of a sequence differed, later runs agreed), so this is **not** a 50/50 coin: it clusters at
the start of a session.

First divergence versus the canonical run is at a **different row each time**:

| run | first differing row | abs diff | row max-abs | differing elements in row |
|---|---:|---:|---:|---:|
| 1 | 58 | 1.719 | 16.72 | 247,700 |
| 2 | 86 | 0.406 | 14.13 | 247,434 |
| 3 | 29 | 3.375 | 16.69 | 248,160 |
| 4 | 64 | 0.094 | 16.44 | 248,149 |

Nearly the whole logits vector differs once it starts (≈248k of 248,320), and the row varies run to
run: this is a **state divergence whose onset is timing-dependent**, not a rounding difference.

## 3. Ruled out (each by direct measurement)

| hypothesis | experiment | result |
|---|---|---|
| uninitialised arena read / read-before-write | `SUPERINFER_QWEN38_ARENA_POISON` = 00 vs FF, 2 fresh runs each | **same poison varies** (00 run1 `8ac527bb` ≠ 00 run2 `6d43d1e9`); different poisons agree. Poison-independent. |
| activation-slot aliasing (lifetime reuse) | `SUPERINFER_QWEN38_DISABLE_ACTIVATION_REUSE=1`, 4 fresh runs | nondeterminism **persists** (3 distinct hashes). |
| `float atomicAdd` ordering | grep of the executor | **no atomics** anywhere in production kernels. |
| state-init racing the first kernel | inspected `copy_to_device` | uses **synchronous** `cudaMemcpy`; no race. |
| inter-stream synchronisation | specializer emits `stream = 0` for all commands | effectively single-stream; no event/wait path taken. |
| a simple single-step kernel hazard | GDN layer-0 fixture ×4, layer-3 fixture ×4 | **perfectly stable**, identical metrics every run. |

## 4. Not yet established

The failing operation is **not localised to one kernel or buffer**. Candidates remaining, in priority
order, are the multi-step state-carrying paths only (they are the ones the single-step fixtures do not
exercise at length):

* KV cache growth / `cache_append_f32_bf16` at the active-length boundary,
* `grouped_attention_bf16_cache_cached` over a growing `positions` window,
* `gated_delta_attention_register_f32` (which was **rewritten this cycle** with per-thread
  register columns) across many continuation steps,
* `causal_conv_silu_f32` convolution-state advance.

The single-step GDN and layer fixtures exercise the last two once and do not reproduce it, which
argues for the *growth*-dependent paths (KV/attention window) rather than a static per-launch hazard —
but that is an inference, not a measurement.

## 5. Why prior validation did not catch it

D-021 has been run many times and mostly agreed, because the failure clusters in the first runs of a
session and disappears once the device is warm. The harness's `repeat=1` hole meant several of those
runs were not actually testing repeatability at all. The historical "one unreproduced anomaly on loop
5, divergence at row 20; 9+ subsequent runs byte-identical" is the same signature.

## 6. Verdict

**Mission outcome B.** Root cause is not established and no fix is claimed; the failure is reproduced
with retained byte-exact evidence, is poison-independent, reuse-independent, atomic-free and
stream-free, and is confined to the full-model multi-step continuation path. The harness defect that
was hiding it is fixed, and `--repeat 1` can no longer report a repeatability-qualified verdict.

Production behaviour is unchanged (serial RMSNorm retained; kernel sum ~22.14 ms/token).
