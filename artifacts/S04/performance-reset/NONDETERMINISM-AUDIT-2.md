# Whole-model nondeterminism audit — round 2 (narrowing)

**Status: NOT root-caused. Two hypotheses closed, one new and important characterisation established.
Production behaviour unchanged.**
git_sha: 08cf885 + this work
gpu: RTX 5090 index 1 (CUDA_VISIBLE_DEVICES=1)

## Phase 1 — per-launch function-attribute mutation (tested, NOT the cause)

`launch_attention_bf16_cache` called `cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)` and
`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, cached_bytes)` **on every launch**, i.e. once per
attention layer per token, while earlier launches of the same function could still be in flight. That
is both unnecessary for Qwen's declared capacity (4096 positions -> 32768 B dynamic + ~1 KiB static,
inside the ordinary 48 KiB limit) and a genuine host-side code smell.

**Change made:** removed both calls; the cached path now launches directly, guarded by a
once-per-process query of the *default* per-block limit (`cudaDevAttrMaxSharedMemoryPerBlock`) with the
uncached kernel as fallback. Kernel arithmetic untouched.

**A/B result:** with the mutation removed, `seeded-long-103` across 8 fresh processes gave
**3 distinct payload hashes** (`6d43d1e9` x6, `fcd093a8`, `1f998e38`). Nondeterminism persists.
**Verdict: exonerated.** The removal is kept as a correctness cleanup only.

## Phase 2 — kernel-family elimination (confounded; no family implicated)

Three diagnostic-only gates added (production default unchanged, all off):

| gate | env | what it selects |
|---|---|---|
| cached attention off | `SUPERINFER_QWEN38_REFERENCE_ATTENTION` | `grouped_attention_bf16_cache` (uncached) |
| register GDN off | `SUPERINFER_QWEN38_REFERENCE_GDN` | `gated_delta_attention_parallel_f32` |
| multi-block conv off | `SUPERINFER_QWEN38_REFERENCE_CONV` | `<<<1,256>>>` grid-stride conv |

Fresh-process `seeded-long-103` payload hashes:

| configuration | runs | distinct hashes |
|---|---:|---:|
| production baseline | 13 | 3 (2 anomalous runs) |
| cached attention off | 10 | **1** |
| multi-block conv off | 5 | **1** |
| register GDN off | 5 | 2 (1 anomalous run) |

**These numbers do not implicate a family, because the experiment is confounded.** The failures
cluster in the *first* runs after a period of GPU idleness and disappear once the device is warm:
in the round-1 session the baseline's 4 anomalous runs were runs 1-4 with runs 5-8 identical; this
session the baseline's anomalies were runs 2 and 5 of the first block, and every later block
(baseline included) ran 5/5 clean. Whichever gate is exercised first therefore absorbs the "cold"
window. The single GDN-off anomaly occurred in the first block; the 10 later attention-off runs and
the 5 later baseline runs were all clean.

**Characterisation established:** the event is rare, clustered at the start of a run sequence, and
**dependent on machine/device state rather than on any kernel family**.

## New and important observation — the *stable* result is itself environment-dependent

With an identical runtime and artifact, the default single-token forward pass produced:

* previous session epoch: `logit=11.6875 checksum=-791994` (repeated)
* this session epoch: `logit=11.625 checksum=-792302` (6/6 fresh processes, perfectly stable)

`nvidia-smi` in this epoch fails with `Failed to initialize NVML: Driver/library version mismatch`
(kernel module 595.84, userspace NVML 595.91), i.e. the **driver stack was updated between the two
epochs**. A single forward pass from a fresh process with zero-initialised state has no cross-token
carry-over, so a change in its logits means the computation depends on machine state even for one
step. This is a much stronger lead than the long-continuation race and is the recommended first
target for the next session.

## Phase 8 — workspace audit (closed)

Every kernel the production providers advertise returns `workspace_bytes = 0`
(`sm120.nvfp4-gemv-rows` k29, `sm120.linear-f32-rows` k30, and all `sm120.baseline` ids). The only
non-zero-workspace candidates belong to the experimental native-MMA provider (k27/k28), which is not
selected in production. The specializer takes the max over candidates, so
`plan.resources().workspace_bytes = 0` and no command ever has `workspace_size != 0`. **No active
command consumes workspace**, so the fact that `workspace_` is not covered by the arena-poison
experiment is moot. Hypothesis closed.

## Correction honoured

The arena-poison result proves only that the outcome is not controlled by the initial byte pattern of
the **device arena**. Shared-memory read-before-write, local/register undefined values, out-of-bounds
access, workspace dependence (now closed), partial producer writes, intra-kernel races and
launch-configuration interactions all remain formally open. Nothing here narrows them further.

## What is NOT established

The first nondeterministic operation. No token/layer/command/kernel/buffer has been identified. Phase-3
state hashing, Phase-4 layer bisection and Phase-5 command localization were not executed: with the
event absent from every long run once the device is warm, and with the state-hash/bisection tooling
requiring fresh-process pairs at ~50 s each, neither could be driven to a signal inside this budget
without first restoring a reliable reproducer.

## Recommended next step

Stop trying to catch the long-continuation race directly. Instead exploit the **epoch-level
reproducer** established above: a single default forward pass, ~28 s, stable within an epoch and
different across epochs. Dump every kernel's output on that one step in two epochs and diff
command-by-command to find the first environment-dependent operation. That is a bounded, cheap,
deterministic A/B with no warm-up confound.
