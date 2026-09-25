# BLOCKED_ENVIRONMENT_REPAIR_REQUIRED

**The NVIDIA driver stack on this machine is incoherent. No SuperInfer GPU correctness evidence may
be produced until it is repaired by the owner.**

## Exact mismatch

| component | version |
|---|---|
| loaded kernel module (`/proc/driver/nvidia/version`) | **595.84** |
| on-disk kernel module (`modinfo -F version nvidia`) | **595.91.07** |
| userspace NVML | **595.91** |
| installed packages | `nvidia-driver-595-open 595.91.07-0ubuntu0.26.04.1`, `libnvidia-compute-595 595.91.07-0ubuntu0.26.04.1` |

`nvidia-smi` fails:

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 595.91
```

The running kernel module (595.84) predates the on-disk 595.91.07 driver package. CUDA itself still
initialises (`cudaGetDeviceCount = 2`, runtime 13.1, driver API 13.2, both RTX 5090s visible), which is
precisely why the previous session was able to keep producing numbers — and precisely why those
numbers cannot be trusted as environment-stable.

## Why this blocks the mission

The sprint's stated precondition is: *"same exact executable, same exact artifact, same exact driver
stack, same GPU, same environment"*. The driver stack is not coherent, and the previous session's
reported cross-epoch change in the first logit (`11.6875` -> `11.625`) coincided with this driver
package upgrade. Until the stack is coherent, any determinism result would be measuring the state of
the driver installation rather than a property of SuperInfer.

Consequences already established from the previous cycles:

* the token-0 checksum of the **same binary and artifact** differed across epochs (`-791994` vs
  `-792302`), so neither value is usable as an oracle;
* the `seeded-long-103` anomalies cluster at the start of a run sequence, which is exactly the
  signature a half-updated driver stack would produce.

## Required repair (owner action — I did not and will not do this)

Reboot the machine (or unload and reload the `nvidia` kernel modules). A module reload would terminate
the user-owned `ninfer-serve` process (pid 368383, GPU 0), so **reboot is the safe option**. I did not
touch `ninfer-serve`, did not attempt `rmmod`, and did not reboot.

## After repair — re-entry checklist

1. Re-run Phase 0 and regenerate `artifacts/S04/performance-reset/ENVIRONMENT-LOCK.json`; require
   `nvidia-smi` to succeed and loaded module == on-disk module == userspace.
2. Freeze the binary: record `sha256` of `superinfer_sm120_qwen38_e2e_artifact` and of the `.sinf`,
   and verify both before every run. **Do not rebuild.**
3. Same-binary one-step repeatability, >= 8 fresh processes (expect `SINGLE_STEP_REPEATABLE`).
4. Same-binary `seeded-long-103`, >= 8 fresh processes, retained captures.
5. Then, and only then, module-loading A/B (`CUDA_MODULE_LOADING=LAZY|EAGER`) and the localization
   plan.

## Work that WAS completed before the gate (environment evidence only)

* Full Phase-0 capture (above) and `ENVIRONMENT-LOCK.json`.
* Binary provenance: the E2E executable embeds **both** a native `sm_120a` cubin and `sm_120a` PTX
  (`cuobjdump --list-elf` / `--list-ptx`) — so the "CMake config vs actual binary contents" question is
  answered, and a JIT path does exist even though a native image is present. This makes the
  `CUDA_DISABLE_PTX_JIT` / `CUDA_FORCE_PTX_JIT` diagnostics meaningful once the environment is fixed.
* Executable sha256 `5ec2c66d0fef59b20275e068b4c2f386b4de377478bf3bf2ddc83774c8a6bf54`,
  artifact sha256 `e25022c8592875449968b9d0b1f56e6800971e0ba04d8a43eec980fe60dc65d5`.

No model run, benchmark, or correctness result was produced in this session.
