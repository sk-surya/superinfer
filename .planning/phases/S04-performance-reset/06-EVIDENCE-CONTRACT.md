# Performance Reset — Minimum Sufficient Evidence Contract

## Purpose

Prevent two failure modes:

1. shipping fast incorrect kernels;
2. spending most of the sprint proving things that do not change the next action.

Evidence is proportional to decision risk.

## Evidence tiers

### Tier 0 — compile/shape sanity

Run continuously during implementation.

- build target;
- launch/error check;
- sentinel/finite outputs;
- expected shape coverage.

This is not promotion evidence.

### Tier 1 — local same-recipe differential

Required before an implementation replaces an existing command.

Use real captured activations where possible.

Report:

- max_abs;
- mean_abs;
- rel-L2;
- deterministic repeat;
- shape/role.

Reduction-order changes may use tolerance-qualified equality. Never require bit identity solely because the old kernel was serial.

### Tier 2 — integrated smoke

Required every time an entire donor family or fusion group is wired.

Run:

- representative layer-3;
- representative GDN layer/state continuation;
- short forced-token full-model continuation;
- one D-021 smoke subset.

If this fails, fix now. Do not benchmark around it.

### Tier 3 — milestone qualification

E0a:
- Tier 1 + Tier 2;
- full model timing;
- nsys summary.

E0b:
- Tier 1 + Tier 2;
- fused/unfused equivalence;
- whole-model timing;
- short D-021 smoke.

Week gate:
- full tools/validate.py --full;
- full D-021;
- fresh-process repeatability;
- real autoregressive GPU feedback;
- same-machine external comparison;
- one long-context steady-state point;
- nsys evidence.

Do not run Tier 3 after every small edit.

## Quantization recipe separation

Three questions must never be conflated:

1. Is the implementation of a quantization recipe correct?
2. How far does that recipe move local activations/logits?
3. Is model quality acceptable?

P9's two-level path does not currently answer question 1 because the implementation passes raw global amax as s_global rather than applying the documented /(448*6) transform.

Therefore:

- P9 findings remain historical;
- W4A4 quality is unresolved;
- none of that blocks W4A16 E0;
- if W4A4 returns later, first repair/verify the canonical recipe independently.

## Performance truth

Every milestone performance record must include:

- git SHA;
- GPU model/index;
- relevant clocks/power state if controlled;
- CUDA/driver;
- exact artifact/checkpoint identity;
- context position;
- token count;
- warmup;
- median and, where cheap, p95;
- device span ms/token;
- wall decode ms/token;
- tok/s;
- projection subtotal;
- launches/token;
- top kernels;
- streamed-byte estimate and effective GB/s;
- external comparator command/result.

## Cache realism

An isolated matrix smaller than L2 is not evidence of full-model bandwidth.

At least one of these must accompany a projection performance claim:

- explicit L2 scrub larger than cache;
- rotating matrices larger than cache;
- complete 401-projection traversal;
- integrated whole-model execution.

Integrated whole-model evidence is preferred.

## Benchmark budget

A benchmark run should answer a decision already on the table.

Do NOT spend time on:

- exhaustive context sweeps during E0;
- multiple external engines after SparkInfer + one secondary reference establish the bar;
- speculative methods before ordinary decode is fast;
- repeated profiler captures when the top cause is already obvious;
- polishing plots.

If two implementations differ by less than 10% and one is already integrated/maintainable, choose it and continue.

## Correctness authority

P7 remains the implementation oracle/fallback for E0 where the intended recipe is unchanged.

D-021 remains the model-level deployment contract.

Intentional future recipe changes require their own quality qualification; do not weaken implementation gates to hide recipe error.

## User-owned GPU safety

Before any GPU command:

- inspect active processes;
- choose a free device;
- never kill/migrate/renice a user-owned workload;
- never assume GPU 0 is free;
- record CUDA_VISIBLE_DEVICES.

P2P tests require both GPUs free. If they are not, skip the probe.
