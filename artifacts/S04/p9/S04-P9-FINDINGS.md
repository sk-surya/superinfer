# S04-P9 — activation-quantization quality recovery: findings and STOP

**Status: STOPPED for master review, per the P9 rule on local-gate vs intentional approximation.**

## What was implemented (P9-1)

Canonical two-level (hierarchical) NVFP4 activation scaling, kernel 28
(`nvfp4_activation_global_amax_f32` -> `nvfp4_activation_quantize_two_level_f32` ->
`nvfp4_linear_mma_f32`):

```
s_global     = global_amax / (448 * 6)                  (FP32, one value per activation vector)
s_block_real = (block_amax / 6) / s_global
s_block      = round_E4M3(s_block_real)                 (clamped, explicit zero handling)
q            = round_E2M1(x / (s_global * s_block))
output       = MMA(E2M1_w, E2M1_x, local_w, local_x) * weight_global_scale * s_global
```

The weight representation is unchanged. The activation global FP32 scale is applied in the epilogue;
the MMA still consumes only E2M1 + UE4M3. The global amax is reduced on device (one block, deterministic)
and read by the MMA kernel directly from workspace — no host round trip, no hot-path allocation. Kernel
28 is selected at specialization time via `SUPERINFER_QWEN38_NATIVE_NVFP4_TWO_LEVEL`.

## LAYER/GDN AFTER P9-1 (thresholds unchanged)

| gate | P7 | one-level amax/6 (k27) | two-level canonical (k28) |
|---|---:|---:|---:|
| layer-3 `max_abs` | 0.00106812 | 3.10462 | **1.65071** |
| layer-3 `mean_abs` | 3.559e-5 | 0.14473 | **0.138151** |
| GDN `max_abs` | 3.128e-4 | 6.044 | **5.89273** |
| GDN `mean_abs` | 3.079e-6 | 0.0458481 | **0.0515523** |

Threshold: layer-3 `max_abs <= 2e-2`, `mean_abs <= 2e-4`. Both recipes fail by ~80-300x. Two-level
improves layer-3 `max_abs` 1.88x but leaves `mean_abs` ~unchanged and slightly *worsens* GDN.

## Diagnosis (P9-1B branch: "barely improved")

The improvement is not material on the mean, so the residual is dominated by **E2M1 grid error**, not
scale encoding. E2M1 has only the magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`; a better block/global scale
cannot fix per-element grid error. This predicts that P9-2 (4-over-6) is another scale-mapping variant and
is unlikely to close a grid-dominated gap.

## P9-3 bounded hybrid (layer-3, two-level, one projection role at a time)

| native role | layer-3 `max_abs` | `mean_abs` |
|---|---:|---:|
| none (P7) | 0.00106812 | 3.559e-5 |
| k | 0.00106812 | 3.559e-5 |
| gate | 0.0982971 | 0.0181044 |
| up | 0.095384 | 0.0184475 |
| down | 0.202816 | 0.0434517 |
| q | 0.41349 | 0.0389822 |
| v | 0.458445 | 0.0962891 |
| o | 0.887596 | 0.0756109 |
| all | 1.65071 | 0.138151 |

`k` is exactly the P7 value because the layer fixture is a single-position decode: attention softmax over
one position makes the output independent of k, so that row is not evidence about k's sensitivity.
Every other role exceeds the gate alone; the least sensitive (`up`/`gate`, ~0.095) is still ~5x over.
Errors accumulate across roles, so a hybrid subset does not rescue the gate.

## Conclusion and requested decision

- The native SM120 block-scaled NVFP4 MMA mechanism is proven, deterministic, and materially faster
  (integrated chat-60 32.5-33.4 s vs P7 35.6-36.9 s at P8RQ; activation global-scale cost added here).
- **No W4A4 activation representation bounded here passes the unchanged layer-3/GDN gates**, including
  hybrid allocation down to a single projection class.
- Yet the **model-level D-021 margin contract passed** at one-level in P8RQ (240 strict rows
  greedy-exact).

That is exactly the case the phase rules reserve for master review: industry-standard NVFP4 recipes show
acceptable downstream/model quality while systematically failing a local differential threshold. The
agent must not create or weaken an approximation-quality contract. **Returning for that decision.**

Remaining bounded approaches not run (P9-2 4/6, P9-4 residual FP4) are recorded as open; the diagnosis
(E2M1 grid dominates) suggests residual FP4 (effective >4-bit activation) is the only one that could
address the dominant error term, at the cost of a second MMA pass over the same weights.
