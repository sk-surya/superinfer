# S04-P9 — Activation-quantization quality recovery (native SM120 NVFP4 MMA)

**Status:** open. P9-RQ classification B is accepted **only for the current one-level
`block-16 amax/6 -> UE4M3 -> E2M1` activation quantizer**. It does not close the SM120 native NVFP4
Tensor Core architecture: the MMA mechanism is proven, deterministic, and materially faster; the
unresolved problem is the **activation representation**.

Nothing in this phase touches startup/TTFT, `linear_f32`, Flash-Next, repacked production weights, or
CUDA-core NVFP4 tuning.

## Gate semantics (removes the P8RQ ambiguity)

- **D-021 is the decisive MODEL-LEVEL margin gate.** It is necessary, not sufficient.
- **Classification A requires ALL binding gates to pass**, each appropriate to the numerical mechanism:
  1. projection/operator correctness contract (real-projection differential for the activation mechanism);
  2. layer-3 differential (threshold unchanged);
  3. GDN differential (threshold unchanged);
  4. full D-021 corpus (unchanged);
  5. determinism / reproduction (same-binary repeatable, fresh-session reproduced).
- **A D-021 pass does NOT override a layer/GDN failure.** P8RQ is exactly that case: D-021 margin verdict
  passed while layer-3/GDN failed, and classification was B.
- **No existing threshold may be loosened during P9.**
- If all industry-standard NVFP4 activation recipes show good downstream quality but systematically fail a
  local differential threshold, the agent must STOP and return that evidence to master review: deciding to
  create a separate approximation-quality contract is a policy decision the agent must not make.

## Priority order (sequential; stop early on success)

P9-0 sensitivity map -> P9-1 canonical hierarchical NVFP4 -> P9-2 adaptive 4/6 -> P9-3 hybrid precision
allocation -> P9-4 residual FP4.

## Performance accounting rules

- Use the complete derived **401-command** census (`tools/qwen38_nvfp4_census.py`).
- Per candidate report separately: activation reduction/global-amax cost; block quantization cost; MMA
  cost; epilogue/global-scale cost; total NVFP4 subsystem ms/token; projected whole-model bound;
  integrated GPU ms/token once available.
- **Never** quote projection-subsystem throughput as model tok/s.
- No repacked production weight layout during P9.

## Promotion target

All correctness gates green, native Tensor Core MMA used for a **substantial fraction** of the 401
projections, materially lower GPU ms/token than P7, and a reproduced integrated speedup. Recovering
quality at a trivial native fraction is a negative result to record and move past.
