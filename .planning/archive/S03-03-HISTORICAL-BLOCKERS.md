# Archived State — S03-03 Numerical Archaeology (historical, superseded)

**Archived:** 2026-09-11
**Reason:** These notes were presented as "current state" in `.planning/STATE.md` after S03 closed. They are retained as historical evidence but are **not current**. S03 closed via S03-R Outcome A under D-021; the model-level `max_abs <= 0.5` source-reference gate they describe is superseded. All local kernel/layer differential gates named here remain binding and unchanged.

The following text is preserved verbatim from the pre-cleanup `STATE.md` current-state/blockers sections.

## Archived "current focus" text (misleading, was pre-S03-R)

S03 is closed: S03-R Outcome A passed session-2 under D-021 (240 strict rows greedy-exact across two byte-identical sessions, 12 near-tie rows in-set, distributional bounds clear, 66 listed long-103 outlier rows reported; all kernel/layer gates unchanged). R01 is the active lane: measure the current unoptimized runtime (decode tok/s + TPOT, prefill tok/s + TTFT, peak VRAM, kernel/region breakdown) before any optimization. No S03 numerical work remains open.

## Archived S03-03 deployment/storage probe narrative

The current S03 reference is pinned to the actually qualified Transformers `5.12.1` / torch `2.13.0+cu130` environment. Its deployment-storage correction models cached GDN decode from position zero, current-row BF16 convolution rounding, and BF16 embedding/final-norm I/O. Five fresh 13-token chat-prefix target processes are byte-identical, but the corrected full-model oracle still has accumulated max-abs logit outliers above the unchanged `0.5` contract. See `artifacts/S03/qwen38-reference-deployment-storage-probe.json` and `artifacts/S03/qwen38-chat-prefix13-repeatability.json`.

The deployment-v8 acceptance rerun completed two fresh target sessions on GPU 0. Plain-short, Unicode, and special-token cases pass numerical/token/repeatability checks; chat-template and varied-length match all greedy tokens and are repeatable but fail the unchanged numerical contract. Exact results are in `artifacts/S03/qwen38-s03-deployment-v8-acceptance.json`. Post-attention/state localization at varied-length step 36 shows deterministic accumulated drift before the MLP, a layer-42 amplification, and no isolated recurrent-state corruption; see `artifacts/S03/qwen38-post-attention-state-localization-v8.json`.

An explicit layer-42 physical-output trace covers the GDN gated-normalization/output path, token-mixer residual, post-attention norm, MLP, and final residual. The recurrent core remains close to the independent oracle, while the deterministic upstream difference is amplified through the gated path; the target residual self-check passes within max `0.0198853`. See `artifacts/S03/qwen38-layer42-post-path-localization-v9.json`.

The pre-closure acceptance review is recorded in `.planning/phases/S03-qwen38-e2e/S03-03-REVIEW-LATEST.md`.

## Archived solver-blocker narrative (S03 numerical archaeology)

- A BF16-KV reference diagnostic changes but does not explain the remaining long-replay logit outliers; selected-hidden tracing localizes row 29's first proven mismatch before LM-head projection. Per-layer post-MLP tracing shows gradual accumulated drift through the decoder rather than one catastrophic layer. See `artifacts/S03/qwen38-layer-boundary-localization.json`.
- An operation-level GDN diagnostic matches layer-0 packed-NVFP4 qkv projection at `1.38283e-5` max error and post-convolution output at `0.00198197`; the recurrent core at `3.6478e-5`; the normalized/gated core output at `0.00436258`. See `artifacts/S03/qwen38-gdn-operation-localization.json`.
- All-layer boundary reduction shows the first material post-token-mixer jump at layer 3, the first full-attention block (`max_abs=0.070608`). See `artifacts/S03/qwen38-full-attention-jump-localization.json`.
- Standalone layer-3 decode over 30 positions passes with BF16 KV (`final_hidden max_abs=0.00111389`). See `artifacts/S03/qwen38-layer3-long-context-differential.json`.
- FP64 NVFP4 accumulator probe: byte-identical 30-token capture, reverted. See `artifacts/S03/qwen38-nvfp4-double-accumulation-probe.json`.
- FP64 RMSNorm reduction probe worsens the staged layer differential; rejected. See `artifacts/S03/qwen38-rmsnorm-precision-probe.json`.
- Layer-0 GDN differential passes 30 state-continuing segments (`max_abs=0.035728455`). See `artifacts/S03/qwen38-gdn-long-context-differential.json`.
- Layer-42 target-input replay matches QKV `0.01714611`, convolution `0.00386417`, recurrent-core `0.00003777`, complete-layer `0.12633133`; the full-model cliff is upstream amplification. See `artifacts/S03/qwen38-layer42-input-amplification.json`.
- Layer-42 GDN `in_proj_z` matches at max `0.01321268` / RMSE `0.00264875`; standalone z hypothesis rejected. See `artifacts/S03/qwen38-gdn-z-projection-localization-v11.json`.
- 30-token chat replay with activation lifetime reuse disabled is byte-identical; activation aliasing rejected. See `artifacts/S03/qwen38-activation-reuse-localization-v12.json`.
- Exact-target-hidden LM-head replay differs by max `0.03125`; final projection ruled out. See `artifacts/S03/qwen38-lm-head-localization-v13.json`.
- Rejected precision hypotheses: reference per-layer BF16 output rounding (`artifacts/S03/qwen38-layer-output-rounding-probe.json`); CUDA `--fmad=false` (`qwen38-fmad-off-probe.json`); Kahan accumulation (`qwen38-kahan-probe.json`); semantic BF16 boundary rounding (`qwen38-semantic-boundary-rounding-probe.json`); NVFP4 pairwise reduction (`qwen38-nvfp4-pairwise-probe.json`); BF16 weight round-trip (`qwen38-bf16-weight-roundtrip-probe.json`); materialization/binding (`qwen38-materialization-localization-v14.json`); RMSNorm `rsqrt` (`qwen38-rmsnorm-rsqrt-probe-v15.json`); reference TF32 disable (`qwen38-reference-tf32-probe-v16.json`).
- Diagnostic command-trace API accepts explicit `(command_id, buffer_id)` result requests. See `artifacts/S03/qwen38-command-trace-output-contract.json`.

## Resolution

S03-R built an independent same-artifact oracle over exact `.sinf` packed bytes and proved the residual model-level error is quantized-deployment conditioning, not a SuperInfer defect. See `.planning/phases/S03-qwen38-e2e/S03-R-SUMMARY.md` and D-021. No local/layer gate was weakened.
