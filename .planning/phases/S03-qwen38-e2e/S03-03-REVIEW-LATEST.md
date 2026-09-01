---
phase: S03-qwen38-e2e
plan: S03-03
reviewed: 2026-09-01
status: blocked
---

# S03-03 acceptance review — current evidence

## Decision

Do not close S03 and do not begin S03F-02. The current `.sinf` execution is
greedy-token correct and repeatable on the qualified RTX 5090, but it does not
meet the unchanged full-model numerical contract for the two longer corpus
cases. No production code change is justified by the evidence currently
available, and no tolerance was changed.

## Acceptance matrix

Source: `artifacts/S03/qwen38-s03-deployment-v8-acceptance.json`.

| Case | Rows | Greedy tokens | Fresh-session replay | Numerical contract |
|---|---:|---|---|---|
| plain-short | 6 | pass | pass | pass |
| unicode | 11 | pass | pass | pass |
| special-tokens | 3 | pass | pass | pass |
| chat-template | 60 | pass | pass | fail; max abs 0.5856843, rows 23/29/30 |
| varied-length | 81 | pass | pass | fail; max abs 4.5100713, rows 36–79 subset |

The contract remains `max_abs <= 0.5`, `mean_abs <= 0.1`, and
`rmse <= 0.15`. The artifact hash is
`e25022c8592875449968b9d0b1f56e6800971e0ba04d8a43eec980fe60dc65d5`; the
corpus hash is
`438fc893d13045df197c37d6e6aeed25eac5a2d6f698d032b7b7e6047a12cf03`.

## Evidence review

The following independent checks constrain the remaining failure:

- Five fresh 13-token chat-prefix processes, two standard 60-token sessions,
  arena poison patterns, and a no-alias activation plan produced stable
  captures. The current failure is not reproduced nondeterminism, an obvious
  uninitialized read, or activation-lifetime reuse.
- A 30-token chat replay with activation reuse disabled is byte-identical to
  the incumbent capture and has the same row-23/29 numerical outliers. This
  independently rejects lifetime aliasing as the source of the failure; see
  `artifacts/S03/qwen38-activation-reuse-localization-v12.json`.
- The deployment-matched Transformers oracle uses BF16 embedding, BF16 KV,
  current-row BF16 GDN convolution state, FP32 recurrent state, and BF16 final
  norm output. Changing these reference storage boundaries does not remove the
  long-replay drift.
- A staged 30-segment GDN differential passes, including projection,
  convolution, recurrent core, normalized/gated output, and final layer checks.
- A standalone layer-3 full-attention decode over 30 positions passes with the
  deployment-matched BF16 KV oracle.
- A target-input layer-42 replay matches QKV, convolution, recurrent core, and
  complete-layer output within its local contract. The full-model layer-42
  cliff is therefore upstream error amplification, not proof of a corrupt
  layer-42 state transition.
- The layer-42 GDN `in_proj_z` projection also matches an independent oracle
  on the exact target hidden input and recurrent state at max absolute error
  `0.01321268` / RMSE `0.00264875`; QKV, convolution, and recurrent core stay
  bounded. The standalone z path is not the source of the cliff; see
  `artifacts/S03/qwen38-gdn-z-projection-localization-v11.json`.
- An independent chunked NVFP4 LM-head replay on the exact target final-hidden
  row differs by at most `0.03125`, ruling out final projection arithmetic as
  the source of the full-model outlier; see
  `artifacts/S03/qwen38-lm-head-localization-v13.json`.
- Per-layer traces show gradual accumulated drift, with a layer-42
  amplification, rather than one isolated failing command.

Primary evidence:

- `artifacts/S03/qwen38-post-attention-state-localization-v8.json`
- `artifacts/S03/qwen38-layer42-input-amplification.json`
- `artifacts/S03/qwen38-layer3-long-context-differential.json`
- `artifacts/S03/qwen38-gdn-long-context-differential.json`
- `artifacts/S03/qwen38-long-replay-diagnostic-round2.json`
- `artifacts/S03/qwen38-gdn-z-projection-localization-v11.json`
- `artifacts/S03/qwen38-activation-reuse-localization-v12.json`
- `artifacts/S03/qwen38-lm-head-localization-v13.json`

## Required closure condition

S03 may close only after one of these is evidenced and reviewed:

1. a root-cause implementation fix makes both longer cases satisfy the
   unchanged numerical contract, with fresh two-session acceptance; or
2. a separately reviewed decision supersedes the numerical contract with an
   independently justified quantized-model contract, including its scope,
   quality implications, and updated acceptance evidence.

The current evidence supports neither condition. Greedy token agreement is
not being substituted for numerical agreement.

## Next boundary

Keep work within S03-03 diagnostics or acceptance evidence. Do not add kernels,
fusion, performance tuning, multi-GPU runtime, PLE runtime, MoE runtime, or
QSA runtime. S03F-01 research remains complete but quality-unqualified for the
local Flash-Next GGUF candidate; S03F-02 remains engineering-blocked by this
open S03 correctness condition.
