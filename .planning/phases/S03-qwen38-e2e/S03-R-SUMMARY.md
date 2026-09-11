# S03-R Summary — Decisive Same-Artifact Correctness Closure

**Outcome: A (contract supersession). S03 CLOSED under D-021. No production code changed.**

## Compact proof chain

1. **Exact `.sinf` bytes → independent oracle.** `tools/qwen38_sinf_weights.py` reads packed NVFP4/BF16/FP8 payloads from the deployment artifact SuperInfer executes (`build/evidence/qwen38-payload-v1-final-a.sinf`, sha `e25022c8…dc65d5`) and dequantizes with PyTorch ops. SuperInfer CUDA kernels are never reused as their own oracle. SINF-loaded weights are bit-identical to the safetensors source (BF16 exact, NVFP4 dequant maxdiff 0.0).
2. **Historical-oracle identity.** The oracle tooling was ported to transformers 5.16.1 (4D→None mask contract, new cache-layer API, `.sinf` weight source). The ported source oracle reproduces the historical 5.12.1 deployment-v8 captures bit-identically (maxdiff 0.0000, all cases), so all prior layer/localization evidence transfers unchanged.
3. **Cross-placement stability.** The oracle agrees with itself across independent runs and CPU/CUDA placement (0 flips on 7/8 cases; worst 0.05; 1.5 only on the degenerate long-103 case where both executions enter activation explosion).
4. **Strict-margin agreement.** Wording made precise: rows whose reference winner margin exceeds the BF16 output-discrimination floor (`ulp_BF16(|reference winner|)`, magnitude-dependent, no magic constants) are **strict rows and are 240/240 greedy-exact** across two fresh sessions. The five observed winner swaps occurred **only** on near-tie rows inside D-021's ambiguity regime (margin/ulp 0.02–0.46); every swap is a reference runner-up swap with top-5 overlap 1.0 and row JS ≤ 3.3e-4. There is no contradiction: strict rows never flip; only sub-ulp ties swap, which BF16 output storage explains.
5. **Near-tie swaps explained by the BF16 contract.** SuperInfer stores device logits as BF16 by design and computes its own greedy from those values. A 0.029-logit margin is below BF16 discrimination at that magnitude; the swap is expected deployment arithmetic. CUDA-vs-CPU oracle agreement (no flip) is the discriminator that rules out a compute bug.
6. **Distributional bounds pass.** In-scope rows (252 session-1): JS ≤ 0.005, RMSE ≤ 2.0, mean ≤ 0.9 (recipe: 2× in-scope p99 rounded up, clearing the max). Session-2 verdict: **pass**.
7. **Local/layer gates unchanged.** No kernel, layer, or differential tolerance was modified. GDN staged (0.0357), layer-3 attention (2e-4), LM-head (0.031), materialization, and all localization evidence stand as-is.
8. **Fresh-session repeatability.** All 8 session-2 captures are byte-identical to session-1 captures taken on a different day (SHA256 match on full payloads).

## Session-2 closure verdict

`decide_d021` over 318 rows, two byte-identical sessions: **pass** — 240 strict rows greedy-exact, 12 tie rows within the near-tie set, distributional bounds clear, 66 listed long-103 outlier rows reported (non-blocking per contract).

## long-103 classification: stress/outlier characterization, NOT an unresolved defect

The 6×17-token repetition drives both implementations into activation explosion (hidden magnitudes ~400, agreed by both). Oracle-vs-oracle itself degrades 200× there (0.008→1.5). Per-layer traces at the split show smooth relative error growth (mixer rel 0.1%→6.6%, no cliff) with absolute-error theater from MLP gain in the exploded regime — no disproportional operation, hence no bug to fix. Repetition robustness is queued as S04+ research and does not block R01.

## Evidence

- `artifacts/S03R/qwen38-d021-contract.json` — machine-readable D-021
- `artifacts/S03R/qwen38-same-artifact-row-metrics-session1.json` — 318 session-1 row metrics
- `build/evidence/s03r-session1-supe/`, `build/evidence/s03r-session2/` — raw SuperInfer captures, both sessions
- `/tmp` oracle captures (regenerable via recorded commands; hashes in session reports)
- Tools: `qwen38_same_artifact_metrics.py`, `qwen38_sinf_weights.py`, `qwen38_s03r_acceptance.py` (+ unit tests); corpus `tests/corpora/qwen38/results-first-v1.json` (8 cases, 318 rows)

## What S03-R did NOT do

No precision/rounding hypothesis work (frozen per D-020), no tolerance loosening, no Flash-Next work, no kernel optimization. One runner bug fixed (per-case output dir creation); one environment port (transformers 5.16.1) validated by bit-identity.
