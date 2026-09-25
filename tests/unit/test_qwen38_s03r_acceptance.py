import unittest

from tools.qwen38_s03r_acceptance import aggregate_rows, bf16_ulp, decide_d021, decide_s03r

CONTRACT = {
    "same_artifact_max_abs": 0.05,
    "same_artifact_mean_abs": 0.01,
    "same_artifact_rmse": 0.015,
    "same_artifact_js": 0.005,
    "require_greedy": True,
    "require_margin_safe": True,
    "require_repeatable": True,
}


def _row(max_abs=0.01, greedy=True, margin_ratio=0.2, js=0.0001):
    return {
        "max_abs": max_abs,
        "mean_abs": max_abs / 4.0,
        "rmse": max_abs / 3.0,
        "greedy_match": greedy,
        "max_error_over_margin": margin_ratio,
        "js_divergence": js,
        "top_k_overlap": 1.0 if greedy else 0.8,
    }


class S03RDecisionTests(unittest.TestCase):
    def test_clean_evidence_is_supersede_candidate(self) -> None:
        rows = [_row() for _ in range(8)]
        summary = aggregate_rows(rows, repeatable=True)
        decision = decide_s03r(summary, CONTRACT)
        self.assertEqual(decision["decision"], "contract_supersede_candidate")

    def test_greedy_flip_is_real_discrepancy(self) -> None:
        rows = [_row() for _ in range(7)] + [_row(greedy=False, margin_ratio=1.4, max_abs=2.0)]
        summary = aggregate_rows(rows, repeatable=True)
        decision = decide_s03r(summary, CONTRACT)
        self.assertEqual(decision["decision"], "real_superinfer_discrepancy")

    def test_margin_unsafe_row_is_real_discrepancy(self) -> None:
        rows = [_row() for _ in range(7)] + [_row(margin_ratio=1.0, max_abs=0.04)]
        summary = aggregate_rows(rows, repeatable=True)
        decision = decide_s03r(summary, CONTRACT)
        self.assertEqual(decision["decision"], "real_superinfer_discrepancy")

    def test_non_repeatable_evidence_is_inconclusive(self) -> None:
        rows = [_row() for _ in range(8)]
        summary = aggregate_rows(rows, repeatable=False)
        decision = decide_s03r(summary, CONTRACT)
        self.assertEqual(decision["decision"], "inconclusive")

    def test_inconclusive_is_never_a_pass(self) -> None:
        rows = [_row() for _ in range(8)]
        summary = aggregate_rows(rows, repeatable=False)
        decision = decide_s03r(summary, CONTRACT)
        self.assertNotIn(decision["decision"], ("pass", "contract_supersede_candidate"))

    def test_aggregation_reports_quantiles_and_worst_rows(self) -> None:
        rows = [_row(max_abs=0.01), _row(max_abs=0.03), _row(max_abs=0.02)]
        summary = aggregate_rows(rows, repeatable=True)
        self.assertEqual(summary["total_rows"], 3)
        self.assertAlmostEqual(summary["max_abs_max"], 0.03)
        self.assertGreaterEqual(summary["max_abs_p95"], summary["max_abs_p50"])
        self.assertEqual(summary["greedy_mismatch_rows"], 0)
        self.assertEqual(summary["margin_unsafe_rows"], 0)


D021_CONTRACT = {
    "tie_top_k": 5,
    "tie_min_overlap": 0.8,
    "tie_max_js": 0.001,
    "max_js": 0.005,
    "max_rmse": 2.0,
    "max_mean_abs": 0.9,
    "outlier_rows": [{"case": "long", "from_row": 37, "reason": "test"}],
}


def _d021_row(row=0, greedy=True, margin=2.0, winner_logit=10.0, cand=0,
              js=1e-5, rmse=0.01, mean=0.005, overlap=1.0):
    return {
        "row": row,
        "greedy_match": greedy,
        "reference_argmax": 0,
        "candidate_argmax": cand,
        "reference_argmax_margin": margin,
        "reference_winner_logit": winner_logit,
        "reference_top_k": [0, 1, 2, 3, 4],
        "top_k_overlap": overlap,
        "js_divergence": js,
        "rmse": rmse,
        "mean_abs": mean,
        "max_abs": margin / 2.0,
    }


class D021VerdictTests(unittest.TestCase):
    def test_bf16_ulp_follows_mantissa(self) -> None:
        self.assertAlmostEqual(bf16_ulp(10.0), 0.0625)
        self.assertAlmostEqual(bf16_ulp(20.0), 0.125)
        with self.assertRaises(ValueError):
            bf16_ulp(0.0)

    def test_strict_rows_must_match(self) -> None:
        verdict = decide_d021({"a": [_d021_row(), _d021_row(row=1)]},
                              D021_CONTRACT, repeatable=True)
        self.assertEqual(verdict["verdict"], "pass")
        self.assertEqual(verdict["strict_rows"], 2)

    def test_strict_flip_fails(self) -> None:
        verdict = decide_d021({"a": [_d021_row(greedy=False, cand=7)]},
                              D021_CONTRACT, repeatable=True)
        self.assertEqual(verdict["verdict"], "fail")

    def test_sub_ulp_tie_within_topk_passes(self) -> None:
        verdict = decide_d021({"a": [_d021_row(greedy=False, margin=0.01, cand=1)]},
                              D021_CONTRACT, repeatable=True)
        self.assertEqual(verdict["verdict"], "pass")
        self.assertEqual(verdict["tie_rows"], 1)

    def test_tie_winner_outside_topk_fails(self) -> None:
        verdict = decide_d021({"a": [_d021_row(greedy=False, margin=0.01, cand=9)]},
                              D021_CONTRACT, repeatable=True)
        self.assertEqual(verdict["verdict"], "fail")

    def test_listed_outliers_are_non_blocking_but_reported(self) -> None:
        bad = _d021_row(row=50, greedy=False, margin=5.0, cand=9,
                        js=0.2, rmse=5.0, mean=3.0, overlap=0.0)
        verdict = decide_d021({"long": [bad]}, D021_CONTRACT, repeatable=True)
        self.assertEqual(verdict["verdict"], "pass")
        self.assertEqual(len(verdict["outlier_reports"]), 1)

    def test_unlisted_distributional_breach_fails(self) -> None:
        verdict = decide_d021({"a": [_d021_row(js=0.05)]},
                              D021_CONTRACT, repeatable=True)
        self.assertEqual(verdict["verdict"], "fail")

    def test_non_repeatable_is_inconclusive(self) -> None:
        verdict = decide_d021({"a": [_d021_row()]}, D021_CONTRACT, repeatable=False)
        self.assertEqual(verdict["verdict"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
