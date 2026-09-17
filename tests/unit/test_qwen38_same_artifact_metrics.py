import math
import unittest

from tools.qwen38_same_artifact_metrics import compare_distribution


class SameArtifactMetricsTests(unittest.TestCase):
    def test_identical_logits_agree_exactly(self) -> None:
        reference = [3.0, 1.0, 0.0, -1.0]
        metrics = compare_distribution(reference, list(reference), top_k=2)
        self.assertTrue(metrics["greedy_match"] is True)
        self.assertEqual(metrics["top_k_overlap"], 1.0)
        self.assertEqual(metrics["js_divergence"], 0.0)
        self.assertGreater(metrics["candidate_argmax_margin"], 0.0)
        self.assertGreaterEqual(metrics["max_error_over_margin"], 0.0)
        self.assertEqual(metrics["max_abs"], 0.0)

    def test_common_shift_is_probabilistically_harmless(self) -> None:
        reference = [3.0, 1.0, 0.0, -1.0]
        candidate = [value + 5.0 for value in reference]
        metrics = compare_distribution(reference, candidate, top_k=2)
        self.assertTrue(metrics["greedy_match"] is True)
        self.assertEqual(metrics["top_k_overlap"], 1.0)
        self.assertAlmostEqual(metrics["js_divergence"], 0.0, places=12)
        self.assertGreater(metrics["candidate_argmax_margin"], 0.0)

    def test_changed_tail_logit_keeps_winner(self) -> None:
        reference = [3.0, 1.0, 0.0, -1.0]
        candidate = [3.0, 1.0, 0.0, -0.5]
        metrics = compare_distribution(reference, candidate, top_k=2)
        self.assertTrue(metrics["greedy_match"] is True)
        self.assertEqual(metrics["top_k_overlap"], 1.0)
        self.assertGreater(metrics["js_divergence"], 0.0)
        self.assertLess(metrics["max_error_over_margin"], 1.0)

    def test_argmax_flip_is_unsafe(self) -> None:
        reference = [3.0, 1.0, 0.0, -1.0]
        candidate = [1.0, 3.0, 0.0, -1.0]
        metrics = compare_distribution(reference, candidate, top_k=2)
        self.assertTrue(metrics["greedy_match"] is False)
        self.assertGreaterEqual(metrics["max_error_over_margin"], 1.0)
        self.assertGreater(metrics["js_divergence"], 0.0)

    def test_tie_set_fields_support_margin_qualified_contracts(self) -> None:
        reference = [3.0, 1.0, 0.0, -1.0]
        metrics = compare_distribution(reference, reference, top_k=2)
        self.assertEqual(metrics["reference_top_k"], [0, 1])
        self.assertEqual(metrics["reference_winner_logit"], 3.0)
        self.assertEqual(metrics["reference_argmax_margin"], 2.0)


if __name__ == "__main__":
    unittest.main()
