import unittest

from tools.qwen38_s03_acceptance import compare_logits, parse_superinfer_output


class Qwen38AcceptanceHarnessTests(unittest.TestCase):
    def test_parses_all_forced_prompt_steps(self) -> None:
        output = (
            "qwen38 e2e token=9419 greedy=12 logit=1.5 checksum=0 "
            "state_buffers=128 commands=4848 arena_bytes=1 "
            "continuation_steps=2 token=1 greedy=13 logit=1.25 "
            "token=2 greedy=14 logit=1.0\n"
        )
        self.assertEqual(
            parse_superinfer_output(output),
            {"greedy": [12, 13, 14], "commands": 4848, "state_buffers": 128},
        )

    def test_logit_comparison_reports_shape_and_error(self) -> None:
        result = compare_logits([1.0, 2.0], [1.0, 2.25])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["max_abs"], 0.25)
        self.assertGreater(result["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
