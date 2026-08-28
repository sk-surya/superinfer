import hashlib
import struct
import unittest

from tools.qwen38_s03_batched_acceptance import (
    NUMERICAL_TOLERANCES,
    compare_row_against_contract,
    validate_reference_capture,
)
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

    def test_numerical_contract_rejects_large_drift_even_when_tokens_match(self) -> None:
        result = compare_row_against_contract([0.0, 1.0], [0.0, 1.0 + NUMERICAL_TOLERANCES["max_abs"] + 1.0])
        self.assertFalse(result["passed"])
        self.assertIn("max_abs", result["failed_metrics"])

    def test_reference_validation_recomputes_hash_and_greedy_sequence(self) -> None:
        values = [0.0, 2.0, 1.0, 3.0]
        payload = struct.pack("<4f", *values)
        diagnostics = {
            "tokens": [7, 8],
            "token_ids_sha256": hashlib.sha256(b"[7,8]").hexdigest(),
            "steps": 2,
            "logits_per_step": 2,
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "greedy_sequence": [1, 1],
        }
        validate_reference_capture([7, 8], diagnostics, values)
        with self.assertRaisesRegex(ValueError, "greedy"):
            validate_reference_capture([7, 8], {**diagnostics, "greedy_sequence": [0, 1]}, values)


if __name__ == "__main__":
    unittest.main()
