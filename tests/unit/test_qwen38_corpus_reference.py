import unittest

from tools.qwen38_corpus_reference import (
    build_argument_parser,
    canonical_token_hash,
    select_cases,
)


class Qwen38CorpusReferenceTests(unittest.TestCase):
    def test_canonical_token_hash_matches_corpus_encoding(self) -> None:
        self.assertEqual(
            canonical_token_hash([9419, 11, 7070, 623, 776, 13]),
            "0d807749a8af1082ce33dde2057ab580a72be144626e613d1b14ee3e2566dfdb",
        )

    def test_case_selection_is_deterministic_and_explicit(self) -> None:
        corpus = {
            "cases": [
                {"id": "a", "token_ids": [1]},
                {"id": "b", "token_ids": [2]},
            ]
        }
        self.assertEqual([case["id"] for case in select_cases(corpus)], ["a", "b"])
        self.assertEqual([case["id"] for case in select_cases(corpus, ["b"])], ["b"])
        with self.assertRaises(ValueError):
            select_cases(corpus, ["missing"])

    def test_reference_cli_exposes_explicit_bf16_kv_diagnostic(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--round-kv",
            ]
        )
        self.assertTrue(args.round_kv)

    def test_reference_cli_exposes_explicit_linear_state_diagnostic(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--round-linear-state",
            ]
        )
        self.assertTrue(args.round_linear_state)

    def test_reference_cli_can_capture_one_normalized_hidden_row(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--hidden-output", "/hidden.f32",
                "--hidden-case", "chat-template",
                "--hidden-step", "29",
            ]
        )
        self.assertEqual(args.hidden_case, "chat-template")
        self.assertEqual(args.hidden_step, 29)

    def test_reference_cli_can_capture_post_layer_boundaries(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--boundaries-output", "/boundaries.f32",
                "--boundary-case", "chat-template",
                "--boundary-step", "29",
            ]
        )
        self.assertEqual(args.boundary_case, "chat-template")
        self.assertEqual(args.boundary_step, 29)

    def test_reference_cli_can_capture_post_token_mixer_boundaries(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--attention-boundaries-output", "/boundaries.f32",
                "--attention-boundary-case", "chat-template",
                "--attention-boundary-step", "29",
            ]
        )
        self.assertEqual(args.attention_boundary_case, "chat-template")
        self.assertEqual(args.attention_boundary_step, 29)

    def test_reference_cli_accepts_explicit_device(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--device", "cuda",
            ]
        )
        self.assertEqual(args.device, "cuda")

    def test_reference_cli_can_capture_kv_snapshot(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--model-dir", "/model",
                "--corpus", "/corpus.json",
                "--output-dir", "/output",
                "--kv-output", "/kv.f32",
                "--kv-case", "chat-template",
                "--kv-step", "28",
                "--kv-layer", "3",
            ]
        )
        self.assertEqual(args.kv_case, "chat-template")
        self.assertEqual(args.kv_step, 28)
        self.assertEqual(args.kv_layer, 3)


if __name__ == "__main__":
    unittest.main()
