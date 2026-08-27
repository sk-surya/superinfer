import unittest

from tools.qwen38_corpus_reference import canonical_token_hash, select_cases


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


if __name__ == "__main__":
    unittest.main()
