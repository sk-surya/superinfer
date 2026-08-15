import json
import unittest
from pathlib import Path

from superinfer.fixtures import make_identity_fixture


class FixtureGoldenTests(unittest.TestCase):
    def test_identity_fixture_matches_canonical_json(self) -> None:
        expected = json.loads(Path(__file__).with_name("identity-2.json").read_text())
        actual = make_identity_fixture(2)
        self.assertEqual(actual.shape, tuple(expected["shape"]))
        self.assertEqual(actual.values, tuple(expected["values"]))


if __name__ == "__main__":
    unittest.main()

