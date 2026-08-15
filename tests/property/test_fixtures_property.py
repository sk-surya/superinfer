import unittest

from superinfer.fixtures import make_identity_fixture


class FixturePropertyTests(unittest.TestCase):
    def test_identity_fixture_has_one_on_the_diagonal_for_small_shapes(self) -> None:
        for size in range(1, 9):
            fixture = make_identity_fixture(size)
            for row in range(size):
                for column in range(size):
                    self.assertEqual(fixture.values[row * size + column], float(row == column))


if __name__ == "__main__":
    unittest.main()

