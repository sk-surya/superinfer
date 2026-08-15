import tempfile
import unittest
from pathlib import Path

from superinfer.artifact import ArtifactError, inspect_artifact, write_artifact


class ArtifactFuzzSmokeTests(unittest.TestCase):
    def test_bounded_single_byte_mutations_never_escape_as_unexpected_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sinf"
            write_artifact(
                {"manifest": {"fixture": True}, "tensors": [], "physical_plan": "plan",
                 "payload_hex": "0011223344556677"},
                path,
            )
            original = bytearray(path.read_bytes())
            for index in range(0, len(original), max(1, len(original) // 32)):
                mutated = bytearray(original)
                mutated[index] ^= 0xFF
                path.write_bytes(mutated)
                try:
                    inspect_artifact(path)
                except ArtifactError:
                    pass


if __name__ == "__main__":
    unittest.main()
