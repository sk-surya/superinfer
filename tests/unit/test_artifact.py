import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from superinfer.artifact import ArtifactError, MAXIMUM_ARTIFACT_BYTES, inspect_artifact, write_artifact


class PythonArtifactTests(unittest.TestCase):
    def test_artifact_limit_covers_pinned_qwen_payload_budget(self) -> None:
        self.assertGreaterEqual(MAXIMUM_ARTIFACT_BYTES, 32 * (1 << 30))

    def test_converter_bytes_and_inspection_are_deterministic(self) -> None:
        source = {
            "manifest": {"revision": "r1", "model": "fixture"},
            "tensors": [{"bytes": 4, "name": "weight"}],
            "physical_plan": "physical-plan:v1 capability=120 catalog=fixture",
            "payload_hex": "01020304",
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.sinf"
            second = Path(directory) / "second.sinf"
            write_artifact(source, first)
            write_artifact(source, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            summary = inspect_artifact(first)
            self.assertEqual(
                summary["sections"],
                ["manifest", "tensor_table", "physical_plan", "payload", "integrity"],
            )

    def test_corruption_fails_before_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.sinf"
            write_artifact(
                {"manifest": {}, "tensors": [], "physical_plan": "plan", "payload_hex": ""}, path
            )
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaises(ArtifactError):
                inspect_artifact(path)

    def test_cli_inspect_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sinf"
            write_artifact(
                {"manifest": {}, "tensors": [], "physical_plan": "plan", "payload_hex": ""}, path
            )
            result = subprocess.run(
                (sys.executable, "-m", "superinfer", "inspect", str(path), "--json"),
                check=True,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": str(Path("python").resolve())},
            )
            self.assertEqual(json.loads(result.stdout)["format_version"], 1)

    def test_cli_convert_accepts_normalized_frontend_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "fixture.sinf"
            source.write_text(
                json.dumps(
                    {"manifest": {}, "tensors": [], "physical_plan": "plan", "payload_hex": ""}
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                (sys.executable, "-m", "superinfer", "convert", str(source), str(output)),
                check=True,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": str(Path("python").resolve())},
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(inspect_artifact(output)["format_version"], 1)


if __name__ == "__main__":
    unittest.main()
