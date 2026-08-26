import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from superinfer.artifact import (
    ArtifactError,
    MAXIMUM_ARTIFACT_BYTES,
    inspect_artifact,
    read_typed_tensor,
    write_artifact,
)


class PythonArtifactTests(unittest.TestCase):
    def test_artifact_limit_covers_pinned_qwen_payload_budget(self) -> None:
        self.assertGreaterEqual(MAXIMUM_ARTIFACT_BYTES, 32 * (1 << 30))

    def test_inspection_can_use_streaming_section_validation(self) -> None:
        source = {
            "manifest": {"revision": "r1"},
            "tensors": [],
            "physical_plan": "plan",
            "payload_hex": "01020304",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sinf"
            write_artifact(source, path)
            with patch("superinfer.artifact.STREAMING_INSPECTION_THRESHOLD_BYTES", 0):
                summary = inspect_artifact(path)
            self.assertEqual(summary["payload_bytes"], 4)

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

    def test_typed_tensor_materialization_preserves_physical_contract(self) -> None:
        source = {
            "manifest": {"revision": "r1"},
            "tensors": [
                {
                    "name": "hidden",
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "artifact_payload_offset": 0,
                    "artifact_payload_end": 8,
                },
                {
                    "name": "layer.weight",
                    "dtype": "U8",
                    "shape": [2, 8],
                    "artifact_payload_offset": 8,
                    "artifact_payload_end": 16,
                },
                {
                    "name": "layer.weight_scale",
                    "dtype": "F8_E4M3",
                    "shape": [2, 1],
                    "artifact_payload_offset": 16,
                    "artifact_payload_end": 18,
                },
            ],
            "physical_plan": "plan",
            "payload_hex": "0001020304050607" "1011121314151617" "1819",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed.sinf"
            write_artifact(source, path)
            hidden = read_typed_tensor(path, "hidden")
            self.assertEqual(hidden.descriptor.dtype, "bf16")
            self.assertEqual(hidden.descriptor.shape, (2, 2))
            self.assertEqual(hidden.descriptor.layout, "row_major")
            self.assertEqual(hidden.descriptor.encoding, "none")
            self.assertEqual(hidden.data, bytes.fromhex("0001020304050607"))

            weight = read_typed_tensor(path, "layer.weight")
            self.assertEqual(weight.descriptor.dtype, "u8")
            self.assertEqual(weight.descriptor.encoding, "nvfp4_packed")
            self.assertEqual(weight.descriptor.storage_bytes, 8)

            scale = read_typed_tensor(path, "layer.weight_scale")
            self.assertEqual(scale.descriptor.dtype, "u8")
            self.assertEqual(scale.descriptor.encoding, "fp8_e4m3_group_scale")

    def test_typed_tensor_materialization_rejects_shape_byte_mismatch(self) -> None:
        source = {
            "manifest": {},
            "tensors": [{
                "name": "bad",
                "dtype": "BF16",
                "shape": [2, 2],
                "artifact_payload_offset": 0,
                "artifact_payload_end": 2,
            }],
            "physical_plan": "plan",
            "payload_hex": "0001",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-typed.sinf"
            write_artifact(source, path)
            with self.assertRaisesRegex(ArtifactError, "payload bytes do not match"):
                read_typed_tensor(path, "bad")

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
