import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

from superinfer.cli import ExitCode, main
from superinfer.fixtures import make_identity_fixture
from superinfer.reference import compare_values
from superinfer.support.evidence import EvidenceWriter
from superinfer.support.seeds import SeedCapture
from superinfer.support.subprocesses import run_command
from superinfer.support.workspace import artifact_workspace


class SeedCaptureTests(unittest.TestCase):
    def test_restores_caller_rng_state_after_deterministic_scope(self) -> None:
        random.seed(99)
        before = random.getstate()
        with SeedCapture(123) as record:
            self.assertEqual(record.seed, 123)
            first = random.random()
        self.assertNotEqual(first, random.random())
        random.setstate(before)
        self.assertEqual(random.random(), 0.40397807494366633)


class WorkspaceAndEvidenceTests(unittest.TestCase):
    def test_workspace_and_evidence_are_reproducible_and_structured(self) -> None:
        with artifact_workspace() as workspace:
            writer = EvidenceWriter(workspace)
            path = writer.write_json("result.json", {"seed": 7, "status": "pass"})
            self.assertEqual(json.loads(path.read_text()), {"seed": 7, "status": "pass"})
            self.assertEqual(writer.append_jsonl("samples.jsonl", {"step": 1}), 1)
            self.assertTrue((workspace / "samples.jsonl").read_text().endswith("\n"))


class FixtureAndReferenceTests(unittest.TestCase):
    def test_identity_fixture_and_reference_comparison_report_error_location(self) -> None:
        fixture = make_identity_fixture(2)
        self.assertEqual(fixture.shape, (2, 2))
        comparison = compare_values(
            fixture.values, (1.0, 0.0, 0.0, 1.01), abs_tol=1e-4, rel_tol=1e-4
        )
        self.assertFalse(comparison.equal)
        self.assertEqual(comparison.max_error_index, 3)


class SubprocessAndCliTests(unittest.TestCase):
    def test_subprocess_failure_writes_machine_readable_context(self) -> None:
        with artifact_workspace() as workspace:
            result = run_command((sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"),
                                 failure_path=workspace / "failure.json", seed=17)
            self.assertEqual(result.returncode, 3)
            failure = json.loads((workspace / "failure.json").read_text())
            self.assertEqual(failure["seed"], 17)
            self.assertEqual(failure["returncode"], 3)

    def test_cli_has_stable_version_and_invalid_argument_exit_categories(self) -> None:
        self.assertEqual(main(["--version"]), ExitCode.OK)
        self.assertEqual(main(["validate", "missing.sinf"]), ExitCode.INPUT_ERROR)


class ImportBoundaryTests(unittest.TestCase):
    def test_core_import_does_not_load_gpu_or_model_frameworks(self) -> None:
        command = (sys.executable, "-c", "import sys, superinfer; print('torch' in sys.modules)")
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                env={"PYTHONPATH": str(Path("python").resolve())})
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
