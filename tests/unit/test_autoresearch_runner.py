"""Unit tests for tools/autoresearch_runner.py.

All commands are synthetic (``python -c``) and require no GPU, repo build, or
network. Tests exercise manifest validation, fail-closed stage ordering,
threshold-gated promotion, metric parsing, and dry-run safety.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.autoresearch_runner import (
    EXIT_ERROR,
    MetricSpec,
    ManifestError,
    load_manifest,
    parse_metrics,
    run_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tools" / "autoresearch_runner.py"


def _py(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def _base_manifest(base: Path) -> dict:
    incumbent = base / "incumbent"
    candidate = base / "candidate"
    incumbent.mkdir(exist_ok=True)
    candidate.mkdir(exist_ok=True)
    (incumbent / "wall_ms.txt").write_text("200\n")
    (candidate / "wall_ms.txt").write_text("100\n")
    bench_manifest = base / "benchmark-manifest.json"
    bench_manifest.write_text(json.dumps({"schema": "test.benchmark.v1"}))
    return {
        "id": "test-experiment",
        "incumbent_commit": "a" * 40,
        "candidate_commit": "b" * 40,
        "candidate_worktree": str(candidate),
        "benchmark_manifest": str(bench_manifest),
        "local_differential_command": _py("print('local_speedup=2.5')"),
        "correctness_command": None,
        "profile_command": None,
        "benchmark_command": _py(
            "import pathlib; print('wall_ms=' + pathlib.Path('wall_ms.txt').read_text().strip())"),
        "reproduction_command": _py("print('wall_ms=100')"),
        "metrics": {
            "local_speedup": {
                "regex": "local_speedup=([0-9.]+)",
                "direction": "higher_is_better",
                "stage": "local",
                "compare": "direct",
            },
            "wall_ms": {
                "regex": "wall_ms=([0-9.]+)",
                "direction": "lower_is_better",
                "stage": "benchmark",
                "compare": "context",
            },
        },
        "thresholds": {"min_local_speedup": 1.5, "min_e2e_gain": None},
        "evidence_dir": str(base / "evidence"),
    }


def _write_manifest(base: Path, payload: dict) -> Path:
    path = base / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


class MetricParsingTests(unittest.TestCase):
    def test_parses_declared_values_and_omits_absent_metrics(self) -> None:
        metrics = (
            MetricSpec("speedup", re.compile(r"speedup=([0-9.]+)"), "higher_is_better",
                       "local", "direct"),
            MetricSpec("latency", re.compile(r"latency_ms=([0-9.]+)"), "lower_is_better",
                       "benchmark", "context"),
        )
        parsed = parse_metrics("speedup=4.25\nnoise\n", metrics)
        self.assertEqual(parsed, {"speedup": 4.25})
        self.assertNotIn("latency", parsed)

    def test_first_match_wins_and_negative_scientific_values_parse(self) -> None:
        metrics = (
            MetricSpec("v", re.compile(r"v=(-?[0-9.eE+-]+)"), "higher_is_better",
                       "benchmark", "context"),
        )
        parsed = parse_metrics("v=1.5e-3\nv=9.0\n", metrics)
        self.assertAlmostEqual(parsed["v"], 1.5e-3)


class ManifestValidationTests(unittest.TestCase):
    def test_missing_required_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            payload = _base_manifest(base)
            del payload["local_differential_command"]
            path = _write_manifest(base, payload)
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_local_context_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            payload = _base_manifest(base)
            payload["metrics"]["local_speedup"]["compare"] = "context"
            path = _write_manifest(base, payload)
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_regex_needs_exactly_one_capture_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            payload = _base_manifest(base)
            payload["metrics"]["wall_ms"]["regex"] = r"wall_ms=[0-9.]+"
            path = _write_manifest(base, payload)
            with self.assertRaises(ManifestError):
                load_manifest(path)


class CliSafetyTests(unittest.TestCase):
    def test_invalid_manifest_exits_nonzero_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sentinel = base / "sentinel.txt"
            payload = _base_manifest(base)
            payload["local_differential_command"] = _py(
                f"import pathlib; pathlib.Path(r'{sentinel}').write_text('x')")
            del payload["benchmark_command"]
            path = _write_manifest(base, payload)
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--manifest", str(path)],
                cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, EXIT_ERROR)
            self.assertFalse(sentinel.exists())
            self.assertFalse((base / "evidence").exists())

    def test_dry_run_prints_plan_and_executes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sentinel = base / "sentinel.txt"
            payload = _base_manifest(base)
            payload["benchmark_command"] = _py(
                f"import pathlib; pathlib.Path(r'{sentinel}').write_text('x')")
            path = _write_manifest(base, payload)
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--manifest", str(path), "--dry-run"],
                cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0)
            self.assertFalse(sentinel.exists())
            self.assertFalse((base / "evidence").exists())
            plan = json.loads(completed.stdout)
            self.assertEqual(
                [(item["stage"], item["context"]) for item in plan],
                [("local", "candidate"), ("benchmark", "incumbent"),
                 ("benchmark", "candidate"), ("reproduction", "candidate")])


class VerdictTests(unittest.TestCase):
    def _run(self, payload_transform=None):
        temporary = tempfile.TemporaryDirectory()
        base = Path(temporary.name)
        payload = _base_manifest(base)
        if payload_transform is not None:
            payload_transform(payload)
        path = _write_manifest(base, payload)
        manifest = load_manifest(path)
        result = run_experiment(
            manifest,
            incumbent_dir=base / "incumbent",
            candidate_dir=base / "candidate",
            evidence_dir=base / "evidence",
        )
        return base, result, temporary

    def test_local_speedup_above_threshold_promotes(self) -> None:
        base, result, temporary = self._run()
        with temporary:
            self.assertEqual(result["verdict"], "promote")
            self.assertTrue((base / "evidence" / "result.json").is_file())
            report = (base / "evidence" / "REPORT.md").read_text()
            self.assertIn("PROMOTE", report)
            local = next(item for item in result["metrics"]
                         if item["name"] == "local_speedup")
            self.assertAlmostEqual(local["gain"], 2.5)
            self.assertTrue(local["meets_threshold"])

    def test_local_speedup_below_threshold_rejects(self) -> None:
        def transform(payload: dict) -> None:
            payload["local_differential_command"] = _py("print('local_speedup=1.1')")
        _base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "reject")

    def test_failed_local_differential_rejects_and_stops(self) -> None:
        def transform(payload: dict) -> None:
            base = Path(payload["evidence_dir"]).parent
            sentinel = base / "benchmark-ran.txt"
            payload["local_differential_command"] = _py(
                "import sys; print('local_speedup=2.5'); sys.exit(3)")
            payload["benchmark_command"] = _py(
                "import pathlib; "
                f"pathlib.Path(r'{sentinel}').write_text('x'); print('wall_ms=100')")
        base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "reject")
            self.assertFalse((base / "benchmark-ran.txt").exists())
            self.assertFalse((base / "evidence" / "benchmark-candidate.stdout.txt").exists())
            statuses = {(record["stage"], record["context"]): record["status"]
                        for record in result["stages"]}
            self.assertEqual(statuses[("local", "candidate")], "fail")
            self.assertEqual(statuses[("benchmark", "candidate")], "not_run")
            self.assertEqual(statuses[("reproduction", "candidate")], "not_run")

    def test_no_thresholds_is_inconclusive_never_promote(self) -> None:
        def transform(payload: dict) -> None:
            payload["thresholds"] = {"min_local_speedup": None, "min_e2e_gain": None}
        _base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertNotEqual(result["verdict"], "promote")

    def test_declared_threshold_without_evidence_rejects(self) -> None:
        def transform(payload: dict) -> None:
            payload["local_differential_command"] = _py("print('nothing here')")
        _base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "reject")
            self.assertIn("no evidence was parsed", " ".join(result["reasons"]))

    def test_e2e_gain_gate_promotes_on_context_before_after(self) -> None:
        def transform(payload: dict) -> None:
            payload["thresholds"] = {"min_local_speedup": None, "min_e2e_gain": 1.5}
            payload["local_differential_command"] = _py("print('ok')")
        _base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "promote")
            wall = next(item for item in result["metrics"] if item["name"] == "wall_ms")
            self.assertAlmostEqual(wall["before"], 200.0)
            self.assertAlmostEqual(wall["after"], 100.0)
            self.assertAlmostEqual(wall["gain"], 2.0)

    def test_e2e_gain_below_threshold_rejects(self) -> None:
        def transform(payload: dict) -> None:
            payload["thresholds"] = {"min_local_speedup": None, "min_e2e_gain": 3.0}
            payload["local_differential_command"] = _py("print('ok')")
        _base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "reject")

    def test_reversed_reproduction_rejects(self) -> None:
        def transform(payload: dict) -> None:
            payload["thresholds"] = {"min_local_speedup": None, "min_e2e_gain": 1.5}
            payload["local_differential_command"] = _py("print('ok')")
            payload["reproduction_command"] = _py("print('wall_ms=250')")
        _base, result, temporary = self._run(transform)
        with temporary:
            self.assertEqual(result["verdict"], "reject")
            self.assertFalse(result["reproduction"]["consistent"])
            self.assertIn("reversed", " ".join(result["reasons"]))


if __name__ == "__main__":
    unittest.main()
