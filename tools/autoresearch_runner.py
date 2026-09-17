#!/usr/bin/env python3
"""Minimum autoresearch experiment runner.

Automates the *mechanics* of the proven manual optimization loop:

    profile -> isolated candidate -> local differential -> model correctness
    -> fixed benchmark -> fresh-session reproduction -> promote/reject

The runner deliberately owns no technical judgment. It executes a declarative
experiment manifest, captures evidence, parses declared numeric metrics with
declared regexes, and applies declared thresholds. It never invents, widens, or
loosens a tolerance: promotion requires an explicit threshold and parsed
evidence, and any failed stage, missing evidence, or reversed fresh-session
reproduction fails closed.

The runner is read-only with respect to the manifest and the repository: it
never runs ``git reset``, never force-pushes, never edits tracked files, and
never deletes user files. Its only writes are inside the manifest's evidence
directory.

Exit codes::

    0  promote (or ``--dry-run`` completed)
    1  usage error, incomplete/unreadable manifest, or execution setup failure
    2  reject
    3  inconclusive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

RESULT_SCHEMA = "superinfer.autoresearch.result.v1"
MANIFEST_SCHEMA = "superinfer.autoresearch.experiment.v1"
MAX_CAPTURE_CHARS = 8000
DIRECTIONS = ("higher_is_better", "lower_is_better")
COMPARE_MODES = ("direct", "context")
STAGES = ("local", "benchmark", "reproduction")

EXIT_PROMOTE = 0
EXIT_ERROR = 1
EXIT_REJECT = 2
EXIT_INCONCLUSIVE = 3


class AutoresearchError(Exception):
    """Base error for runner failures."""


class ManifestError(AutoresearchError):
    """The manifest is absent, unreadable, or structurally invalid."""


@dataclass(frozen=True)
class MetricSpec:
    """A declared numeric metric parsed from a stage's stdout.

    ``compare`` is ``"direct"`` when the captured value already is the gated
    gain (for example a local microbenchmark speedup) and ``"context"`` when
    the value is a raw timing that must be compared between the incumbent and
    candidate contexts.
    """

    name: str
    regex: re.Pattern[str]
    direction: str
    stage: str
    compare: str


@dataclass(frozen=True)
class ExperimentManifest:
    """Validated, immutable view of a declarative experiment manifest."""

    id: str
    incumbent_commit: str
    candidate_commit: str
    candidate_worktree: str | None
    benchmark_manifest: str
    local_differential_command: tuple[str, ...]
    correctness_command: tuple[str, ...] | None
    profile_command: tuple[str, ...] | None
    benchmark_command: tuple[str, ...]
    reproduction_command: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]
    min_local_speedup: float | None
    min_e2e_gain: float | None
    evidence_dir: str
    source_path: Path
    sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _sha256_bytes(path.read_bytes())


def _resolve(path_str: str, base: Path) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _require_command(value: Any, field: str, *, allow_none: bool) -> tuple[str, ...] | None:
    if value is None:
        if allow_none:
            return None
        raise ManifestError(f"{field} is required and must be a non-empty command array")
    if (not isinstance(value, list) or not value
            or not all(isinstance(item, str) and item for item in value)):
        raise ManifestError(f"{field} must be a non-empty array of strings")
    return tuple(value)


def _require_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} must be a number or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ManifestError(f"{field} must be finite")
    if numeric < 0.0:
        raise ManifestError(f"{field} must be non-negative")
    return numeric


def _parse_metrics(raw: Any) -> tuple[MetricSpec, ...]:
    if not isinstance(raw, dict) or not raw:
        raise ManifestError("metrics must be a non-empty object")
    specs: list[MetricSpec] = []
    for name, entry in raw.items():
        if not isinstance(name, str) or not name:
            raise ManifestError("metric names must be non-empty strings")
        if not isinstance(entry, dict):
            raise ManifestError(f"metrics.{name} must be an object")
        regex_source = entry.get("regex")
        if not isinstance(regex_source, str) or not regex_source:
            raise ManifestError(f"metrics.{name}.regex must be a non-empty string")
        direction = entry.get("direction")
        if direction not in DIRECTIONS:
            raise ManifestError(
                f"metrics.{name}.direction must be one of {DIRECTIONS}")
        stage = entry.get("stage", "benchmark")
        if stage not in STAGES:
            raise ManifestError(f"metrics.{name}.stage must be one of {STAGES}")
        default_compare = "direct" if stage == "local" else "context"
        compare = entry.get("compare", default_compare)
        if compare not in COMPARE_MODES:
            raise ManifestError(
                f"metrics.{name}.compare must be one of {COMPARE_MODES}")
        if stage == "local" and compare == "context":
            raise ManifestError(
                f"metrics.{name}: the local differential runs once, so compare "
                "must be 'direct'")
        try:
            pattern = re.compile(regex_source)
        except re.error as error:
            raise ManifestError(f"metrics.{name}.regex does not compile: {error}") from error
        if pattern.groups != 1:
            raise ManifestError(
                f"metrics.{name}.regex must contain exactly one capture group "
                f"(found {pattern.groups})")
        specs.append(MetricSpec(name=name, regex=pattern, direction=direction,
                                stage=stage, compare=compare))
    return tuple(specs)


def _validate(raw: Any, source_path: Path, digest: str) -> ExperimentManifest:
    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be a JSON object")
    for field in ("id", "incumbent_commit", "candidate_commit", "benchmark_manifest",
                  "local_differential_command", "correctness_command", "profile_command",
                  "benchmark_command", "reproduction_command", "metrics", "thresholds",
                  "evidence_dir"):
        if field not in raw:
            raise ManifestError(f"manifest is missing required field: {field}")
    thresholds = raw.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ManifestError("thresholds must be an object")
    if "min_local_speedup" not in thresholds or "min_e2e_gain" not in thresholds:
        raise ManifestError(
            "thresholds must declare both min_local_speedup and min_e2e_gain "
            "(values may be null)")
    candidate_worktree = raw.get("candidate_worktree")
    if candidate_worktree is not None:
        candidate_worktree = _require_string(candidate_worktree, "candidate_worktree")
    return ExperimentManifest(
        id=_require_string(raw.get("id"), "id"),
        incumbent_commit=_require_string(raw.get("incumbent_commit"), "incumbent_commit"),
        candidate_commit=_require_string(raw.get("candidate_commit"), "candidate_commit"),
        candidate_worktree=candidate_worktree,
        benchmark_manifest=_require_string(raw.get("benchmark_manifest"), "benchmark_manifest"),
        local_differential_command=_require_command(
            raw.get("local_differential_command"), "local_differential_command",
            allow_none=False) or (),
        correctness_command=_require_command(
            raw.get("correctness_command"), "correctness_command", allow_none=True),
        profile_command=_require_command(
            raw.get("profile_command"), "profile_command", allow_none=True),
        benchmark_command=_require_command(
            raw.get("benchmark_command"), "benchmark_command", allow_none=False) or (),
        reproduction_command=_require_command(
            raw.get("reproduction_command"), "reproduction_command", allow_none=False) or (),
        metrics=_parse_metrics(raw.get("metrics")),
        min_local_speedup=_require_number(thresholds.get("min_local_speedup"),
                                          "thresholds.min_local_speedup"),
        min_e2e_gain=_require_number(thresholds.get("min_e2e_gain"),
                                     "thresholds.min_e2e_gain"),
        evidence_dir=_require_string(raw.get("evidence_dir"), "evidence_dir"),
        source_path=source_path,
        sha256=digest,
    )


def load_manifest(path: Path) -> ExperimentManifest:
    """Read, hash, and validate a manifest. Raises :class:`ManifestError`."""
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"manifest is not valid UTF-8 JSON: {error}") from error
    return _validate(raw, path, digest)


def parse_metrics(stdout: str, metrics: Sequence[MetricSpec]) -> dict[str, float]:
    """Parse declared metric values from ``stdout`` by regex search.

    A declared metric that does not appear in ``stdout`` is absent from the
    returned mapping; callers treat absence as missing evidence, never as zero.
    """
    parsed: dict[str, float] = {}
    for spec in metrics:
        match = spec.regex.search(stdout)
        if match is None:
            continue
        try:
            parsed[spec.name] = float(match.group(1))
        except (TypeError, ValueError):
            continue
    return parsed


def _specs_for_stage(stage: str, metrics: Sequence[MetricSpec]) -> tuple[MetricSpec, ...]:
    if stage == "local":
        return tuple(spec for spec in metrics if spec.stage == "local")
    if stage == "benchmark":
        return tuple(spec for spec in metrics if spec.stage == "benchmark")
    if stage == "reproduction":
        # The reproduction command is the benchmark run again, so benchmark
        # metrics are also parsed there.
        return tuple(spec for spec in metrics if spec.stage in ("benchmark", "reproduction"))
    return ()


def _plan_stages(manifest: ExperimentManifest, incumbent: Path,
                 candidate: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    if manifest.profile_command is not None:
        plan.append({"stage": "profile", "context": "candidate",
                     "cwd": str(candidate), "command": list(manifest.profile_command)})
    plan.append({"stage": "local", "context": "candidate",
                 "cwd": str(candidate), "command": list(manifest.local_differential_command)})
    if manifest.correctness_command is not None:
        plan.append({"stage": "correctness", "context": "candidate",
                     "cwd": str(candidate), "command": list(manifest.correctness_command)})
    plan.append({"stage": "benchmark", "context": "incumbent",
                 "cwd": str(incumbent), "command": list(manifest.benchmark_command)})
    plan.append({"stage": "benchmark", "context": "candidate",
                 "cwd": str(candidate), "command": list(manifest.benchmark_command)})
    plan.append({"stage": "reproduction", "context": "candidate",
                 "cwd": str(candidate), "command": list(manifest.reproduction_command)})
    return plan


def _truncate(text: str) -> str:
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    head = MAX_CAPTURE_CHARS // 2
    tail = MAX_CAPTURE_CHARS - head
    return (f"{text[:head]}\n...[truncated {len(text) - MAX_CAPTURE_CHARS} chars]...\n"
            f"{text[-tail:]}")


def _record(stage: str, context: str, item: dict[str, Any], *,
            status: str, returncode: int | None, duration_s: float | None,
            stdout: str = "", stderr: str = "",
            stdout_path: str | None = None, stderr_path: str | None = None,
            parsed: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "context": context,
        "cwd": item["cwd"],
        "command": list(item["command"]),
        "status": status,
        "returncode": returncode,
        "duration_s": duration_s,
        "stdout": _truncate(stdout),
        "stderr": _truncate(stderr),
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "metrics": parsed or {},
    }


def _execute(item: dict[str, Any], evidence_dir: Path,
             specs: Sequence[MetricSpec]) -> dict[str, Any]:
    stage = str(item["stage"])
    context = str(item["context"])
    started = time.monotonic()
    try:
        process = subprocess.run(list(item["command"]), cwd=item["cwd"],
                                 capture_output=True, text=True, check=False)
    except OSError as error:
        raise AutoresearchError(
            f"failed to execute {stage}[{context}]: {error}") from error
    duration_s = time.monotonic() - started
    stdout_path = evidence_dir / f"{stage}-{context}.stdout.txt"
    stderr_path = evidence_dir / f"{stage}-{context}.stderr.txt"
    stdout_path.write_text(process.stdout)
    stderr_path.write_text(process.stderr)
    status = "pass" if process.returncode == 0 else "fail"
    return _record(stage, context, item, status=status, returncode=process.returncode,
                   duration_s=duration_s, stdout=process.stdout, stderr=process.stderr,
                   stdout_path=str(stdout_path), stderr_path=str(stderr_path),
                   parsed=parse_metrics(process.stdout, specs))


def _gain(before: float | None, after: float | None, direction: str) -> float | None:
    if before is None or after is None or before == 0.0:
        return None
    if direction == "higher_is_better":
        return after / before
    return before / after


def _metric_payload(spec: MetricSpec, *, before: float | None, after: float | None,
                    gain: float | None, threshold: float | None) -> dict[str, Any]:
    return {
        "name": spec.name,
        "stage": spec.stage,
        "direction": spec.direction,
        "compare": spec.compare,
        "regex": spec.regex.pattern,
        "before": before,
        "after": after,
        "gain": gain,
        "threshold": threshold,
        "meets_threshold": (gain is not None and threshold is not None
                            and gain >= threshold),
    }


def _collect_metrics(manifest: ExperimentManifest,
                     records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(record["stage"], record["context"]): record for record in records}
    local = by_key.get(("local", "candidate"), {})
    bench_before = by_key.get(("benchmark", "incumbent"), {})
    bench_after = by_key.get(("benchmark", "candidate"), {})
    evidence: list[dict[str, Any]] = []
    for spec in manifest.metrics:
        if spec.stage == "local":
            after = local.get("metrics", {}).get(spec.name)
            before = None
            gain = after
            threshold = manifest.min_local_speedup
        elif spec.compare == "direct":
            after = bench_after.get("metrics", {}).get(spec.name)
            before = None
            gain = after
            threshold = manifest.min_e2e_gain
        else:
            before = bench_before.get("metrics", {}).get(spec.name)
            after = bench_after.get("metrics", {}).get(spec.name)
            gain = _gain(before, after, spec.direction)
            threshold = manifest.min_e2e_gain
        evidence.append(_metric_payload(spec, before=before, after=after,
                                        gain=gain, threshold=threshold))
    return evidence


def _reproduction_check(manifest: ExperimentManifest,
                        records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Confirm the benchmark direction survives a fresh session."""
    by_key = {(record["stage"], record["context"]): record for record in records}
    bench_before = by_key.get(("benchmark", "incumbent"), {})
    bench_after = by_key.get(("benchmark", "candidate"), {})
    reproduction = by_key.get(("reproduction", "candidate"), {})
    checks: list[dict[str, Any]] = []
    for spec in manifest.metrics:
        if spec.compare != "context":
            continue
        before = bench_before.get("metrics", {}).get(spec.name)
        after = bench_after.get("metrics", {}).get(spec.name)
        repeated = reproduction.get("metrics", {}).get(spec.name)
        candidate_gain = _gain(before, after, spec.direction)
        repeated_gain = _gain(before, repeated, spec.direction)
        if candidate_gain is None or repeated_gain is None:
            continue
        consistent = (candidate_gain >= 1.0) == (repeated_gain >= 1.0)
        checks.append({
            "name": spec.name,
            "before": before,
            "after": after,
            "reproduced_after": repeated,
            "candidate_gain": candidate_gain,
            "reproduced_gain": repeated_gain,
            "consistent": consistent,
        })
    return {"checks": checks, "consistent": all(c["consistent"] for c in checks)}


def _decide(manifest: ExperimentManifest, evidence: Sequence[dict[str, Any]],
            reproduction: dict[str, Any], *, all_stages_passed: bool,
            failure_reason: str | None) -> tuple[str, list[str]]:
    if not all_stages_passed:
        return "reject", [failure_reason or "a required stage failed"]
    thresholds_declared = (manifest.min_local_speedup is not None
                           or manifest.min_e2e_gain is not None)
    if not thresholds_declared:
        return "inconclusive", [
            "no thresholds declared; the runner never invents a tolerance, so no "
            "promotion is possible"
        ]
    reasons: list[str] = []
    failed = False
    for item in evidence:
        threshold = item["threshold"]
        if threshold is None:
            continue
        if item["gain"] is None:
            reasons.append(f"{item['name']}: threshold {threshold} declared but no "
                           "evidence was parsed")
            failed = True
            continue
        if item["gain"] < threshold:
            reasons.append(f"{item['name']}: gain {item['gain']:.6g} below threshold "
                           f"{threshold:.6g}")
            failed = True
        else:
            reasons.append(f"{item['name']}: gain {item['gain']:.6g} meets threshold "
                           f"{threshold:.6g}")
    if not failed and reproduction["consistent"] is False:
        return "reject", reasons + [
            "fresh-session reproduction reversed the benchmark direction"
        ]
    if failed:
        return "reject", reasons
    return "promote", reasons or ["all declared thresholds satisfied"]


def run_experiment(manifest: ExperimentManifest, *, incumbent_dir: Path | str | None = None,
                   candidate_dir: Path | str | None = None,
                   evidence_dir: Path | str | None = None,
                   dry_run: bool = False) -> dict[str, Any]:
    """Execute (or plan) an experiment and return the machine-readable result."""
    incumbent = Path(incumbent_dir).resolve() if incumbent_dir else Path.cwd().resolve()
    if candidate_dir:
        candidate = Path(candidate_dir).resolve()
    elif manifest.candidate_worktree:
        candidate = _resolve(manifest.candidate_worktree, manifest.source_path.parent)
    else:
        candidate = incumbent
    evidence = (Path(evidence_dir).resolve() if evidence_dir
                else _resolve(manifest.evidence_dir, manifest.source_path.parent))
    plan = _plan_stages(manifest, incumbent, candidate)
    manifest_payload = {
        "path": str(manifest.source_path),
        "sha256": manifest.sha256,
        "id": manifest.id,
    }
    if dry_run:
        return {
            "schema": RESULT_SCHEMA,
            "dry_run": True,
            "manifest": manifest_payload,
            "incumbent_commit": manifest.incumbent_commit,
            "candidate_commit": manifest.candidate_commit,
            "benchmark_manifest": manifest.benchmark_manifest,
            "contexts": {"incumbent_dir": str(incumbent), "candidate_dir": str(candidate)},
            "thresholds": {"min_local_speedup": manifest.min_local_speedup,
                           "min_e2e_gain": manifest.min_e2e_gain},
            "metrics": [
                {"name": spec.name, "regex": spec.regex.pattern,
                 "direction": spec.direction, "stage": spec.stage, "compare": spec.compare}
                for spec in manifest.metrics
            ],
            "plan": plan,
            "verdict": None,
            "reasons": [],
        }
    evidence.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failure_reason: str | None = None
    for index, item in enumerate(plan):
        specs = _specs_for_stage(str(item["stage"]), manifest.metrics)
        record = _execute(item, evidence, specs)
        records.append(record)
        if record["status"] != "pass":
            failure_reason = (f"{record['stage']}[{record['context']}] exited "
                              f"{record['returncode']}; later stages not run")
            for skipped in plan[index + 1:]:
                records.append(_record(str(skipped["stage"]), str(skipped["context"]),
                                       skipped, status="not_run", returncode=None,
                                       duration_s=None))
            break
    all_stages_passed = all(record["status"] == "pass" for record in records)
    evidence_items = _collect_metrics(manifest, records)
    reproduction = _reproduction_check(manifest, records)
    verdict, reasons = _decide(manifest, evidence_items, reproduction,
                               all_stages_passed=all_stages_passed,
                               failure_reason=failure_reason)
    result = {
        "schema": RESULT_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "dry_run": False,
        "manifest": manifest_payload,
        "incumbent_commit": manifest.incumbent_commit,
        "candidate_commit": manifest.candidate_commit,
        "candidate_worktree": manifest.candidate_worktree,
        "benchmark_manifest": {
            "path": manifest.benchmark_manifest,
            "sha256": _sha256_path(_resolve(manifest.benchmark_manifest,
                                            manifest.source_path.parent)),
        },
        "contexts": {"incumbent_dir": str(incumbent), "candidate_dir": str(candidate)},
        "thresholds": {"min_local_speedup": manifest.min_local_speedup,
                       "min_e2e_gain": manifest.min_e2e_gain},
        "stages": records,
        "metrics": evidence_items,
        "reproduction": reproduction,
        "failure_reason": failure_reason,
        "verdict": verdict,
        "reasons": reasons,
        "evidence_dir": str(evidence),
    }
    result_path = evidence / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (evidence / "REPORT.md").write_text(_report_markdown(result))
    return result


def _report_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    verdict = str(result["verdict"]).upper()
    lines.append(f"# Autoresearch Experiment Report — {result['manifest']['id']}")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"- Manifest: `{result['manifest']['path']}` "
                 f"(sha256 `{result['manifest']['sha256']}`)")
    lines.append(f"- Incumbent commit: `{result['incumbent_commit']}`")
    lines.append(f"- Candidate commit: `{result['candidate_commit']}`")
    lines.append(f"- Incumbent dir: `{result['contexts']['incumbent_dir']}`")
    lines.append(f"- Candidate dir: `{result['contexts']['candidate_dir']}`")
    lines.append(f"- Evidence dir: `{result['evidence_dir']}`")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- `min_local_speedup`: {result['thresholds']['min_local_speedup']}")
    lines.append(f"- `min_e2e_gain`: {result['thresholds']['min_e2e_gain']}")
    lines.append("")
    lines.append("## Stages")
    lines.append("")
    lines.append("| stage | context | command | exit | duration (s) | status |")
    lines.append("|---|---|---|---:|---:|---|")
    for record in result["stages"]:
        command = shlex.join(record["command"])
        returncode = "" if record["returncode"] is None else str(record["returncode"])
        duration = "" if record["duration_s"] is None else f"{record['duration_s']:.3f}"
        lines.append(f"| {record['stage']} | {record['context']} | `{command}` | "
                     f"{returncode} | {duration} | {record['status']} |")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | stage | direction | before | after | gain | threshold | pass |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for item in result["metrics"]:
        def _fmt(value: Any) -> str:
            return "" if value is None else f"{float(value):.6g}"
        lines.append(f"| {item['name']} | {item['stage']} | {item['direction']} | "
                     f"{_fmt(item['before'])} | {_fmt(item['after'])} | {_fmt(item['gain'])} | "
                     f"{_fmt(item['threshold'])} | {item['meets_threshold']} |")
    lines.append("")
    lines.append("## Fresh-session reproduction")
    lines.append("")
    lines.append(f"- Consistent: {result['reproduction']['consistent']}")
    for check in result["reproduction"]["checks"]:
        lines.append(f"- `{check['name']}`: candidate gain "
                     f"{check['candidate_gain']:.6g}, reproduced gain "
                     f"{check['reproduced_gain']:.6g}, consistent {check['consistent']}")
    if not result["reproduction"]["checks"]:
        lines.append("- No context metrics available to reproduce.")
    lines.append("")
    lines.append("## Reasons")
    lines.append("")
    for reason in result["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autoresearch_runner",
        description="Execute the proven autoresearch loop from a declarative manifest.")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to the experiment manifest JSON.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the manifest and print the command sequence only.")
    parser.add_argument("--incumbent-dir", type=Path, default=None,
                        help="Working directory for the incumbent context (default: cwd).")
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="Working directory for the candidate context (default: manifest "
                             "candidate_worktree, else the incumbent directory).")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="Override the manifest evidence directory.")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as error:
        print(f"autoresearch_runner: {error}", file=sys.stderr)
        return EXIT_ERROR
    try:
        result = run_experiment(manifest, incumbent_dir=args.incumbent_dir,
                                candidate_dir=args.candidate_dir,
                                evidence_dir=args.evidence_dir, dry_run=args.dry_run)
    except AutoresearchError as error:
        print(f"autoresearch_runner: {error}", file=sys.stderr)
        return EXIT_ERROR

    if args.dry_run:
        print(json.dumps(result["plan"], indent=2, sort_keys=True))
        return EXIT_PROMOTE
    summary = {key: result[key] for key in ("verdict", "reasons", "metrics", "reproduction")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return {"promote": EXIT_PROMOTE, "reject": EXIT_REJECT,
            "inconclusive": EXIT_INCONCLUSIVE}[str(result["verdict"])]


if __name__ == "__main__":
    raise SystemExit(main())
