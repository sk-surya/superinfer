#!/usr/bin/env python3
"""Run the pinned Qwen3.8 corpus against the real artifact and reference.

This harness intentionally labels its corpus mode as ``decode_replay``: each prompt token is
submitted to the existing one-token physical plan in a fresh session. The separate static-prefill
entry-point evidence is recorded in the S03 artifact report; this harness does not conflate the two
execution modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


_STEP_RE = re.compile(r"(?:^| )token=(\d+) greedy=(\d+) logit=([-+0-9.eE]+)")
_COMMANDS_RE = re.compile(r" commands=(\d+)")
_STATE_RE = re.compile(r" state_buffers=(\d+)")


def parse_superinfer_output(output: str) -> dict[str, Any]:
    """Parse the stable summary emitted by the CUDA acceptance executable."""
    matches = _STEP_RE.findall(output)
    if not matches:
        raise ValueError(f"SuperInfer output has no token result: {output[-500:]}")
    commands = _COMMANDS_RE.search(output)
    state_buffers = _STATE_RE.search(output)
    if commands is None or state_buffers is None:
        raise ValueError(f"SuperInfer output lacks execution counters: {output[-500:]}")
    return {
        "greedy": [int(greedy) for _, greedy, _ in matches],
        "commands": int(commands.group(1)),
        "state_buffers": int(state_buffers.group(1)),
    }


def compare_logits(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, float | int]:
    """Return shape and FP32 error diagnostics without hiding a length mismatch."""
    if len(reference) != len(candidate):
        raise ValueError(f"logit lengths differ: reference={len(reference)} candidate={len(candidate)}")
    if not reference:
        raise ValueError("logit vectors must not be empty")
    errors = [abs(float(left) - float(right)) for left, right in zip(reference, candidate)]
    return {
        "count": len(errors),
        "max_abs": max(errors),
        "mean_abs": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def _floats(path: Path) -> list[float]:
    import struct

    payload = path.read_bytes()
    if len(payload) % 4 != 0:
        raise ValueError(f"capture is not FP32-aligned: {path}")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_case(
    executable: Path,
    artifact: Path,
    reference_script: Path,
    model_dir: Path,
    case: dict[str, Any],
    output_dir: Path,
    repeat: int,
    reference_python: str,
) -> dict[str, Any]:
    token_ids = case.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        return {"id": case.get("id", "unknown"), "status": "not_run", "reason": "token_ids_missing"}
    if not all(isinstance(token, int) and token >= 0 for token in token_ids):
        raise ValueError(f"invalid token IDs in case {case.get('id')}")
    case_id = str(case["id"])
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    tokens = ",".join(str(token) for token in token_ids)
    reference_capture = case_dir / "reference.f32"
    reference_command = [
        reference_python,
        str(reference_script),
        "--model-dir",
        str(model_dir),
        "--output",
        str(reference_capture),
        "--tokens",
        tokens,
    ]
    reference_run = subprocess.run(reference_command, check=False, capture_output=True, text=True)
    if reference_run.returncode != 0:
        raise RuntimeError(f"reference failed for {case_id}: {reference_run.stderr[-2000:]}")
    reference_logits = _floats(reference_capture)
    rows = len(token_ids)
    if len(reference_logits) % rows != 0:
        raise ValueError(f"reference rows are malformed for {case_id}")
    vocab = len(reference_logits) // rows
    reference_rows = [reference_logits[index * vocab : (index + 1) * vocab] for index in range(rows)]
    reference_diagnostics = json.loads(reference_capture.with_suffix(".json").read_text())

    runs: list[dict[str, Any]] = []
    for iteration in range(repeat):
        base_capture = case_dir / f"superinfer-{iteration}.f32"
        continuation_capture = case_dir / f"superinfer-{iteration}-continuation.f32"
        environment = os.environ.copy()
        environment.update(
            {
                "SUPERINFER_QWEN38_ARTIFACT": str(artifact),
                "SUPERINFER_QWEN38_INITIAL_TOKEN": str(token_ids[0]),
                "SUPERINFER_QWEN38_LOGITS_F32": str(base_capture),
            }
        )
        if len(token_ids) > 1:
            environment["SUPERINFER_QWEN38_CONTINUATION"] = "1"
            environment["SUPERINFER_QWEN38_CONTINUATION_TOKENS"] = ",".join(
                str(token) for token in token_ids[1:]
            )
            environment["SUPERINFER_QWEN38_CONTINUATION_LOGITS_F32"] = str(continuation_capture)
        run = subprocess.run([str(executable)], check=False, capture_output=True, text=True, env=environment)
        if run.returncode != 0:
            raise RuntimeError(f"SuperInfer failed for {case_id} run {iteration}: {run.stderr[-2000:]}")
        parsed = parse_superinfer_output(run.stdout)
        candidate_logits = _floats(base_capture)
        if len(token_ids) > 1:
            candidate_logits += _floats(continuation_capture)
        if len(candidate_logits) != len(reference_logits):
            raise ValueError(f"candidate rows are malformed for {case_id}")
        candidate_rows = [candidate_logits[index * vocab : (index + 1) * vocab] for index in range(rows)]
        metrics = [compare_logits(reference_rows[index], candidate_rows[index]) for index in range(rows)]
        runs.append(
            {
                "iteration": iteration,
                "greedy_sequence": parsed["greedy"],
                "token_match": parsed["greedy"] == reference_diagnostics["greedy_sequence"],
                "metrics": metrics,
                "capture_sha256": hashlib.sha256(
                    base_capture.read_bytes()
                    + (continuation_capture.read_bytes() if len(token_ids) > 1 else b"")
                ).hexdigest(),
                "commands": parsed["commands"],
                "state_buffers": parsed["state_buffers"],
                "stdout": run.stdout.strip(),
            }
        )
    return {
        "id": case_id,
        "status": "pass" if all(run["token_match"] for run in runs) else "fail",
        "mode": "decode_replay",
        "token_ids": token_ids,
        "reference_greedy_sequence": reference_diagnostics["greedy_sequence"],
        "reference_output_sha256": reference_diagnostics["output_sha256"],
        "runs": runs,
        "repeatable": len({run["capture_sha256"] for run in runs}) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    corpus = json.loads(args.corpus.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen38-s03-acceptance-", dir=args.output.parent) as temporary:
        results = [
            _run_case(
                args.executable,
                args.artifact,
                args.reference_script,
                args.model_dir,
                case,
                Path(temporary),
                args.repeat,
                args.reference_python,
            )
            for case in corpus["cases"]
        ]
    report = {
        "schema": "superinfer.qwen38.s03.acceptance-run.v1",
        "status": "pass" if all(result["status"] in {"pass", "not_run"} for result in results) else "fail",
        "evidence_mode": "decode_replay",
        "artifact": {"path": str(args.artifact), "sha256": _sha256(args.artifact)},
        "corpus": {"path": str(args.corpus), "sha256": _sha256(args.corpus)},
        "reference_script": str(args.reference_script),
        "repeat": args.repeat,
        "cases": results,
        "limitations": [
            "This corpus report exercises one-token decode replay; static sequence-shaped prefill is reported separately.",
            "Cases without pinned token_ids are recorded as not_run.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": results}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
