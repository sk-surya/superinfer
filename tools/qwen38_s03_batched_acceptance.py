#!/usr/bin/env python3
"""Compare the real Qwen3.8 artifact against cached batched reference captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.qwen38_s03_acceptance import parse_superinfer_output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floats(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 4 != 0:
        raise ValueError(f"capture is not FP32-aligned: {path}")
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def metrics(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, float | int]:
    if len(reference) != len(candidate):
        raise ValueError(f"capture lengths differ: {len(reference)} != {len(candidate)}")
    errors = [abs(float(left) - float(right)) for left, right in zip(reference, candidate)]
    return {
        "count": len(errors),
        "max_abs": max(errors),
        "mean_abs": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def run_case(executable: Path, artifact: Path, reference_dir: Path, case: dict[str, Any],
             output_dir: Path, repeat: int) -> dict[str, Any]:
    case_id = str(case["id"])
    token_ids = [int(token) for token in case["token_ids"]]
    reference_path = reference_dir / f"{case_id}.f32"
    reference_diagnostics = json.loads(reference_path.with_suffix(".json").read_text())
    reference_values = floats(reference_path)
    vocab = int(reference_diagnostics["logits_per_step"])
    if len(reference_values) != len(token_ids) * vocab:
        raise ValueError(f"reference rows are malformed for {case_id}")
    case_output = output_dir / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for iteration in range(repeat):
        base_capture = case_output / f"run-{iteration}-base.f32"
        continuation_capture = case_output / f"run-{iteration}-continuation.f32"
        environment = os.environ.copy()
        environment.update({
            "SUPERINFER_QWEN38_ARTIFACT": str(artifact),
            "SUPERINFER_QWEN38_INITIAL_TOKEN": str(token_ids[0]),
            "SUPERINFER_QWEN38_LOGITS_F32": str(base_capture),
        })
        if len(token_ids) > 1:
            environment.update({
                "SUPERINFER_QWEN38_CONTINUATION": "1",
                "SUPERINFER_QWEN38_CONTINUATION_TOKENS": ",".join(str(token) for token in token_ids[1:]),
                "SUPERINFER_QWEN38_CONTINUATION_LOGITS_F32": str(continuation_capture),
            })
        process = subprocess.run([str(executable)], capture_output=True, text=True,
                                 check=False, env=environment)
        if process.returncode != 0:
            raise RuntimeError(f"SuperInfer failed for {case_id} run {iteration}: {process.stderr[-2000:]}")
        parsed = parse_superinfer_output(process.stdout)
        candidate_payload = base_capture.read_bytes()
        if len(token_ids) > 1:
            candidate_payload += continuation_capture.read_bytes()
        candidate_values = list(struct.unpack(f"<{len(candidate_payload) // 4}f", candidate_payload))
        if len(candidate_values) != len(reference_values):
            raise ValueError(f"candidate rows are malformed for {case_id}")
        row_metrics = [
            metrics(reference_values[position * vocab : (position + 1) * vocab],
                    candidate_values[position * vocab : (position + 1) * vocab])
            for position in range(len(token_ids))
        ]
        runs.append({
            "iteration": iteration,
            "greedy_sequence": parsed["greedy"],
            "token_match": parsed["greedy"] == reference_diagnostics["greedy_sequence"],
            "capture_sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "metrics": row_metrics,
            "commands": parsed["commands"],
            "state_buffers": parsed["state_buffers"],
            "stdout": process.stdout.strip(),
        })
    return {
        "id": case_id,
        "token_ids": token_ids,
        "token_ids_sha256": case["token_ids_sha256"],
        "reference_capture_sha256": reference_diagnostics["output_sha256"],
        "reference_greedy_sequence": reference_diagnostics["greedy_sequence"],
        "runs": runs,
        "token_match": all(run["token_match"] for run in runs),
        "repeatable": len({run["capture_sha256"] for run in runs}) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    corpus = json.loads(args.corpus.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [run_case(args.executable, args.artifact, args.reference_dir, case, args.output, args.repeat)
             for case in corpus["cases"]]
    report = {
        "schema": "superinfer.qwen38.s03.batched.acceptance.v1",
        "status": "pass" if all(case["token_match"] and case["repeatable"] for case in cases) else "fail",
        "mode": "decode_replay_against_batched_reference",
        "artifact": {"path": str(args.artifact), "sha256": sha256(args.artifact)},
        "corpus": {"path": str(args.corpus), "sha256": sha256(args.corpus)},
        "repeat": args.repeat,
        "cases": cases,
        "limitations": [
            "Runtime corpus execution uses the validated one-token physical plan with explicit state continuation.",
            "Reference and candidate logits are compared as FP32 rows after BF16 candidate conversion.",
            "This report contains correctness and determinism evidence, not performance claims.",
        ],
    }
    report_path = args.output / "acceptance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": cases}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
