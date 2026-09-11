#!/usr/bin/env python3
"""S03-R decisive same-artifact acceptance runner.

Compares SuperInfer execution against an independent same-artifact oracle
(PyTorch/Transformers math over exact `.sinf` packed bytes; SuperInfer CUDA
kernels are never reused as their own oracle) across the results-first corpus.

Per row it records greedy agreement, top-k overlap, argmax margins,
max/mean/RMSE, Jensen-Shannon divergence, and error-to-margin ratio. The
module decides exactly one of ``contract_supersede_candidate``,
``real_superinfer_discrepancy``, or ``inconclusive``. Inconclusive evidence
is never converted into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qwen38_s03_acceptance import parse_superinfer_output  # noqa: E402
from tools.qwen38_same_artifact_metrics import compare_distribution  # noqa: E402

SCHEMA = "superinfer.qwen38.s03r.acceptance.v1"


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = probability * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def aggregate_rows(rows: Sequence[dict[str, Any]], repeatable: bool) -> dict[str, Any]:
    """Aggregate per-row same-artifact comparisons into a decision summary."""
    if not rows:
        raise ValueError("no rows to aggregate")
    for key in ("max_abs", "mean_abs", "rmse", "js_divergence",
                "max_error_over_margin", "top_k_overlap", "greedy_match"):
        for row in rows:
            if key not in row:
                raise ValueError(f"row is missing metric: {key}")
    ordered = sorted(rows, key=lambda r: float(r["max_abs"]), reverse=True)
    summary: dict[str, Any] = {"total_rows": len(rows), "repeatable": bool(repeatable)}
    for key in ("max_abs", "mean_abs", "rmse", "js_divergence",
                "max_error_over_margin", "top_k_overlap"):
        values = sorted(float(row[key]) for row in rows)
        summary[f"{key}_max"] = values[-1]
        summary[f"{key}_p50"] = _quantile(values, 0.50)
        summary[f"{key}_p95"] = _quantile(values, 0.95)
    summary["greedy_mismatch_rows"] = sum(0 if row["greedy_match"] else 1 for row in rows)
    summary["margin_unsafe_rows"] = sum(
        1 for row in rows if float(row["max_error_over_margin"]) >= 1.0
    )
    summary["worst_row_rank"] = [
        {"rank": rank, "max_abs": ordered[rank]["max_abs"],
         "greedy_match": ordered[rank]["greedy_match"],
         "max_error_over_margin": ordered[rank]["max_error_over_margin"]}
        for rank in range(min(5, len(ordered)))
    ]
    return summary


def decide_s03r(summary: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Decide exactly one S03-R outcome from an aggregated summary.

    ``contract`` carries evidence-derived same-artifact bounds; it never
    weakens local kernel/layer differentials, which are inputs, not outputs.
    """
    reasons: list[str] = []
    for key in ("total_rows", "repeatable", "greedy_mismatch_rows", "margin_unsafe_rows",
                "max_abs_max", "mean_abs_max", "rmse_max", "js_divergence_max"):
        if key not in summary:
            return {"decision": "inconclusive",
                    "reasons": [f"summary is missing required field: {key}"]}
    if not summary["repeatable"] and contract.get("require_repeatable", True):
        reasons.append("captures are not repeatable across fresh sessions")
        return {"decision": "inconclusive", "reasons": reasons}
    if int(summary["greedy_mismatch_rows"]) > 0:
        reasons.append(f"{summary['greedy_mismatch_rows']} row(s) flip the greedy winner")
        return {"decision": "real_superinfer_discrepancy", "reasons": reasons}
    if int(summary["margin_unsafe_rows"]) > 0 and contract.get("require_margin_safe", True):
        reasons.append(f"{summary['margin_unsafe_rows']} row(s) have error >= winner margin")
        return {"decision": "real_superinfer_discrepancy", "reasons": reasons}
    bound_failures = []
    for metric in ("max_abs", "mean_abs", "rmse", "js_divergence"):
        bound = contract.get(f"same_artifact_{metric}")
        if bound is not None and float(summary[f"{metric}_max"]) > float(bound):
            bound_failures.append(f"{metric} {summary[f'{metric}_max']} > {bound}")
    if bound_failures:
        reasons.append("same-artifact numerical divergence beyond layer-justified bound: "
                       + "; ".join(bound_failures))
        return {"decision": "real_superinfer_discrepancy", "reasons": reasons}
    reasons.append("same-artifact execution agrees within the layer-justified contract; "
                   "remaining source-reference mismatch is quantization-contract scope")
    return {"decision": "contract_supersede_candidate", "reasons": reasons}


def bf16_ulp(value: float) -> float:
    """Spacing of BF16 values at the magnitude of ``value`` (8 mantissa bits)."""
    import math

    if value == 0.0 or not math.isfinite(value):
        raise ValueError("BF16 floor is defined only for finite nonzero logits")
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def decide_d021(case_rows: dict[str, list[dict[str, Any]]], contract: dict[str, Any],
                repeatable: bool) -> dict[str, Any]:
    """Apply the D-021 margin-qualified contract to per-row same-artifact metrics.

    ``case_rows`` maps case id to the row-metric dicts produced by
    :func:`compare_distribution` (which must include ``reference_argmax`` and
    ``reference_argmax_margin``). Rows listed in the contract's
    ``outlier_rows`` are reported but non-blocking; any other breach fails.
    """
    for key in ("tie_top_k", "tie_min_overlap", "tie_max_js", "max_js",
                "max_rmse", "max_mean_abs", "outlier_rows"):
        if key not in contract:
            return {"verdict": "inconclusive",
                    "reasons": [f"D-021 contract is missing required field: {key}"]}
    outlier_ranges = [(entry["case"], int(entry["from_row"]))
                      for entry in contract["outlier_rows"]]
    strict_rows = strict_failures = tie_rows = tie_failures = 0
    dist_failures: list[str] = []
    outlier_reports: list[str] = []
    reasons: list[str] = []
    for case_id, rows in case_rows.items():
        listed_from = next((start for name, start in outlier_ranges if name == case_id), None)
        for row in rows:
            row_index = int(row.get("row", -1))
            if listed_from is not None and row_index >= listed_from:
                outlier_reports.append(
                    f"{case_id} r{row_index}: outlier max={row['max_abs']:.3f} "
                    f"match={row['greedy_match']} js={row['js_divergence']:.4g}")
                continue
            if "reference_argmax_margin" not in row or "reference_argmax" not in row:
                return {"verdict": "inconclusive",
                        "reasons": [f"{case_id} r{row_index}: row lacks margin fields"]}
            margin = float(row["reference_argmax_margin"])
            if "reference_winner_logit" not in row or "reference_top_k" not in row:
                return {"verdict": "inconclusive",
                        "reasons": [f"{case_id} r{row_index}: row lacks tie-set fields"]}
            ulp = bf16_ulp(float(row["reference_winner_logit"]))
            if margin > ulp:
                strict_rows += 1
                if not row["greedy_match"]:
                    strict_failures += 1
                    reasons.append(f"{case_id} r{row_index}: strict-row flip "
                                   f"(margin {margin:.4f} > ulp {ulp:.4f})")
            else:
                tie_rows += 1
                if row["candidate_argmax"] not in list(row["reference_top_k"]):
                    tie_failures += 1
                    reasons.append(f"{case_id} r{row_index}: tie winner "
                                   f"{row['candidate_argmax']} outside reference top-k")
                if float(row["top_k_overlap"]) < float(contract["tie_min_overlap"]):
                    tie_failures += 1
                    reasons.append(f"{case_id} r{row_index}: tie top-k overlap "
                                   f"{row['top_k_overlap']} below floor")
                if float(row["js_divergence"]) > float(contract["tie_max_js"]):
                    tie_failures += 1
                    reasons.append(f"{case_id} r{row_index}: tie JS "
                                   f"{row['js_divergence']:.4g} above bound")
            if float(row["js_divergence"]) > float(contract["max_js"]):
                dist_failures.append(f"{case_id} r{row_index}: JS {row['js_divergence']:.4g}")
            if float(row["rmse"]) > float(contract["max_rmse"]):
                dist_failures.append(f"{case_id} r{row_index}: RMSE {row['rmse']:.4g}")
            if float(row["mean_abs"]) > float(contract["max_mean_abs"]):
                dist_failures.append(f"{case_id} r{row_index}: mean {row['mean_abs']:.4g}")
    if not repeatable:
        return {"verdict": "inconclusive", "reasons": ["captures are not repeatable"]}
    failures = reasons + dist_failures
    if failures:
        return {"verdict": "fail", "reasons": failures,
                "strict_rows": strict_rows, "strict_failures": strict_failures,
                "tie_rows": tie_rows, "tie_failures": tie_failures,
                "outlier_reports": outlier_reports}
    reasons.append(f"D-021 holds: {strict_rows} strict rows greedy-exact, "
                   f"{tie_rows} tie rows within near-tie set, distributional bounds clear, "
                   f"{len(outlier_reports)} listed outlier rows reported")
    return {"verdict": "pass", "reasons": reasons,
            "strict_rows": strict_rows, "tie_rows": tie_rows,
            "outlier_reports": outlier_reports}


def _floats(path: Path) -> list[float]:
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


def _run_superinfer_case(executable: Path, artifact: Path, case: dict[str, Any],
                         output_dir: Path, repeat: int) -> dict[str, Any]:
    token_ids = [int(t) for t in case["token_ids"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for iteration in range(repeat):
        base_capture = output_dir / f"superinfer-{iteration}.f32"
        continuation_capture = output_dir / f"superinfer-{iteration}-continuation.f32"
        environment = os.environ.copy()
        for key in [k for k in environment if k.startswith("SUPERINFER_QWEN38_")]:
            environment.pop(key)
        environment.update({
            "SUPERINFER_QWEN38_ARTIFACT": str(artifact),
            "SUPERINFER_QWEN38_INITIAL_TOKEN": str(token_ids[0]),
            "SUPERINFER_QWEN38_LOGITS_F32": str(base_capture),
        })
        if len(token_ids) > 1:
            environment["SUPERINFER_QWEN38_CONTINUATION"] = "1"
            environment["SUPERINFER_QWEN38_CONTINUATION_TOKENS"] = ",".join(
                str(t) for t in token_ids[1:])
            environment["SUPERINFER_QWEN38_CONTINUATION_LOGITS_F32"] = str(continuation_capture)
        run = subprocess.run([str(executable)], check=False, capture_output=True,
                             text=True, env=environment)
        if run.returncode != 0:
            raise RuntimeError(f"SuperInfer failed for {case['id']} run {iteration}: "
                               f"exit={run.returncode} stderr={run.stderr[-2000:]} "
                               f"stdout={run.stdout[-500:]}")
        parsed = parse_superinfer_output(run.stdout)
        payload = base_capture.read_bytes()
        if len(token_ids) > 1:
            payload += continuation_capture.read_bytes()
        runs.append({
            "iteration": iteration,
            "greedy": parsed["greedy"],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "commands": parsed["commands"],
            "state_buffers": parsed["state_buffers"],
            "logits": list(struct.unpack(f"<{len(payload) // 4}f", payload)),
        })
    return {
        "id": case["id"],
        "token_ids": token_ids,
        "runs": runs,
        "repeatable": len({r["payload_sha256"] for r in runs}) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--sinf-reference-dir", type=Path, required=True,
                        help="per-case .f32+.json from qwen38_corpus_reference.py --sinf-artifact")
    parser.add_argument("--source-reference-dir", type=Path, default=None,
                        help="optional per-case .f32+.json from the safetensors source oracle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--contract", type=Path, default=None,
                        help="JSON with same_artifact_* bounds; decision uses it verbatim")
    parser.add_argument("--d021-contract", type=Path, default=None,
                        help="D-021 margin-qualified contract JSON; adds a verdict section")
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")

    corpus = json.loads(args.corpus.read_text())
    contract = json.loads(args.contract.read_text()) if args.contract else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    overall_repeatable = True
    with tempfile.TemporaryDirectory(prefix="qwen38-s03r-", dir=args.output.parent) as temporary:
        for case in corpus["cases"]:
            token_ids = case.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                case_reports.append({"id": case.get("id", "unknown"), "status": "not_run",
                                     "reason": "token_ids_missing"})
                continue
            case_id = str(case["id"])
            ref_capture = args.sinf_reference_dir / f"{case_id}.f32"
            ref_diag = json.loads(ref_capture.with_suffix(".json").read_text())
            ref_logits = _floats(ref_capture)
            vocab = int(ref_diag["logits_per_step"])
            rows = len(token_ids)
            ref_rows = [ref_logits[i * vocab:(i + 1) * vocab] for i in range(rows)]
            target = _run_superinfer_case(args.executable, args.artifact, case,
                                          Path(temporary) / case_id, args.repeat)
            overall_repeatable = overall_repeatable and target["repeatable"]
            first = target["runs"][0]["logits"]
            cand_rows = [first[i * vocab:(i + 1) * vocab] for i in range(rows)]
            row_metrics = [compare_distribution(ref_rows[i], cand_rows[i], top_k=args.top_k)
                           for i in range(rows)]
            for index, metrics in enumerate(row_metrics):
                metrics["row"] = index
            source_rows = None
            if args.source_reference_dir is not None:
                src = _floats(args.source_reference_dir / f"{case_id}.f32")
                source_rows = [src[i * vocab:(i + 1) * vocab] for i in range(rows)]
            case_reports.append({
                "id": case_id,
                "status": "pass",
                "token_ids": token_ids,
                "sinf_reference_sha256": ref_diag["output_sha256"],
                "sinf_reference_greedy": ref_diag["greedy_sequence"],
                "superinfer_greedy": target["runs"][0]["greedy"],
                "repeatable": target["repeatable"],
                "capture_sha256": target["runs"][0]["payload_sha256"],
                "rows": row_metrics,
                "source_attribution": (
                    [compare_distribution(source_rows[i], cand_rows[i], top_k=args.top_k)
                     for i in range(rows)] if source_rows is not None else None),
            })
            all_rows.extend(row_metrics)
    summary = aggregate_rows(all_rows, repeatable=overall_repeatable)
    decision = decide_s03r(summary, contract)
    d021_verdict = None
    if args.d021_contract is not None:
        d021 = json.loads(args.d021_contract.read_text())
        d021_rows = {report["id"]: report["rows"] for report in case_reports
                     if report.get("status") == "pass"}
        d021_verdict = decide_d021(d021_rows, d021, overall_repeatable)
    report = {
        "schema": SCHEMA,
        "artifact": {"path": str(args.artifact), "sha256": _sha256(args.artifact)},
        "corpus": {"path": str(args.corpus), "sha256": _sha256(args.corpus)},
        "contract": contract,
        "d021_contract": (str(args.d021_contract) if args.d021_contract else None),
        "d021_verdict": d021_verdict,
        "top_k": args.top_k,
        "repeat": args.repeat,
        "summary": summary,
        "decision": decision,
        "cases": case_reports,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": decision["decision"], "summary": summary,
                      "reasons": decision["reasons"],
                      "d021_verdict": d021_verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
