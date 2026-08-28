#!/usr/bin/env python3
"""Verify pinned Qwen3.8 text/chat tokenization independently of model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


def canonical_token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_case(tokenizer: Any, case: dict[str, Any]) -> list[int]:
    if "messages" in case:
        values = tokenizer.apply_chat_template(
            case["messages"],
            add_generation_prompt=bool(case.get("add_generation_prompt", False)),
            tokenize=True,
        )
    else:
        values = tokenizer.encode(str(case["text"]), add_special_tokens=False)
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    all_passed = False
    try:
        from transformers import AutoTokenizer

        corpus = json.loads(args.corpus.read_text())
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        expected_files = corpus["tokenizer"]["files"]
        file_results = {
            name: {"expected": expected, "actual": file_sha256(args.model_dir / name)}
            for name, expected in expected_files.items()
        }
        cases: list[dict[str, Any]] = []
        all_passed = all(result["expected"] == result["actual"] for result in file_results.values())
        for case in corpus["cases"]:
            actual = tokenize_case(tokenizer, case)
            expected = [int(value) for value in case["token_ids"]]
            token_result = {
                "id": case["id"],
                "expected_token_ids": expected,
                "actual_token_ids": actual,
                "expected_sha256": case["token_ids_sha256"],
                "actual_sha256": canonical_token_hash(actual),
                "pass": actual == expected and canonical_token_hash(actual) == case["token_ids_sha256"],
            }
            all_passed = all_passed and token_result["pass"]
            cases.append(token_result)
        report = {
            "schema": "superinfer.qwen38.tokenizer.contract.v1",
            "status": "pass" if all_passed else "fail",
            "model": args.model_dir.name,
            "transformers_version": __import__("transformers").__version__,
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_files": file_results,
            "cases": cases,
        }
    except Exception as error:  # noqa: BLE001 - CLI must leave bounded evidence on any input failure.
        report = {
            "schema": "superinfer.qwen38.tokenizer.contract.v1",
            "status": "fail",
            "model": args.model_dir.name,
            "error": f"{type(error).__name__}: {error}",
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": cases}, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
