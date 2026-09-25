"""Unit tests for the D-021 acceptance runner's repeat semantics and divergence reporting.

The runner's repeat loop spawns a fresh executable process per iteration, so ``--repeat 1`` is one
capture. A single hash trivially satisfies ``len({hash}) == 1``, which previously let a run that
requires repeatability report a verdict from one execution. These tests pin the fix and the exact
first-divergence reporting.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = REPO_ROOT / "tools" / "qwen38_s03r_acceptance.py"

_spec = importlib.util.spec_from_file_location("qwen38_s03r_acceptance", _RUNNER)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)


def _payload(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def test_repeat_one_is_inconclusive_when_d021_contract_supplied() -> None:
    invalid, reason = runner.repeat_semantics({}, True, 1)
    assert invalid is True
    assert reason == "D-021 repeatability requires at least two fresh process captures"


def test_repeat_one_is_inconclusive_when_contract_requires_repeatability() -> None:
    invalid, reason = runner.repeat_semantics({"require_repeatable": True}, False, 1)
    assert invalid is True
    assert reason is not None and "at least two fresh process captures" in reason


def test_repeat_two_is_accepted() -> None:
    invalid, reason = runner.repeat_semantics({}, True, 2)
    assert invalid is False and reason is None


def test_repeat_one_ok_when_repeatability_not_required() -> None:
    invalid, reason = runner.repeat_semantics({"require_repeatable": False}, False, 1)
    assert invalid is False and reason is None


def test_first_divergence_reports_exact_element() -> None:
    a = _payload([1.0, 2.0, 3.0, 4.0])
    b = _payload([1.0, 2.0, 3.5, 4.0])
    result = runner.first_divergence(a, b, vocab=2)
    assert result["identical"] is False
    assert result["first_differing_element"] == 2
    assert result["first_differing_row"] == 1
    assert result["first_differing_vocab_index"] == 0
    assert result["value_run_a"] == 3.0
    assert result["value_run_b"] == 3.5
    assert result["abs_difference"] == 0.5
    assert result["differing_elements_in_row"] == 1
    assert result["max_abs_over_row"] == 0.5


def test_first_divergence_identical_payload() -> None:
    payload = _payload([1.0, 2.0])
    assert runner.first_divergence(payload, payload, vocab=2)["identical"] is True


def test_first_divergence_truncated_payload_does_not_crash() -> None:
    a = _payload([1.0, 2.0, 3.0])
    b = _payload([1.0, 2.0])
    result = runner.first_divergence(a, b, vocab=3)
    assert result["identical"] is False
    assert "length_a" in result and "length_b" in result
