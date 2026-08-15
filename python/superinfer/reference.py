"""Independent high-precision comparison helpers for future tensor oracles."""

from __future__ import annotations

from dataclasses import dataclass
from math import fabs
from typing import Sequence


@dataclass(frozen=True)
class Comparison:
    """Comparison result retaining the largest error and its flat index."""

    equal: bool
    max_abs_error: float
    max_rel_error: float
    max_error_index: int


def compare_values(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    abs_tol: float,
    rel_tol: float,
) -> Comparison:
    """Compare flat numeric sequences without importing a production tensor framework."""

    if len(expected) != len(actual):
        return Comparison(False, float("inf"), float("inf"), min(len(expected), len(actual)))
    max_abs = 0.0
    max_rel = 0.0
    max_index = 0
    for index, (reference, candidate) in enumerate(zip(expected, actual, strict=True)):
        absolute = fabs(candidate - reference)
        relative = absolute / max(fabs(reference), 1e-30)
        if absolute > max_abs or relative > max_rel:
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            max_index = index
    return Comparison(max_abs <= abs_tol or max_rel <= rel_tol, max_abs, max_rel, max_index)

