#!/usr/bin/env python3
"""Distribution-level comparison for one S03-R logit row.

Compares a reference logit row against a candidate row without inferring an
acceptance threshold. Thresholds belong to the S03-R decision artifact and a
superseding ADR, never to this module.
"""

from __future__ import annotations

import math
from typing import Sequence

_EPSILON = 1e-12


def _argmax(values: Sequence[float]) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def _margin(values: Sequence[float], winner: int) -> float:
    runner_up = float("-inf")
    for index, value in enumerate(values):
        if index != winner and value > runner_up:
            runner_up = value
    if runner_up == float("-inf"):
        return float("inf")
    return float(values[winner]) - float(runner_up)


def _top_k_indices(values: Sequence[float], top_k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:top_k]


def _js_divergence(reference: Sequence[float], candidate: Sequence[float],
                   support: Sequence[int]) -> float:
    """Jensen-Shannon divergence over a renormalized bounded support.

    Softmax is computed after subtracting the support max, so common logit
    shifts are probabilistically harmless. ``_EPSILON`` is used only inside
    ``log`` for numerical stability and is documented here, not hidden.
    """
    ref_max = max(reference[i] for i in support)
    cand_max = max(candidate[i] for i in support)
    ref_exp = [math.exp(reference[i] - ref_max) for i in support]
    cand_exp = [math.exp(candidate[i] - cand_max) for i in support]
    ref_sum = math.fsum(ref_exp)
    cand_sum = math.fsum(cand_exp)
    divergence = 0.0
    for ref_e, cand_e in zip(ref_exp, cand_exp):
        p = ref_e / ref_sum
        q = cand_e / cand_sum
        m = 0.5 * (p + q)
        if p > 0.0:
            divergence += 0.5 * p * math.log(p / max(m, _EPSILON))
        if q > 0.0:
            divergence += 0.5 * q * math.log(q / max(m, _EPSILON))
    return max(divergence, 0.0)


def compare_distribution(reference: Sequence[float], candidate: Sequence[float],
                         top_k: int = 5) -> dict[str, float | int | bool]:
    """Compare one reference logit row against one candidate row.

    ``top_k`` fixes the index-overlap set. The Jensen-Shannon support is the
    union of both top-``k`` sets plus every index whose absolute error exceeds
    ``1e-9``, so materially changed tail rows remain visible while identical
    rows keep a bounded support. No acceptance threshold is inferred here.
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"logit lengths differ: reference={len(reference)} candidate={len(candidate)}"
        )
    if not reference:
        raise ValueError("logit vectors must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    reference = [float(v) for v in reference]
    candidate = [float(v) for v in candidate]

    errors = [abs(r - c) for r, c in zip(reference, candidate)]
    max_abs = max(errors)
    mean_abs = math.fsum(errors) / len(errors)
    rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))

    reference_argmax = _argmax(reference)
    candidate_argmax = _argmax(candidate)
    greedy_match = reference_argmax == candidate_argmax
    reference_margin = _margin(reference, reference_argmax)
    candidate_margin = _margin(candidate, candidate_argmax)

    effective_k = min(top_k, len(reference))
    ref_top = set(_top_k_indices(reference, effective_k))
    cand_top = set(_top_k_indices(candidate, effective_k))
    top_k_overlap = len(ref_top & cand_top) / effective_k
    reference_top_k = _top_k_indices(reference, effective_k)
    reference_winner_logit = float(reference[reference_argmax])

    support = sorted(ref_top | cand_top | {
        i for i, e in enumerate(errors) if e > 1e-9
    })
    js_divergence = _js_divergence(reference, candidate, support)

    denominator = reference_margin if math.isfinite(reference_margin) else 0.0
    if denominator <= 0.0:
        max_error_over_margin = float("inf") if max_abs > 0.0 else 0.0
    else:
        max_error_over_margin = max_abs / denominator

    return {
        "count": len(reference),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "greedy_match": greedy_match,
        "reference_argmax": reference_argmax,
        "candidate_argmax": candidate_argmax,
        "reference_argmax_margin": reference_margin,
        "candidate_argmax_margin": candidate_margin,
        "top_k": effective_k,
        "top_k_overlap": top_k_overlap,
        "reference_top_k": reference_top_k,
        "reference_winner_logit": reference_winner_logit,
        "js_divergence": js_divergence,
        "max_error_over_margin": max_error_over_margin,
    }
