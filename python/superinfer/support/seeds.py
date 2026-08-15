"""Deterministic seed capture with restoration of caller state."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedRecord:
    """Seed metadata retained in evidence bundles."""

    seed: int
    algorithm: str = "python.random"


class SeedCapture:
    """Temporarily seeds Python RNG and restores the prior state on exit."""

    def __init__(self, seed: int) -> None:
        self._record = SeedRecord(seed)
        self._prior_state: object | None = None

    def __enter__(self) -> SeedRecord:
        self._prior_state = random.getstate()
        random.seed(self._record.seed)
        return self._record

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._prior_state is not None:
            random.setstate(self._prior_state)

