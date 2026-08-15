"""Public inspection helper kept separate from CLI argument handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from superinfer.artifact import inspect_artifact


def inspect_path(path: Path) -> dict[str, Any]:
    """Validate and summarize one `.sinf` path without requiring a GPU."""

    return inspect_artifact(path)

