"""Temporary workspaces for reproducible artifact/test evidence."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


@contextmanager
def artifact_workspace(root: Path | None = None) -> Iterator[Path]:
    """Yield an empty temporary directory and remove it when the scope exits."""

    parent = str(root) if root is not None else None
    with TemporaryDirectory(prefix="superinfer-artifact-", dir=parent) as directory:
        yield Path(directory)

