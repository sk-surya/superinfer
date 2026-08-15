"""Subprocess execution that preserves actionable failure context."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Captured process result with no implicit exception on a non-zero exit."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_command(
    args: Sequence[str],
    *,
    failure_path: Path | None = None,
    seed: int | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run a command and optionally write deterministic machine-readable failure context."""

    completed = subprocess.run(
        tuple(args),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    result = CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)
    if result.returncode != 0 and failure_path is not None:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "args": result.args,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "seed": seed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return result

