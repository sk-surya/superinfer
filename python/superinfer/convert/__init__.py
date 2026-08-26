"""Model conversion frontends and normalized artifact conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from superinfer.artifact import write_artifact
from .qwen38 import write_qwen38_metadata_artifact, write_qwen38_payload_artifact


def convert_file(source: Path, output: Path) -> None:
    """Convert normalized JSON frontend output into a deterministic `.sinf` artifact."""

    normalized: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    write_artifact(normalized, output)


__all__ = ["convert_file", "write_qwen38_metadata_artifact", "write_qwen38_payload_artifact"]
