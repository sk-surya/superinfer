"""Deterministic JSON evidence writers used by tests and later benchmark tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class EvidenceWriter:
    """Owns a root directory and writes normalized JSON/JSONL evidence beneath it."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self._root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def append_jsonl(self, name: str, payload: Mapping[str, Any]) -> int:
        path = self._root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        with path.open(encoding="utf-8") as stream:
            return sum(1 for _ in stream)
