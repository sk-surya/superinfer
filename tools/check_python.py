"""Dependency-free Python syntax and import-boundary check for CPU CI."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_CORE_IMPORTS = {"torch", "transformers", "safetensors", "cuda"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    package = root / "python" / "superinfer"
    errors: list[str] = []
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in FORBIDDEN_CORE_IMPORTS:
                    errors.append(f"{path}: forbidden core import {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"checked {len(tuple(package.rglob('*.py')))} Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

