"""Unit tests for tools/qwen38_nvfp4_census.py.

The census is a parser over a produced Physical Plan dump, so it needs golden, corruption, and
truncation coverage: a malformed command must be reported, never silently counted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_CENSUS_PATH = REPO_ROOT / "tools" / "qwen38_nvfp4_census.py"

_spec = importlib.util.spec_from_file_location("qwen38_nvfp4_census", _CENSUS_PATH)
assert _spec is not None and _spec.loader is not None
census = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = census
_spec.loader.exec_module(census)


def _nvfp4_command(rows: int, inputs: int, command_id: int = 0) -> str:
    return (
        f"id={command_id} kernel=13 buffers="
        f"{100}:{inputs * 4}:1,"
        f"{200}:{rows * (inputs // 2)}:6,"
        f"{300}:{rows * (inputs // 16)}:6,"
        f"{400}:4:1,"
        f"{500}:{rows * 4}:1"
    )


def test_golden_census_shapes_and_counts() -> None:
    text = "\n".join(
        [
            "id=0 kernel=8 buffers=1:4:5,2:8:3",
            _nvfp4_command(248320, 5120, 1),
            _nvfp4_command(17408, 5120, 2),
            _nvfp4_command(17408, 5120, 3),
            _nvfp4_command(12288, 5120, 4),
        ]
    )
    counts, errors = census.derive_census(text)
    assert errors == []
    assert counts[census.Nvfp4Shape(248320, 5120)] == 1
    assert counts[census.Nvfp4Shape(17408, 5120)] == 2
    assert counts[census.Nvfp4Shape(12288, 5120)] == 1
    assert sum(counts.values()) == 4


def test_ignores_non_nvfp4_kernels() -> None:
    text = "\n".join(
        [
            "id=0 kernel=17 buffers=1:20480:1,2:10240:3",
            "id=1 kernel=4 buffers=1:4:1,2:4:1,3:4:1",
        ]
    )
    counts, errors = census.derive_census(text)
    assert counts == {}
    assert errors == ["no nvfp4_linear commands found in the dump"]


def test_wrong_packed_byte_count_is_reported_not_counted() -> None:
    bad = _nvfp4_command(17408, 5120).replace(f":{17408 * 2560}:6", ":999:6")
    counts, errors = census.derive_census(bad)
    assert counts == {}
    assert any("packed bytes" in error for error in errors)


def test_wrong_scale_byte_count_is_reported() -> None:
    bad = _nvfp4_command(1024, 5120).replace(f":{1024 * 320}:6", ":7:6")
    counts, errors = census.derive_census(bad)
    assert any("scale bytes" in error for error in errors)


def test_wrong_operand_arity_is_reported() -> None:
    bad = "id=0 kernel=13 buffers=1:20480:1,2:26214400:6,3:3276800:6,4:4:1"
    counts, errors = census.derive_census(bad)
    assert counts == {}
    assert any("operands" in error for error in errors)


def test_wrong_storage_dtype_is_reported() -> None:
    bad = _nvfp4_command(6144, 5120).replace(":6,", ":1,")
    counts, errors = census.derive_census(bad)
    assert any("u8 storage" in error for error in errors)


def test_truncated_dump_reports_no_commands() -> None:
    counts, errors = census.derive_census("id=0 kernel=13 buffer")
    assert counts == {}
    assert errors == ["no nvfp4_linear commands found in the dump"]


def test_header_render_asserts_total_and_is_deterministic() -> None:
    counts = {
        census.Nvfp4Shape(17408, 5120): 2,
        census.Nvfp4Shape(248320, 5120): 1,
    }
    first = census.render_header(counts, provenance="unit-test")
    second = census.render_header(counts, provenance="unit-test")
    assert first == second
    assert "kTotalNvfp4Launches = 3" in first
    # Deterministic ordering: highest multiplicity first, then largest rows.
    assert first.index('"17408x5120"') < first.index('"248320x5120"')
