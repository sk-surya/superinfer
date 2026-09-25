#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export for the SuperInfer Qwen runtime.

Produces a ranked kernel table, device-idle/launch-gap decomposition, CUDA
runtime API launch overhead, and synchronization counts. Reads only nsys
sqlite; no GPU required.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _string_ids(db: sqlite3.Connection) -> dict[int, str]:
    return {row[0]: row[1] for row in db.execute("select id, value from StringIds")}


def summarize(db_path: Path, top: int) -> dict:
    db = sqlite3.connect(str(db_path))
    names = _string_ids(db)

    kernels = []
    for start, end, device, stream, short in db.execute(
        'select start, "end", deviceId, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL'
    ):
        kernels.append((start, end, device, stream, names.get(short, str(short))))
    kernels.sort(key=lambda k: k[0])

    by_name: dict[str, list[int]] = {}
    for start, end, device, stream, name in kernels:
        by_name.setdefault(name, []).append(end - start)
    table = []
    for name, durations in by_name.items():
        total = sum(durations)
        table.append(
            {
                "kernel": name,
                "launches": len(durations),
                "total_ms": total / 1e6,
                "avg_us": (total / len(durations)) / 1e3,
                "max_us": max(durations) / 1e3,
            }
        )
    table.sort(key=lambda r: r["total_ms"], reverse=True)
    total_kernel_ms = sum(r["total_ms"] for r in table)

    gaps = []
    idle_ns = 0
    for previous, current in zip(kernels, kernels[1:]):
        gap = current[0] - previous[1]
        if gap > 0:
            idle_ns += gap
            gaps.append(gap)
    span = (kernels[-1][1] - kernels[0][0]) if kernels else 0

    runtime_by_api: dict[str, list[int]] = {}
    for start, end, name_id in db.execute(
        "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        runtime_by_api.setdefault(names.get(name_id, str(name_id)), []).append(end - start)
    runtime_table = [
        {"api": api, "calls": len(v), "total_ms": sum(v) / 1e6, "avg_us": sum(v) / len(v) / 1e3}
        for api, v in runtime_by_api.items()
    ]
    runtime_table.sort(key=lambda r: r["total_ms"], reverse=True)

    syncs = db.execute("select count(*) from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION").fetchone()[0]
    memcpy = db.execute(
        "select count(*), coalesce(sum(bytes),0) from CUPTI_ACTIVITY_KIND_MEMCPY"
    ).fetchone()

    return {
        "kernel_count": len(kernels),
        "kernel_total_ms": total_kernel_ms,
        "device_span_ms": span / 1e6,
        "device_idle_ms": idle_ns / 1e6,
        "device_idle_pct_of_span": (100.0 * idle_ns / span) if span else 0.0,
        "synchronizations": syncs,
        "memcpy_events": memcpy[0],
        "memcpy_bytes": memcpy[1],
        "kernel_table": table[:top],
        "runtime_api_table": runtime_table[:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    report = summarize(args.sqlite, args.top)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
