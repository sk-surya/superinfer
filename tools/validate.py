"""Run the reproducible CPU validation profile and retain failure evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_step(
    name: str,
    command: tuple[str, ...],
    *,
    cwd: Path,
    evidence: Path,
    env: dict[str, str],
) -> None:
    completed = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f"{name}.json").write_text(
            json.dumps(
                {
                    "step": name,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"validation step failed: {name}")
    print(f"PASS {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="also run the sanitizer and wheel checks"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build" / "cpu-dev"
    evidence = build_dir / "artifacts" / "S00" / "validation"
    environment = {**os.environ, "PYTHONPATH": str(root / "python")}

    run_step("python-check", (sys.executable, "tools/check_python.py"), cwd=root, evidence=evidence,
             env=environment)
    run_step("python-tests", (sys.executable, "-m", "unittest", "discover", "-s", "tests",
                               "-p", "test_*.py"), cwd=root, evidence=evidence, env=environment)
    run_step("cmake-configure", ("cmake", "--preset", "cpu-dev"), cwd=root, evidence=evidence,
             env=environment)
    run_step(
        "cmake-build", ("cmake", "--build", "--preset", "cpu-dev"), cwd=root, evidence=evidence,
        env=environment
    )
    run_step(
        "ctest", ("ctest", "--preset", "cpu-dev"), cwd=root, evidence=evidence, env=environment
    )

    validation_tmp = Path(
        os.environ.get("SUPERINFER_VALIDATION_TMP", "/tmp/superinfer-validation")
    )
    install_dir = validation_tmp / "install"
    run_step("cmake-install", ("cmake", "--install", str(build_dir), "--prefix", str(install_dir)),
             cwd=root, evidence=evidence, env=environment)
    consumer_binary = validation_tmp / "consumer"
    run_step(
        "install-consumer",
        (
            "c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I",
            str(install_dir / "include"), "tests/install/consumer.cpp", "-o", str(consumer_binary)
        ),
        cwd=root,
        evidence=evidence,
        env=environment,
    )
    run_step(
        "install-consumer-run",
        (str(consumer_binary),),
        cwd=root,
        evidence=evidence,
        env=environment,
    )

    if args.full:
        run_step("cmake-sanitize-configure", ("cmake", "--preset", "cpu-sanitize"), cwd=root,
                 evidence=evidence, env=environment)
        run_step("cmake-sanitize-build", ("cmake", "--build", "--preset", "cpu-sanitize"), cwd=root,
                 evidence=evidence, env=environment)
        run_step(
            "ctest-sanitize", ("ctest", "--preset", "cpu-sanitize"), cwd=root, evidence=evidence,
            env=environment
        )
        wheel_dir = validation_tmp / "wheel"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        wheel_command = (
            "import setuptools.build_meta as backend; "
            f"backend.build_wheel({str(wheel_dir)!r})"
        )
        run_step("python-wheel", (sys.executable, "-c", wheel_command), cwd=root, evidence=evidence,
                 env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
