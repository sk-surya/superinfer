"""Stable command-line boundary for local validation and future converter commands."""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum
from pathlib import Path
from typing import Sequence

from superinfer import __version__


class ExitCode(IntEnum):
    """Process exit categories used by automation and CI."""

    OK = 0
    INVALID_ARGUMENT = 2
    INPUT_ERROR = 3
    INTERNAL_ERROR = 4


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="superinfer", description="SuperInfer developer tools")
    parser.add_argument("--version", action="store_true", help="print the package version")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate", help="validate a local artifact path")
    validate.add_argument("artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> ExitCode:
    """Run the CLI and return a stable category; callers own process termination."""

    try:
        args = _parser().parse_args(argv)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return ExitCode.INVALID_ARGUMENT

    if args.version:
        print(__version__)
        return ExitCode.OK
    if args.command is None:
        _parser().print_help()
        return ExitCode.OK
    if args.command == "validate":
        if not args.artifact.is_file():
            print(f"input error: artifact does not exist: {args.artifact}", file=sys.stderr)
            return ExitCode.INPUT_ERROR
        print(f"valid: {args.artifact}")
        return ExitCode.OK
    return ExitCode.INTERNAL_ERROR
