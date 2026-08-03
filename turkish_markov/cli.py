"""Argparse command-line interface for building and sampling models."""

from __future__ import annotations

import argparse
import io
import math
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import TextIO

from . import __version__
from .database import MarkovDatabase
from .exceptions import ConfigurationError, MarkovError
from .indexer import BuildConfig, BuildReport, build_database, estimated_free_space_warning
from .markov import MarkovGenerator


def _configure_utf8_stdio() -> None:
    """Make Turkish CLI input/output deterministic on legacy Windows locales."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _temperature(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return parsed


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


class _ProgressPrinter:
    def __init__(self, stream: TextIO = sys.stderr) -> None:
        self.stream = stream
        self.is_terminal = bool(getattr(stream, "isatty", lambda: False)())
        self.last_phase = ""
        self.last_bucket = -1

    def __call__(
        self, phase: str, current: int, total: int, message: str
    ) -> None:
        ratio = 1.0 if total <= 0 else min(1.0, current / total)
        if self.is_terminal:
            width = 28
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            if phase == "index":
                count = f"{_human_bytes(current)}/{_human_bytes(total)}"
            else:
                count = f"{current}/{total}"
            self.stream.write(
                f"\r{phase:>6} [{bar}] {ratio:6.1%} {count} {message[:32]:32}"
            )
            self.stream.flush()
            if phase == "done":
                self.stream.write("\n")
            return

        bucket = int(ratio * 10)
        if phase != self.last_phase or bucket != self.last_bucket or phase == "done":
            self.stream.write(f"{phase}: {ratio:.0%} {message}\n")
            self.stream.flush()
            self.last_phase = phase
            self.last_bucket = bucket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turkish-markov",
        description="Build and sample disk-backed Turkish Markov models.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--debug", action="store_true", help="show a traceback for operational errors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-db", help="stream a UTF-8 corpus into a SQLite Markov database"
    )
    build.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="directory containing .txt files (a single .txt file also works)",
    )
    build.add_argument(
        "--db", type=Path, default=Path("markov.db"), help="output database path"
    )
    build.add_argument(
        "--order",
        type=_positive_int,
        default=2,
        help="number of preceding units in each state (default: 2)",
    )
    build.add_argument(
        "--mode",
        "--level",
        choices=("word", "char", "both"),
        default="word",
        help="model(s) to build (default: word)",
    )
    build.add_argument(
        "--workers",
        type=_nonnegative_int,
        default=0,
        help="worker processes; 0 chooses automatically (default: 0)",
    )
    build.add_argument(
        "--chunk-size-mb",
        type=_positive_int,
        default=256,
        help="target byte-range size before paragraph alignment (default: 256)",
    )
    build.add_argument(
        "--batch-size",
        type=_positive_int,
        default=100_000,
        help="unique transitions retained per worker batch (default: 100000)",
    )
    build.add_argument(
        "--encoding-errors",
        choices=("strict", "replace", "ignore"),
        default="strict",
        help="invalid UTF-8 policy (default: strict)",
    )
    build.add_argument(
        "--max-paragraph-mb",
        type=_positive_int,
        default=16,
        help="RAM safety cap for a paragraph record (default: 16)",
    )
    build.add_argument(
        "--temp-dir",
        type=Path,
        help="directory for process-local SQLite shards",
    )
    build.add_argument(
        "--overwrite", action="store_true", help="atomically replace an existing database"
    )
    build.add_argument(
        "--quiet", action="store_true", help="suppress progress and the build summary"
    )

    generate = subparsers.add_parser(
        "generate", help="generate text or a pseudo-word from a completed database"
    )
    generate.add_argument(
        "--db", type=Path, default=Path("markov.db"), help="input database path"
    )
    generate.add_argument(
        "--mode",
        "--level",
        choices=("word", "char"),
        default="word",
        help="generation model (default: word)",
    )
    generate.add_argument(
        "--prompt",
        required=True,
        help="starting text, or a character prefix in char mode (may be empty)",
    )
    generate.add_argument(
        "--length",
        required=True,
        type=_positive_int,
        help="new words (word mode) or maximum new characters (char mode)",
    )
    generate.add_argument(
        "--temperature",
        type=_temperature,
        default=1.0,
        help="0 is greedy; larger values increase variety (default: 1.0)",
    )
    generate.add_argument("--seed", type=int, help="repeatable random seed")
    return parser


def _print_report(report: BuildReport, stream: TextIO = sys.stderr) -> None:
    row_summary = ", ".join(
        f"{mode}={count:,}" for mode, count in report.row_counts.items()
    )
    stream.write(
        "built "
        f"{report.database_path} from {report.source_files} file(s), "
        f"{_human_bytes(report.source_bytes)}, {report.chunks} chunk(s), "
        f"order {report.order}; transition rows: {row_summary}\n"
    )
    stream.write(
        f"processed {report.stats.paragraphs:,} paragraphs and "
        f"{report.stats.transitions:,} transition observations"
    )
    if report.stats.decode_replacements:
        stream.write(
            f"; replaced {report.stats.decode_replacements:,} invalid UTF-8 sequence(s)"
        )
    stream.write("\n")


def _run_build(arguments: argparse.Namespace) -> int:
    config = BuildConfig(
        input_path=arguments.input_dir,
        database_path=arguments.db,
        order=arguments.order,
        mode=arguments.mode,
        workers=arguments.workers,
        chunk_size_bytes=arguments.chunk_size_mb * 1024 * 1024,
        batch_size=arguments.batch_size,
        encoding_errors=arguments.encoding_errors,
        max_paragraph_chars=arguments.max_paragraph_mb * 1024 * 1024,
        overwrite=arguments.overwrite,
        temp_dir=arguments.temp_dir,
    )
    config.validate()
    if not arguments.quiet:
        warning = estimated_free_space_warning(config)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
    report = build_database(
        config,
        progress=None if arguments.quiet else _ProgressPrinter(),
    )
    if not arguments.quiet:
        _print_report(report)
    return 0


def _run_generate(arguments: argparse.Namespace) -> int:
    with MarkovDatabase(arguments.db) as database:
        generator = MarkovGenerator(database, seed=arguments.seed)
        generated = generator.generate(
            arguments.mode,
            arguments.prompt,
            length=arguments.length,
            temperature=arguments.temperature,
        )
    print(generated)
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build-db":
            return _run_build(arguments)
        if arguments.command == "generate":
            return _run_generate(arguments)
        raise ConfigurationError(f"unknown command: {arguments.command}")
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            return 0
    except (MarkovError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if arguments.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
