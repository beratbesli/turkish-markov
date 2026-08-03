"""Streaming, multiprocessing corpus indexer.

Large files are divided into byte ranges aligned to paragraph boundaries.  Each
spawned worker owns a private SQLite shard, so CPU-heavy normalization/counting
scales without creating multiple writers contending for the final database.
"""

from __future__ import annotations

import atexit
import codecs
import multiprocessing
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import BinaryIO

from .cleaner import (
    clean_text,
    is_lexical_token,
    iter_character_words,
    iter_sentences,
    tokenize,
)
from .database import (
    BOS_TOKEN,
    EOS_TOKEN,
    MODEL_IDS,
    create_final_database,
    create_shard_connection,
    encode_state,
    finalize_database,
    merge_shard,
    upsert_shard_rows,
)
from .exceptions import ConfigurationError, CorpusError, DatabaseError

_PARAGRAPH_BOUNDARY_RE = re.compile(rb"\r?\n[ \t]*\r?\n")
_READ_BLOCK_BYTES = 1024 * 1024
_BOUNDARY_SCAN_BLOCK_BYTES = 64 * 1024
_MAX_BOUNDARY_SCAN_BYTES = 4 * 1024 * 1024

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Validated settings for a database build."""

    input_path: Path
    database_path: Path
    order: int = 2
    mode: str = "word"
    workers: int = 0
    chunk_size_bytes: int = 256 * 1024 * 1024
    batch_size: int = 100_000
    encoding_errors: str = "strict"
    max_paragraph_chars: int = 16 * 1024 * 1024
    overwrite: bool = False
    temp_dir: Path | None = None

    @property
    def modes(self) -> tuple[str, ...]:
        return ("word", "char") if self.mode == "both" else (self.mode,)

    @property
    def resolved_workers(self) -> int:
        if self.workers > 0:
            return self.workers
        cpu_count = os.cpu_count() or 2
        # Eight workers is a balanced default for simultaneous CPU and shard I/O.
        return max(1, min(8, cpu_count - 1))

    def validate(self) -> None:
        if self.order < 1:
            raise ConfigurationError("--order must be at least 1")
        if self.mode not in {"word", "char", "both"}:
            raise ConfigurationError("--mode must be word, char, or both")
        if self.workers < 0:
            raise ConfigurationError("--workers cannot be negative")
        if self.chunk_size_bytes < 64 * 1024:
            raise ConfigurationError("chunk size must be at least 64 KiB")
        if self.batch_size < 1_000:
            raise ConfigurationError("--batch-size must be at least 1000")
        if self.encoding_errors not in {"strict", "replace", "ignore"}:
            raise ConfigurationError(
                "--encoding-errors must be strict, replace, or ignore"
            )
        if self.max_paragraph_chars < 1024:
            raise ConfigurationError("maximum paragraph size is too small")


@dataclass(frozen=True, slots=True)
class FileChunk:
    path: str
    start: int
    end: int
    number: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class WorkerStats:
    bytes_processed: int = 0
    paragraphs: int = 0
    word_tokens: int = 0
    character_words: int = 0
    transitions: int = 0
    decode_replacements: int = 0

    def add(self, other: "WorkerStats") -> None:
        self.bytes_processed += other.bytes_processed
        self.paragraphs += other.paragraphs
        self.word_tokens += other.word_tokens
        self.character_words += other.character_words
        self.transitions += other.transitions
        self.decode_replacements += other.decode_replacements


@dataclass(frozen=True, slots=True)
class BuildReport:
    database_path: Path
    source_files: int
    source_bytes: int
    chunks: int
    workers: int
    modes: tuple[str, ...]
    order: int
    row_counts: dict[str, int]
    stats: WorkerStats


def discover_text_files(input_path: Path, *, exclude: Path | None = None) -> list[Path]:
    """Find UTF-8 ``.txt`` inputs recursively, or accept one file directly."""

    path = input_path.expanduser().resolve()
    excluded = exclude.expanduser().resolve() if exclude is not None else None
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            {
                candidate.resolve()
                for candidate in path.rglob("*.txt")
                if candidate.is_file()
            }
        )
    else:
        raise CorpusError(f"input path does not exist: {path}")
    files = [candidate for candidate in candidates if candidate != excluded]
    if not files:
        raise CorpusError(f"no .txt corpus files found under: {path}")
    return files


def _find_boundary(
    file_handle: BinaryIO,
    nominal_offset: int,
    file_size: int,
    max_scan_bytes: int,
) -> int:
    """Find a paragraph boundary, with newline/whitespace safety fallbacks."""

    if nominal_offset >= file_size:
        return file_size
    handle = file_handle
    handle.seek(nominal_offset)
    scanned = 0
    carry = b""
    first_newline_end: int | None = None
    while scanned < max_scan_bytes:
        block = handle.read(
            min(_BOUNDARY_SCAN_BLOCK_BYTES, max_scan_bytes - scanned)
        )
        if not block:
            return file_size
        block_start = handle.tell() - len(block)
        if first_newline_end is None:
            newline_at = block.find(b"\n")
            if newline_at >= 0:
                first_newline_end = block_start + newline_at + 1
        data = carry + block
        data_start = block_start - len(carry)
        match = _PARAGRAPH_BOUNDARY_RE.search(data)
        if match is not None:
            return min(file_size, data_start + match.end())
        # Enough overlap for ordinary blank lines containing indentation.
        carry = data[-4096:]
        scanned += len(block)
    if first_newline_end is not None:
        return first_newline_end

    # Bound planning time for pathological unwrapped/unspaced data. Move the
    # hard boundary forward by at most three bytes so neither adjacent decoder
    # starts or finishes inside a UTF-8 code point.
    hard_offset = min(file_size, nominal_offset + max_scan_bytes)
    if hard_offset >= file_size:
        return file_size
    handle.seek(hard_offset)
    probe = handle.read(4)
    for index, byte in enumerate(probe):
        if byte & 0b1100_0000 != 0b1000_0000:
            return hard_offset + index
    return min(file_size, hard_offset + len(probe))


def make_file_chunks(path: Path, chunk_size_bytes: int) -> list[FileChunk]:
    """Partition one file into deterministic paragraph-aligned ranges."""

    file_size = path.stat().st_size
    if file_size == 0:
        return []
    chunks: list[FileChunk] = []
    start = 0
    number = 0
    with path.open("rb") as handle:
        while start < file_size:
            nominal_end = min(file_size, start + chunk_size_bytes)
            end = (
                file_size
                if nominal_end == file_size
                else _find_boundary(
                    handle,
                    nominal_end,
                    file_size,
                    min(chunk_size_bytes, _MAX_BOUNDARY_SCAN_BYTES),
                )
            )
            if end <= start:
                raise CorpusError(f"could not advance chunk boundary in {path}")
            chunks.append(FileChunk(str(path), start, end, number))
            number += 1
            start = end
    return chunks


class _FrequencyAccumulator:
    def __init__(self, connection: sqlite3.Connection, batch_size: int) -> None:
        self.connection = connection
        self.batch_size = batch_size
        self.counts: Counter[tuple[int, str, str, str]] = Counter()

    def add(self, model_id: int, context: Sequence[str], next_token: str) -> None:
        state = encode_state(context)
        self.counts[(model_id, state, next_token, context[-1])] += 1
        if len(self.counts) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.counts:
            return
        rows = (
            (model_id, state, next_token, last_token, frequency)
            for (model_id, state, next_token, last_token), frequency in self.counts.items()
        )
        upsert_shard_rows(self.connection, rows)
        self.counts.clear()


_worker_connection: sqlite3.Connection | None = None
_worker_order = 0
_worker_modes: tuple[str, ...] = ()
_worker_batch_size = 0
_worker_encoding_errors = "strict"
_worker_max_paragraph_chars = 0


def _close_worker_connection() -> None:
    global _worker_connection
    if _worker_connection is not None:
        _worker_connection.close()
        _worker_connection = None


def _initialize_worker(
    shard_directory: str,
    order: int,
    modes: tuple[str, ...],
    batch_size: int,
    encoding_errors: str,
    max_paragraph_chars: int,
) -> None:
    """Create one SQLite shard per spawned process (module-level for Windows)."""

    global _worker_connection, _worker_order, _worker_modes
    global _worker_batch_size, _worker_encoding_errors, _worker_max_paragraph_chars
    shard_path = Path(shard_directory) / f"worker-{os.getpid()}.sqlite"
    _worker_connection = create_shard_connection(shard_path)
    _worker_order = order
    _worker_modes = modes
    _worker_batch_size = batch_size
    _worker_encoding_errors = encoding_errors
    _worker_max_paragraph_chars = max_paragraph_chars
    atexit.register(_close_worker_connection)


def _iter_decoded_lines(
    chunk: FileChunk, replacement_counter: list[int]
) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors=_worker_encoding_errors)
    path = Path(chunk.path)
    remaining = chunk.size
    text_buffer = ""
    byte_position = chunk.start
    with path.open("rb") as handle:
        handle.seek(chunk.start)
        while remaining:
            raw = handle.read(min(_READ_BLOCK_BYTES, remaining))
            if not raw:
                raise CorpusError(
                    f"unexpected end of file in {path} at byte {byte_position}; "
                    "the source may have changed during indexing"
                )
            remaining -= len(raw)
            try:
                decoded = decoder.decode(raw, final=False)
            except UnicodeDecodeError as exc:
                error_offset = byte_position + exc.start
                raise CorpusError(
                    f"invalid UTF-8 in {path} near byte {error_offset}; "
                    "use --encoding-errors replace to continue lossy indexing"
                ) from exc
            byte_position += len(raw)
            replacement_counter[0] += decoded.count("\ufffd")
            lines = (text_buffer + decoded).split("\n")
            text_buffer = lines.pop()
            for line in lines:
                yield line.removesuffix("\r")
            # A newline-free record must not bypass the paragraph RAM cap. The
            # forced fragment becomes a model reset later, which is preferable
            # to retaining an arbitrarily large malformed record in memory.
            while len(text_buffer) > _worker_max_paragraph_chars:
                split_at = text_buffer.rfind(
                    " ", 0, _worker_max_paragraph_chars + 1
                )
                if split_at < _worker_max_paragraph_chars // 2:
                    split_at = _worker_max_paragraph_chars
                yield text_buffer[:split_at]
                text_buffer = text_buffer[split_at:].lstrip()
        try:
            tail = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise CorpusError(
                f"incomplete UTF-8 sequence at the end of chunk {chunk.number} in {path}"
            ) from exc
        replacement_counter[0] += tail.count("\ufffd")
        text_buffer += tail
        if text_buffer:
            yield text_buffer.removesuffix("\r")


def _iter_paragraphs(chunk: FileChunk, stats: WorkerStats) -> Iterator[str]:
    parts: list[str] = []
    character_count = 0
    replacements = [0]
    for line in _iter_decoded_lines(chunk, replacements):
        stripped = line.strip()
        if not stripped:
            if parts:
                yield " ".join(parts)
                parts.clear()
                character_count = 0
            continue
        parts.append(stripped)
        character_count += len(stripped) + 1
        # This safety valve bounds RAM even for malformed corpora with no blank
        # lines.  Normal book paragraphs never approach this threshold.
        if character_count >= _worker_max_paragraph_chars:
            yield " ".join(parts)
            parts.clear()
            character_count = 0
    if parts:
        yield " ".join(parts)
    stats.decode_replacements += replacements[0]


def _add_sequence(
    accumulator: _FrequencyAccumulator,
    model_id: int,
    tokens: Sequence[str],
    stats: WorkerStats,
) -> None:
    if not tokens:
        return
    context: deque[str] = deque([BOS_TOKEN] * _worker_order, maxlen=_worker_order)
    for next_token in chain(tokens, (EOS_TOKEN,)):
        accumulator.add(model_id, tuple(context), next_token)
        context.append(next_token)
        stats.transitions += 1


def _process_chunk(chunk: FileChunk) -> WorkerStats:
    if _worker_connection is None:
        raise RuntimeError("worker database was not initialized")
    accumulator = _FrequencyAccumulator(_worker_connection, _worker_batch_size)
    stats = WorkerStats(bytes_processed=chunk.size)
    for paragraph in _iter_paragraphs(chunk, stats):
        cleaned = clean_text(paragraph)
        if not cleaned:
            continue
        stats.paragraphs += 1
        tokens = tokenize(cleaned, already_clean=True)
        if "word" in _worker_modes:
            stats.word_tokens += sum(is_lexical_token(token) for token in tokens)
            for sentence in iter_sentences(tokens):
                _add_sequence(
                    accumulator, MODEL_IDS["word"], sentence, stats
                )
        if "char" in _worker_modes:
            for word in iter_character_words(tokens):
                stats.character_words += 1
                _add_sequence(
                    accumulator, MODEL_IDS["char"], list(word), stats
                )
    accumulator.flush()
    return stats


def _cleanup_staging_files(staging_path: Path) -> None:
    for path in (
        staging_path,
        Path(f"{staging_path}-wal"),
        Path(f"{staging_path}-shm"),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def build_database(
    config: BuildConfig,
    *,
    progress: ProgressCallback | None = None,
) -> BuildReport:
    """Stream and index a corpus, then atomically publish the final database."""

    config.validate()
    database_path = config.database_path.expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists() and not config.overwrite:
        raise DatabaseError(
            f"database already exists: {database_path} (pass --overwrite to replace it)"
        )

    files = discover_text_files(config.input_path, exclude=database_path)
    source_signatures = [
        (path, stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
        for path in files
        for stat in (path.stat(),)
    ]
    source_records = [
        (path, size, modified_ns)
        for path, size, modified_ns, _, _ in source_signatures
    ]
    source_bytes = sum(record[1] for record in source_records)
    chunks = [
        chunk
        for path, _, _ in source_records
        for chunk in make_file_chunks(path, config.chunk_size_bytes)
    ]
    if not chunks:
        raise CorpusError("all discovered corpus files are empty")

    workers = min(config.resolved_workers, len(chunks))
    temp_parent = (
        config.temp_dir.expanduser().resolve()
        if config.temp_dir
        else database_path.parent
    )
    temp_parent.mkdir(parents=True, exist_ok=True)
    staging_path = database_path.with_name(
        f".{database_path.name}.{uuid.uuid4().hex}.building"
    )
    total_stats = WorkerStats()
    row_counts: dict[str, int] = {}

    try:
        with tempfile.TemporaryDirectory(
            prefix="turkish-markov-shards-", dir=temp_parent
        ) as shard_directory:
            if progress:
                progress("index", 0, source_bytes, f"indexing with {workers} workers")
            spawn_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=spawn_context,
                initializer=_initialize_worker,
                initargs=(
                    shard_directory,
                    config.order,
                    config.modes,
                    config.batch_size,
                    config.encoding_errors,
                    config.max_paragraph_chars,
                ),
            ) as executor:
                future_chunks = {
                    executor.submit(_process_chunk, chunk): chunk for chunk in chunks
                }
                completed_bytes = 0
                try:
                    for future in as_completed(future_chunks):
                        chunk = future_chunks[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            for pending in future_chunks:
                                pending.cancel()
                            raise CorpusError(
                                f"failed while indexing {chunk.path} "
                                f"bytes {chunk.start}..{chunk.end}: {exc}"
                            ) from exc
                        total_stats.add(result)
                        completed_bytes += chunk.size
                        if progress:
                            progress(
                                "index",
                                completed_bytes,
                                source_bytes,
                                Path(chunk.path).name,
                            )
                except KeyboardInterrupt:
                    for pending in future_chunks:
                        pending.cancel()
                    raise

            shard_paths = sorted(Path(shard_directory).glob("worker-*.sqlite"))
            if not shard_paths:
                raise DatabaseError("indexing workers produced no SQLite shards")
            connection = create_final_database(
                staging_path,
                order=config.order,
                modes=config.modes,
                sources=source_records,
                encoding_errors=config.encoding_errors,
            )
            try:
                for index, shard_path in enumerate(shard_paths, start=1):
                    merge_shard(connection, shard_path)
                    if progress:
                        progress("merge", index, len(shard_paths), shard_path.name)
                row_counts = finalize_database(connection)
            finally:
                connection.close()

            # Closing the final connection after a successful TRUNCATE checkpoint
            # normally removes these. Explicit cleanup avoids orphaned staging
            # sidecars on SQLite/Windows combinations that retain empty files.
            for sidecar in (Path(f"{staging_path}-wal"), Path(f"{staging_path}-shm")):
                try:
                    sidecar.unlink()
                except FileNotFoundError:
                    pass

        empty_modes = [mode for mode in config.modes if row_counts.get(mode, 0) == 0]
        if empty_modes:
            raise CorpusError(
                "corpus produced no transitions for model(s): " + ", ".join(empty_modes)
            )
        for path, size, modified_ns, device, inode in source_signatures:
            try:
                current = path.stat()
            except OSError as exc:
                raise CorpusError(f"source disappeared during build: {path}") from exc
            if (
                current.st_size != size
                or current.st_mtime_ns != modified_ns
                or current.st_dev != device
                or current.st_ino != inode
            ):
                raise CorpusError(
                    f"source changed during build; refusing to publish mixed counts: {path}"
                )
        if database_path.exists() and not config.overwrite:
            raise DatabaseError(f"database appeared during build: {database_path}")
        os.replace(staging_path, database_path)
        if progress:
            progress("done", source_bytes, source_bytes, str(database_path))
    except (OSError, sqlite3.Error) as exc:
        _cleanup_staging_files(staging_path)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            raise DatabaseError(
                f"disk is full while building {database_path}; choose a larger --temp-dir"
            ) from exc
        raise DatabaseError(f"database build failed: {exc}") from exc
    except Exception:
        _cleanup_staging_files(staging_path)
        raise

    return BuildReport(
        database_path=database_path,
        source_files=len(files),
        source_bytes=source_bytes,
        chunks=len(chunks),
        workers=workers,
        modes=config.modes,
        order=config.order,
        row_counts=row_counts,
        stats=total_stats,
    )


def estimated_free_space_warning(config: BuildConfig) -> str | None:
    """Return a conservative disk-space warning without blocking a build."""

    try:
        files = discover_text_files(config.input_path, exclude=config.database_path)
        source_bytes = sum(path.stat().st_size for path in files)
        target_free = shutil.disk_usage(config.database_path.resolve().parent).free
        temp_target = config.temp_dir or config.database_path.resolve().parent
        temp_free = shutil.disk_usage(temp_target).free
    except OSError:
        return None
    recommended = source_bytes * (3 if config.mode == "both" else 2)
    if min(target_free, temp_free) < recommended:
        gib = recommended / (1024**3)
        return (
            f"available disk space is below the conservative {gib:.1f} GiB "
            "estimate; high-order word models can exceed the source size"
        )
    return None
