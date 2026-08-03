"""SQLite schema, bulk merging, and read-side transition queries."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from .cleaner import CLEANER_VERSION
from .exceptions import DatabaseError

SCHEMA_VERSION: Final[str] = "1"
STATE_SEPARATOR: Final[str] = "\x1f"
BOS_TOKEN: Final[str] = "<|BOS|>"
EOS_TOKEN: Final[str] = "<|EOS|>"
MODEL_IDS: Final[dict[str, int]] = {"word": 1, "char": 2}


def encode_state(tokens: Sequence[str]) -> str:
    """Encode a state tuple compactly and reversibly as SQLite TEXT."""

    if not tokens:
        raise ValueError("a Markov state cannot be empty")
    if any(STATE_SEPARATOR in token for token in tokens):
        raise ValueError("a token contains the reserved state separator")
    return STATE_SEPARATOR.join(tokens)


def decode_state(state: str) -> tuple[str, ...]:
    """Decode a state produced by :func:`encode_state`."""

    return tuple(state.split(STATE_SEPARATOR))


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Stored configuration for one model in a database."""

    model_id: int
    mode: str
    order: int
    transition_rows: int


_SHARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS transitions (
    model_id INTEGER NOT NULL,
    state TEXT NOT NULL,
    next_token TEXT NOT NULL,
    last_token TEXT NOT NULL,
    frequency INTEGER NOT NULL CHECK (frequency > 0),
    PRIMARY KEY (model_id, state, next_token)
) WITHOUT ROWID;
"""

_FINAL_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL UNIQUE CHECK (mode IN ('word', 'char')),
    state_size INTEGER NOT NULL CHECK (state_size >= 1),
    cleaner_version TEXT NOT NULL,
    complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0, 1)),
    transition_rows INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;

CREATE TABLE sources (
    path TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL
) WITHOUT ROWID;

CREATE TABLE transitions (
    model_id INTEGER NOT NULL REFERENCES models(id),
    state TEXT NOT NULL,
    next_token TEXT NOT NULL,
    frequency INTEGER NOT NULL CHECK (frequency > 0),
    PRIMARY KEY (model_id, state, next_token)
) WITHOUT ROWID;

CREATE TABLE backoff_transitions (
    model_id INTEGER NOT NULL REFERENCES models(id),
    last_token TEXT NOT NULL,
    next_token TEXT NOT NULL,
    frequency INTEGER NOT NULL CHECK (frequency > 0),
    PRIMARY KEY (model_id, last_token, next_token)
) WITHOUT ROWID;
"""


def create_shard_connection(path: Path, *, cache_mib: int = 32) -> sqlite3.Connection:
    """Open a disposable worker-owned SQLite shard optimized for bulk UPSERTs."""

    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.executescript(_SHARD_SCHEMA)
    return connection


def upsert_shard_rows(
    connection: sqlite3.Connection,
    rows: Iterable[tuple[int, str, str, str, int]],
) -> None:
    """Merge one bounded in-memory frequency batch into a worker shard."""

    sql = """
        INSERT INTO transitions
            (model_id, state, next_token, last_token, frequency)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(model_id, state, next_token) DO UPDATE SET
            frequency = frequency + excluded.frequency
    """
    with connection:
        connection.executemany(sql, rows)


def _set_metadata(connection: sqlite3.Connection, key: str, value: object) -> None:
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, serialized),
    )


def create_final_database(
    path: Path,
    *,
    order: int,
    modes: Sequence[str],
    sources: Sequence[tuple[Path, int, int]],
    encoding_errors: str,
) -> sqlite3.Connection:
    """Create a new WAL-enabled final database at a staging path."""

    connection = sqlite3.connect(path, timeout=60.0)
    try:
        connection.execute("PRAGMA page_size=8192")
        journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(journal_mode).lower() != "wal":
            raise DatabaseError(
                f"SQLite could not enable WAL mode for staging database: {path}"
            )
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("PRAGMA wal_autocheckpoint=10000")
        connection.executescript(_FINAL_SCHEMA)
        now = datetime.now(timezone.utc).isoformat()
        with connection:
            _set_metadata(connection, "schema_version", SCHEMA_VERSION)
            _set_metadata(connection, "build_complete", "0")
            _set_metadata(connection, "created_utc", now)
            _set_metadata(connection, "cleaner_version", CLEANER_VERSION)
            _set_metadata(connection, "order", str(order))
            _set_metadata(connection, "modes", list(modes))
            _set_metadata(connection, "encoding", "utf-8")
            _set_metadata(connection, "encoding_errors", encoding_errors)
            _set_metadata(connection, "source_files", len(sources))
            _set_metadata(connection, "source_bytes", sum(item[1] for item in sources))
            for mode in modes:
                connection.execute(
                    """
                    INSERT INTO models(id, mode, state_size, cleaner_version)
                    VALUES (?, ?, ?, ?)
                    """,
                    (MODEL_IDS[mode], mode, order, CLEANER_VERSION),
                )
            connection.executemany(
                "INSERT INTO sources(path, size_bytes, modified_ns) VALUES (?, ?, ?)",
                ((str(path.resolve()), size, modified_ns) for path, size, modified_ns in sources),
            )
        return connection
    except Exception:
        connection.close()
        raise


def merge_shard(connection: sqlite3.Connection, shard_path: Path) -> None:
    """Merge one process-local shard into the final database atomically."""

    connection.execute("ATTACH DATABASE ? AS worker_shard", (str(shard_path),))
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO transitions(model_id, state, next_token, frequency)
            SELECT model_id, state, next_token, frequency
            FROM worker_shard.transitions
            WHERE 1
            ON CONFLICT(model_id, state, next_token) DO UPDATE SET
                frequency = frequency + excluded.frequency
            """
        )
        connection.execute(
            """
            INSERT INTO backoff_transitions(
                model_id, last_token, next_token, frequency
            )
            SELECT model_id, last_token, next_token, SUM(frequency)
            FROM worker_shard.transitions
            GROUP BY model_id, last_token, next_token
            HAVING SUM(frequency) > 0
            ON CONFLICT(model_id, last_token, next_token) DO UPDATE SET
                frequency = frequency + excluded.frequency
            """
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("DETACH DATABASE worker_shard")


def finalize_database(connection: sqlite3.Connection) -> dict[str, int]:
    """Create read indexes, mark all models complete, and checkpoint WAL data."""

    row_counts: dict[str, int] = {}
    with connection:
        rows = connection.execute("SELECT id, mode FROM models ORDER BY id").fetchall()
        for model_id, mode in rows:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM transitions WHERE model_id = ?", (model_id,)
                ).fetchone()[0]
            )
            row_counts[str(mode)] = count
            connection.execute(
                """
                UPDATE models
                SET complete = 1, transition_rows = ?
                WHERE id = ?
                """,
                (count, model_id),
            )
        _set_metadata(connection, "transition_rows", row_counts)
        _set_metadata(connection, "build_complete", "1")
    connection.execute("ANALYZE")
    connection.execute("PRAGMA optimize")
    checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is not None and int(checkpoint[0]) != 0:
        raise DatabaseError("could not checkpoint the final WAL before publication")
    return row_counts


class MarkovDatabase:
    """Read-only transition store used by text generators."""

    def __init__(self, path: str | Path, *, check_same_thread: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise DatabaseError(f"database does not exist: {self.path}")
        # Completed builds are checkpointed before publication. Immutable mode
        # lets SQLite read them from directories where creating -wal/-shm files
        # is not permitted.
        uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=30.0,
                check_same_thread=check_same_thread,
            )
            self._connection = connection
            self._connection.execute("PRAGMA query_only=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute("PRAGMA cache_size=-262144")
            self._connection.execute("PRAGMA mmap_size=1073741824")
            self._validate()
        except DatabaseError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise DatabaseError(
                f"cannot open Markov database {self.path}: {exc}"
            ) from exc

    def _validate(self) -> None:
        try:
            metadata = dict(self._connection.execute("SELECT key, value FROM metadata"))
        except sqlite3.Error as exc:
            raise DatabaseError(f"not a compatible Markov database: {self.path}") from exc
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise DatabaseError(
                "unsupported database schema version "
                f"{metadata.get('schema_version', '<missing>')}; expected {SCHEMA_VERSION}"
            )
        if metadata.get("build_complete") != "1":
            raise DatabaseError("database build is incomplete")
        if metadata.get("cleaner_version") != CLEANER_VERSION:
            raise DatabaseError(
                "database uses an incompatible cleaner version; rebuild the database"
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "MarkovDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def model(self, mode: str) -> ModelInfo:
        row = self._connection.execute(
            """
            SELECT id, mode, state_size, transition_rows
            FROM models
            WHERE mode = ? AND complete = 1
            """,
            (mode,),
        ).fetchone()
        if row is None:
            raise DatabaseError(
                f"database has no complete {mode!r} model; rebuild with --mode {mode} or --mode both"
            )
        return ModelInfo(int(row[0]), str(row[1]), int(row[2]), int(row[3]))

    def exact_transitions(
        self, model_id: int, state_tokens: Sequence[str]
    ) -> list[tuple[str, int]]:
        state = encode_state(state_tokens)
        return [
            (str(token), int(frequency))
            for token, frequency in self._connection.execute(
                """
                SELECT next_token, frequency
                FROM transitions
                WHERE model_id = ? AND state = ?
                ORDER BY next_token
                """,
                (model_id, state),
            )
        ]

    def backoff_transitions(
        self, model_id: int, last_token: str
    ) -> list[tuple[str, int]]:
        """Return materialized first-order candidates for a missing full state."""

        return [
            (str(token), int(frequency))
            for token, frequency in self._connection.execute(
                """
                SELECT next_token, frequency
                FROM backoff_transitions
                WHERE model_id = ? AND last_token = ?
                ORDER BY next_token
                """,
                (model_id, last_token),
            )
        ]

    def transition_frequency(
        self, model_id: int, state_tokens: Sequence[str], next_token: str
    ) -> int:
        """Return one exact transition count without loading all state choices."""

        state = encode_state(state_tokens)
        row = self._connection.execute(
            """
            SELECT frequency
            FROM transitions
            WHERE model_id = ? AND state = ? AND next_token = ?
            """,
            (model_id, state, next_token),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def metadata(self) -> dict[str, str]:
        return dict(self._connection.execute("SELECT key, value FROM metadata"))
