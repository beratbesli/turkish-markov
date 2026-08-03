from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from turkish_markov.database import BOS_TOKEN, MarkovDatabase, encode_state
from turkish_markov.exceptions import CorpusError, DatabaseError, GenerationError
from turkish_markov.indexer import BuildConfig, build_database, make_file_chunks
from turkish_markov.markov import MarkovGenerator


class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "Türkçe kitap.txt"
        self.corpus.write_text(
            "IĞDIR güzel bir şehirdir.\n"
            "Bu satır önceki satırın sert sarılmış devamıdır.\n\n"
            "İZMİR güzeldir! Ankara’da hayat vardır.\n\n"
            "IĞDIR soğuktur. KÂĞIT kalem ile yazılır.\n",
            encoding="utf-8",
        )
        self.database_path = self.root / "model.sqlite"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self) -> None:
        report = build_database(
            BuildConfig(
                input_path=self.corpus,
                database_path=self.database_path,
                order=2,
                mode="both",
                workers=1,
                chunk_size_bytes=1024 * 1024,
                batch_size=1_000,
            )
        )
        self.assertGreater(report.row_counts["word"], 0)
        self.assertGreater(report.row_counts["char"], 0)

    def test_build_query_generate_and_integrity(self) -> None:
        self._build()
        connection = sqlite3.connect(self.database_path)
        try:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            frequency = connection.execute(
                """
                SELECT frequency
                FROM transitions
                WHERE model_id = 1 AND state = ? AND next_token = 'ığdır'
                """,
                (encode_state([BOS_TOKEN, BOS_TOKEN]),),
            ).fetchone()
            self.assertEqual(frequency, (2,))
        finally:
            connection.close()

        with MarkovDatabase(self.database_path) as database:
            generator = MarkovGenerator(database, seed=7)
            text = generator.generate(
                "word", "IĞDIR", length=3, temperature=0
            )
            self.assertTrue(text.startswith("ığdır"))
            self.assertNotIn(BOS_TOKEN, text)
            pseudo_word = generator.generate(
                "char", "I", length=12, temperature=0.8
            )
            self.assertTrue(pseudo_word.startswith("ı"))
            self.assertNotIn(BOS_TOKEN, pseudo_word)

    def test_existing_database_requires_overwrite(self) -> None:
        self._build()
        with self.assertRaises(DatabaseError):
            self._build()

    def test_unknown_prompt_is_an_actionable_error(self) -> None:
        self._build()
        with MarkovDatabase(self.database_path) as database:
            generator = MarkovGenerator(database, seed=1)
            with self.assertRaises(GenerationError):
                generator.generate("word", "qwx", length=1, temperature=1.0)

    def test_invalid_utf8_is_strict_by_default_and_replace_is_counted(self) -> None:
        malformed = self.root / "malformed.txt"
        malformed.write_bytes(b"iyi bir s\xc3\xb6z.\n\nbozuk \xff metin.\n")
        strict_database = self.root / "strict.sqlite"
        with self.assertRaises(CorpusError):
            build_database(
                BuildConfig(
                    input_path=malformed,
                    database_path=strict_database,
                    workers=1,
                    batch_size=1_000,
                )
            )
        self.assertFalse(strict_database.exists())

        replace_database = self.root / "replace.sqlite"
        report = build_database(
            BuildConfig(
                input_path=malformed,
                database_path=replace_database,
                workers=1,
                batch_size=1_000,
                encoding_errors="replace",
            )
        )
        self.assertEqual(report.stats.decode_replacements, 1)

    def test_chunk_boundaries_align_after_blank_lines(self) -> None:
        path = self.root / "chunks.txt"
        path.write_text(
            "birinci paragraf satır bir\n"
            "satır iki\n\n"
            "ikinci paragraf\n\n"
            "üçüncü paragraf\n",
            encoding="utf-8",
        )
        chunks = make_file_chunks(path, 20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, path.stat().st_size)
        self.assertTrue(all(left.end == right.start for left, right in zip(chunks, chunks[1:])))

    def test_unbroken_file_uses_utf8_safe_hard_boundaries(self) -> None:
        path = self.root / "unbroken.txt"
        path.write_text("ağ" * 120_000, encoding="utf-8")
        chunks = make_file_chunks(path, 64 * 1024)
        raw = path.read_bytes()
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(chunk.size for chunk in chunks), len(raw))
        for chunk in chunks:
            # Both sides of every forced boundary are valid UTF-8 streams.
            raw[chunk.start : chunk.end].decode("utf-8", errors="strict")

    def test_multi_chunk_counts_match_single_chunk_counts(self) -> None:
        large_corpus = self.root / "large.txt"
        paragraph = "IĞDIR güzeldir. İZMİR de güzeldir.\n\n"
        # Slightly above a 64 KiB test chunk, with abundant paragraph
        # boundaries so two processes can own exact independent ranges.
        repetitions = (64 * 1024 // len(paragraph.encode("utf-8"))) + 200
        large_corpus.write_text(paragraph * repetitions, encoding="utf-8")
        multi_db = self.root / "multi.sqlite"
        single_db = self.root / "single.sqlite"
        build_database(
            BuildConfig(
                input_path=large_corpus,
                database_path=multi_db,
                order=2,
                mode="word",
                workers=2,
                chunk_size_bytes=64 * 1024,
                batch_size=1_000,
            )
        )
        build_database(
            BuildConfig(
                input_path=large_corpus,
                database_path=single_db,
                order=2,
                mode="word",
                workers=1,
                chunk_size_bytes=256 * 1024,
                batch_size=1_000,
            )
        )

        def logical_rows(path: Path) -> list[tuple[object, ...]]:
            connection = sqlite3.connect(path)
            try:
                return connection.execute(
                    """
                    SELECT model_id, state, next_token, frequency
                    FROM transitions
                    ORDER BY model_id, state, next_token
                    """
                ).fetchall()
            finally:
                connection.close()

        self.assertEqual(logical_rows(multi_db), logical_rows(single_db))


if __name__ == "__main__":
    unittest.main()
