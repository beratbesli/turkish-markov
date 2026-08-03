from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN = PROJECT_ROOT / "main.py"


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(MAIN), *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )

    def test_help_lists_both_commands(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("build-db", result.stdout)
        self.assertIn("generate", result.stdout)

    def test_cli_build_and_generate_keep_stdout_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.txt"
            database = root / "markov.db"
            corpus.write_text(
                "Merhaba dünya. Merhaba güzel dünya.\n",
                encoding="utf-8",
            )
            build = self.run_cli(
                "build-db",
                "--input-dir",
                str(corpus),
                "--db",
                str(database),
                "--order",
                "1",
                "--workers",
                "1",
                "--batch-size",
                "1000",
                "--quiet",
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            self.assertEqual(build.stdout, "")
            generated = self.run_cli(
                "generate",
                "--db",
                str(database),
                "--prompt",
                "MERHABA",
                "--length",
                "2",
                "--temperature",
                "0",
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertTrue(generated.stdout.startswith("merhaba "))
            self.assertEqual(generated.stderr, "")

    def test_invalid_temperature_is_usage_error(self) -> None:
        result = self.run_cli(
            "generate",
            "--prompt",
            "x",
            "--length",
            "1",
            "--temperature",
            "nan",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("finite number", result.stderr)


if __name__ == "__main__":
    unittest.main()
