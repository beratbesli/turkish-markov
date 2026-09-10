# Contributing

Thank you for helping improve Turkish Markov.

1. Open an issue for changes that alter the database schema, tokenization, or corpus format.
2. Create a focused branch and include tests for behavior changes.
3. Run `python -m unittest discover -v` and `python -m build` before opening a pull request.
4. Do not commit corpora, generated databases, credentials, or data of uncertain origin.
5. For dataset changes, complete the provenance manifest and document the right to use and
   redistribute every source. A license label without primary evidence is not sufficient.

Pull requests should explain compatibility impact and include a reproducible example when
generation output or performance changes.
