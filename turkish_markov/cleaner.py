"""Turkish-aware normalization and tokenization.

The same functions are used while indexing and while interpreting prompts.  That
symmetry is important: even a small normalization mismatch makes valid prompts
look absent from the model.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence

CLEANER_VERSION = "tr-cleaner-v1"

# Turkish plus the circumflexed letters still found in contemporary corpora.
TURKISH_CHARACTER_ALPHABET = frozenset(
    "abcçdefgğhıijklmnoöprsştuüvyzâîû"
)

_TURKISH_LOWER_TRANSLATION = str.maketrans({"I": "ı", "İ": "i"})
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u02bc": "'",
        "\uff07": "'",
        "`": "'",
        "´": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\ufeff": None,
        "\u00ad": None,
    }
)

# Number comes first so a decimal is not split into three tokens.  The broad
# Unicode lexeme branch deliberately keeps mixed OCR tokens intact; character
# training can then reject the whole malformed token instead of accepting a
# plausible-looking substring from it.
_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)*"
    r"|[^\W_]+(?:['-][^\W_]+)*"
    r"|\.{3,}"
    r"|[….!?,;:()%\[\]{}\"/-]",
    flags=re.UNICODE,
)
_MULTI_DOT_RE = re.compile(r"\.{3,}")
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_RE = re.compile(r"\s+([,.;:!?…%\)\]\}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])\s+")
_QUOTED_TEXT_RE = re.compile(r'"\s*([^"\n]*?)\s*"')

SENTENCE_END_TOKENS = frozenset({".", "!", "?", "…"})
SENTENCE_TRAILING_TOKENS = frozenset({'"', ")", "]", "}"})
DEFAULT_ABBREVIATIONS = frozenset(
    {"dr", "prof", "doç", "sn", "bkz", "örn", "vb", "vs"}
)


def turkish_lower(text: str) -> str:
    """Lowercase text with Turkish dotted/dotless-I semantics.

    NFKC first composes inputs such as ``I + COMBINING DOT ABOVE`` into ``İ``;
    translating the two Turkish uppercase I forms before ``str.lower`` avoids
    the language-independent ``I -> i`` behavior.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_TURKISH_LOWER_TRANSLATION).lower()
    return unicodedata.normalize("NFC", normalized)


def _replace_controls(text: str) -> str:
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs"}:
            # All controls—including zero-width separators—become boundaries.
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def clean_text(text: str, *, lowercase: bool = True) -> str:
    """Return deterministic, whitespace-normalized UTF-8-friendly text."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION_TRANSLATION)
    text = _replace_controls(text)
    text = _MULTI_DOT_RE.sub("…", text)
    if lowercase:
        text = turkish_lower(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str, *, already_clean: bool = False) -> list[str]:
    """Tokenize words, numbers, and useful punctuation without ASCII bias."""

    cleaned = text if already_clean else clean_text(text)
    return [
        "…" if token.startswith("...") else token
        for token in _TOKEN_RE.findall(cleaned)
    ]


def is_lexical_token(token: str) -> bool:
    """Return whether a token represents a word/number rather than punctuation."""

    return any(char.isalnum() for char in token)


def iter_sentences(
    tokens: Sequence[str],
    *,
    abbreviations: frozenset[str] = DEFAULT_ABBREVIATIONS,
) -> Iterator[list[str]]:
    """Split tokenized paragraph text into sentence-like model sequences."""

    current: list[str] = []
    pending_end = False
    for token in tokens:
        if pending_end and token not in SENTENCE_TRAILING_TOKENS | SENTENCE_END_TOKENS:
            yield current
            current = []
            pending_end = False
        current.append(token)
        if token not in SENTENCE_END_TOKENS:
            continue
        if token == "." and len(current) >= 2 and current[-2] in abbreviations:
            continue
        pending_end = True
    if current:
        yield current


def iter_character_words(
    tokens_or_text: Iterable[str] | str,
    *,
    strict_alphabet: bool = True,
    min_length: int = 1,
    max_length: int = 64,
) -> Iterator[str]:
    """Yield clean training words for the character-level model.

    Apostrophe suffixes are joined (``ankara'da -> ankarada``), hyphenated
    compounds are split, and mixed alphanumeric/OCR tokens are rejected.
    """

    tokens = (
        tokenize(tokens_or_text)
        if isinstance(tokens_or_text, str)
        else tokens_or_text
    )
    for token in tokens:
        if not is_lexical_token(token):
            continue
        for part in token.split("-"):
            word = part.replace("'", "")
            if not (min_length <= len(word) <= max_length):
                continue
            if not all(char.isalpha() for char in word):
                continue
            if strict_alphabet and any(
                char not in TURKISH_CHARACTER_ALPHABET for char in word
            ):
                continue
            yield word


def normalize_character_prompt(prompt: str) -> str:
    """Normalize a character-model prefix and require at most one word."""

    cleaned = clean_text(prompt)
    raw_tokens = [
        token
        for token in tokenize(cleaned, already_clean=True)
        if is_lexical_token(token)
    ]
    if not raw_tokens:
        return ""
    if len(raw_tokens) != 1:
        raise ValueError("character prompt must contain at most one word")
    word = raw_tokens[0].replace("'", "").replace("-", "")
    if not word or not all(char.isalpha() for char in word):
        raise ValueError("character prompt must contain letters only")
    if any(char not in TURKISH_CHARACTER_ALPHABET for char in word):
        raise ValueError("character prompt contains letters outside the Turkish alphabet")
    return word


def detokenize(tokens: Sequence[str]) -> str:
    """Join word-model tokens with conventional Turkish punctuation spacing."""

    if not tokens:
        return ""
    text = " ".join(tokens)
    text = _SPACE_BEFORE_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _QUOTED_TEXT_RE.sub(r'"\1"', text)
    return text.strip()
