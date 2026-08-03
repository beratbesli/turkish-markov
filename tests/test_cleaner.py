from __future__ import annotations

import unittest

from turkish_markov.cleaner import (
    clean_text,
    detokenize,
    iter_character_words,
    iter_sentences,
    normalize_character_prompt,
    tokenize,
    turkish_lower,
)


class TurkishCleanerTests(unittest.TestCase):
    def test_turkish_i_rules_and_combining_dot(self) -> None:
        self.assertEqual(turkish_lower("I İ i ı"), "ı i i ı")
        self.assertEqual(
            clean_text("IĞDIR İZMİR KIŞ IŞIĞI"),
            "ığdır izmir kış ışığı",
        )
        self.assertEqual(clean_text("I\u0307STANBUL"), "istanbul")
        self.assertNotIn("\u0307", clean_text("I\u0307STANBUL"))

    def test_circumflexes_and_apostrophes_are_preserved(self) -> None:
        self.assertEqual(
            clean_text("KÂĞIT MİLLÎ SÜKÛN"),
            "kâğıt millî sükûn",
        )
        cleaned = clean_text("Ankara’da, İzmir‘de; Türkiyeʼnin")
        self.assertEqual(cleaned, "ankara'da, izmir'de; türkiye'nin")

    def test_tokenization_keeps_internal_joiners(self) -> None:
        tokens = tokenize("“Ankara'da yemyeşil-masmavi bir dünya…”")
        self.assertEqual(
            tokens,
            ['"', "ankara'da", "yemyeşil-masmavi", "bir", "dünya", "…", '"'],
        )
        self.assertEqual(detokenize(["merhaba", ",", "dünya", "!"]), "merhaba, dünya!")

    def test_character_words_are_strict_and_do_not_cross_words(self) -> None:
        words = list(iter_character_words("ığdır'da, kâğıt-kalem ve café42"))
        self.assertEqual(words, ["ığdırda", "kâğıt", "kalem", "ve"])
        self.assertEqual(normalize_character_prompt("IŞ"), "ış")
        with self.assertRaises(ValueError):
            normalize_character_prompt("iki kelime")

    def test_abbreviation_does_not_end_sentence(self) -> None:
        sentences = list(iter_sentences(tokenize("Dr. Ali geldi. Sonra gitti!")))
        self.assertEqual(
            sentences,
            [["dr", ".", "ali", "geldi", "."], ["sonra", "gitti", "!"]],
        )

    def test_sentence_closers_stay_with_preceding_sentence(self) -> None:
        tokens = tokenize('"Merhaba." Sonra (gitti!)')
        self.assertEqual(
            list(iter_sentences(tokens)),
            [['"', "merhaba", ".", '"'], ["sonra", "(", "gitti", "!", ")"]],
        )
        self.assertEqual(
            detokenize(['"', "merhaba", ",", "dünya", ".", '"']),
            '"merhaba, dünya."',
        )


if __name__ == "__main__":
    unittest.main()
