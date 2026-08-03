from __future__ import annotations

import random
import unittest

from turkish_markov.markov import weighted_sample


class WeightedSamplingTests(unittest.TestCase):
    def test_zero_and_tiny_temperatures_choose_the_mode(self) -> None:
        candidates = [("az", 1), ("çok", 100)]
        source = random.Random(1)
        self.assertEqual(
            weighted_sample(candidates, temperature=0, random_source=source),
            "çok",
        )
        self.assertEqual(
            weighted_sample(candidates, temperature=1e-320, random_source=source),
            "çok",
        )


if __name__ == "__main__":
    unittest.main()
