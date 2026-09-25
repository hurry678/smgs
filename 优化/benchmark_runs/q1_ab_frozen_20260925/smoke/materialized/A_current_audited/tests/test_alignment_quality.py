import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignment import assess_alignment, phrase_groups


QUALITY = {
    "minimum_phrase_duration_s": 0.08,
    "maximum_words_per_second": 8.0,
    "maximum_instant_word_fraction": 0.2,
    "low_word_probability": 0.2,
    "review_low_probability_fraction": 0.0,
}


def word(start, end, probability=0.9):
    return {"start_s": start, "end_s": end, "instant": end <= start,
            "probability": probability}


class AlignmentQualityTest(unittest.TestCase):
    def test_normal_phrase_passes_without_review(self):
        words = [word(0.0, 0.4), word(0.4, 0.8), word(0.8, 1.2)]
        result = assess_alignment(words, [{"phrase_index": 0, "start_s": 0.0,
                                           "end_s": 1.2, "word_indices": [0, 1, 2]}], QUALITY)
        self.assertTrue(result["hard_passed"])
        self.assertFalse(result["review_required"])

    def test_many_instant_words_are_rejected(self):
        words = [word(3.9, 3.9) for _ in range(5)] + [word(3.9, 3.98)]
        result = assess_alignment(words, [{"phrase_index": 0, "start_s": 3.9,
                                           "end_s": 3.98, "word_indices": list(range(6))}], QUALITY)
        self.assertFalse(result["hard_passed"])
        self.assertIn("p0:implausible_word_rate", result["hard_issues"])
        self.assertIn("p0:excess_instant_words", result["hard_issues"])

    def test_short_phrase_is_rejected(self):
        result = assess_alignment([word(1.0, 1.02)], [{"phrase_index": 0, "start_s": 1.0,
                                                       "end_s": 1.02, "word_indices": [0]}], QUALITY)
        self.assertFalse(result["hard_passed"])
        self.assertIn("p0:phrase_too_short", result["hard_issues"])

    def test_low_probability_requires_review_but_is_not_hard_failure(self):
        result = assess_alignment([word(0.0, 0.5, 0.1)], [{"phrase_index": 0, "start_s": 0.0,
                                                           "end_s": 0.5, "word_indices": [0]}], QUALITY)
        self.assertTrue(result["hard_passed"])
        self.assertTrue(result["review_required"])

    def test_tiny_isolated_group_merges_with_nearest_neighbor(self):
        words = [
            {**word(0.0, 0.06), "text": "And, ", "char_start": 0, "char_end": 5,
             "end_clipped": False, "alignment_source": "mfa"},
            {**word(0.06, 0.45), "text": "then ", "char_start": 5, "char_end": 10,
             "end_clipped": False, "alignment_source": "mfa"},
            {**word(0.45, 0.9), "text": "continue.", "char_start": 10, "char_end": 19,
             "end_clipped": False, "alignment_source": "mfa"},
        ]
        config = {
            "gap_s": 0.3, "max_phrase_words": 8, "max_phrase_s": 4.0,
            "low_word_probability": 0.2,
        }
        phrases = phrase_groups(words, "And, then continue.", config, QUALITY)
        self.assertEqual(phrases[0]["word_indices"], [0, 1, 2])
        self.assertIn("automatic_quality_merge", phrases[0]["flags"])
        self.assertTrue(assess_alignment(words, phrases, QUALITY)["hard_passed"])


if __name__ == "__main__":
    unittest.main()
