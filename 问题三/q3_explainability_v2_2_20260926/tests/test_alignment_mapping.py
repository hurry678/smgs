from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from q3_prepare_alignment_v2 import (  # noqa: E402
    map_mfa_entries_to_words,
    proportional_rows,
    token_time_rows,
)


class AlignmentMappingTests(unittest.TestCase):
    def test_mfa_words_map_back_to_original_punctuation(self) -> None:
        text = "Well, this is good!"
        entries = [
            [0.10, 0.40, "well"],
            [0.45, 0.70, "this"],
            [0.72, 0.82, "is"],
            [0.84, 1.20, "good"],
        ]
        words = map_mfa_entries_to_words(text, entries, duration=1.5)
        self.assertEqual("".join(word["text"] for word in words), text)
        self.assertEqual(words[0]["text"], "Well, ")
        self.assertEqual(words[-1]["text"], "good!")

    def test_token_mapping_uses_word_union_and_frames(self) -> None:
        words = [
            {
                "char_start": 0,
                "char_end": 6,
                "time_start_seconds": 0.1,
                "time_end_seconds": 0.4,
            },
            {
                "char_start": 6,
                "char_end": 10,
                "time_start_seconds": 0.5,
                "time_end_seconds": 0.8,
            },
        ]
        offsets = np.asarray([[0, 4], [4, 8], [8, 10]])
        valid = np.asarray([True, True, True])
        rows = token_time_rows(offsets, valid, words, duration=1.0, fps=25.0)
        self.assertEqual(rows[1]["time_start_seconds"], 0.1)
        self.assertEqual(rows[1]["time_end_seconds"], 0.8)
        self.assertEqual(rows[1]["frame_start"], 2)
        self.assertEqual(rows[1]["frame_end"], 20)
        self.assertEqual(rows[1]["status"], "mfa_forced_alignment")

    def test_proportional_fallback_is_explicit(self) -> None:
        offsets = np.asarray([[0, 2], [2, 5], [0, 0]])
        valid = np.asarray([True, True, False])
        rows = proportional_rows(
            offsets,
            valid,
            text_length=5,
            duration=10.0,
            fps=30.0,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["status"],
            "approximate_proportional_token_to_video",
        )
        self.assertEqual(rows[1]["time_end_est"], 10.0)
        self.assertEqual(rows[0]["confidence"], 0.0)

    def test_mfa_bracket_label_accepts_parenthesized_source(self) -> None:
        words = map_mfa_entries_to_words(
            "(laughs) okay",
            [[0.1, 0.3, "[bracketed]"], [0.4, 0.8, "okay"]],
            duration=1.0,
        )
        self.assertEqual("".join(word["text"] for word in words), "(laughs) okay")

    def test_punctuation_only_mfa_token_is_consumed_literally(self) -> None:
        words = map_mfa_entries_to_words(
            "okay]",
            [[0.1, 0.5, "okay"], [0.5, 0.6, "]"]],
            duration=1.0,
        )
        self.assertEqual("".join(word["text"] for word in words), "okay]")


if __name__ == "__main__":
    unittest.main()
