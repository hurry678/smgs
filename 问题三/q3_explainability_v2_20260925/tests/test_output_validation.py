from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from q3_build_outputs_v2 import validate_sample  # noqa: E402


def valid_sample() -> dict:
    return {
        "sample_id": "01",
        "polarity": "Positive",
        "intensity": 0.8,
        "primary_modality": "audio",
        "primary_modality_basis": "largest_positive_class_shapley",
        "strongest_modality": "text",
        "support_modality": "audio",
        "class_contrib_text": -0.7,
        "class_contrib_audio": 0.4,
        "class_contrib_vision": 0.2,
        "class_shapley_sum_error": 1e-7,
        "intensity_shapley_sum_error": 1e-7,
        "class_pair_interactions": {
            "text_audio": 0.1,
            "text_vision": 0.0,
            "audio_vision": -0.1,
        },
        "intensity_pair_interactions": {
            "text_audio": 0.0,
            "text_vision": 0.1,
            "audio_vision": 0.2,
        },
        "key_evidence": {
            "text": [
                {
                    "evidence_direction": "support",
                    "class_delta": 0.2,
                    "mapping_status": "exact",
                    "text_char_start": 0,
                    "text_char_end": 4,
                    "text_snippet": "good",
                }
            ],
            "audio": [
                {
                    "evidence_direction": "support",
                    "class_delta": 0.1,
                    "mapping_status": "forced_alignment",
                    "time_start_seconds": 0.1,
                    "time_end_seconds": 0.4,
                }
            ],
            "vision": [
                {
                    "evidence_direction": "support",
                    "class_delta": 0.1,
                    "mapping_status": "forced_alignment",
                    "time_start_seconds": 0.1,
                    "time_end_seconds": 0.4,
                    "frame_start_est": 3,
                    "frame_end_est": 10,
                }
            ],
        },
        "counter_evidence": {"text": [], "audio": [], "vision": []},
        "no_evidence_reason": None,
    }


class OutputValidationTests(unittest.TestCase):
    def test_valid_sample(self) -> None:
        self.assertEqual(validate_sample(valid_sample(), "good movie", "01"), [])

    def test_rejects_absolute_influence_as_primary_support(self) -> None:
        sample = valid_sample()
        sample["primary_modality"] = "text"
        errors = validate_sample(sample, "good movie", "01")
        self.assertTrue(
            any("primary_modality violates support rule" in error for error in errors)
        )

    def test_rejects_relative_only_attachment_mapping(self) -> None:
        sample = valid_sample()
        sample["key_evidence"]["audio"][0] = {
            "evidence_direction": "support",
            "class_delta": 0.1,
            "mapping_status": "relative_only",
            "relative_start": 0.1,
            "relative_end": 0.2,
        }
        errors = validate_sample(sample, "good movie", "01")
        self.assertTrue(
            any("invalid audio mapping status" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
