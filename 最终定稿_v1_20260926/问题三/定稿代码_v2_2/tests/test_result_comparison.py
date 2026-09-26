from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EXTENSION_ROOT / "scripts" / "compare_q3v2_2_results.py"


def payload(window: int, intensity: float, evidence_position: int) -> dict:
    return {
        "report": {"protocol": {"window_size": window}},
        "samples": [
            {
                "sample_id": "01",
                "polarity": "Positive",
                "intensity": intensity,
                "log_mean_probability": [-2.0, -1.0, -0.2],
                "primary_modality": "text",
                "key_evidence": {
                    "text": [{"positions": [evidence_position]}],
                    "audio": [],
                    "vision": [],
                },
                "counter_evidence": {"text": [], "audio": [], "vision": []},
            }
        ],
    }


class ResultComparisonTests(unittest.TestCase):
    def run_comparison(self, new_intensity: float) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_path = root / "old.json"
            new_path = root / "new.json"
            out_path = root / "comparison.json"
            old_path.write_text(json.dumps(payload(3, 0.5, 1)), encoding="utf-8")
            new_path.write_text(
                json.dumps(payload(5, new_intensity, 2)),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--old-raw",
                    str(old_path),
                    "--new-raw",
                    str(new_path),
                    "--out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            result = json.loads(out_path.read_text(encoding="utf-8"))
            return completed.returncode, result

    def test_accepts_explanation_only_change(self) -> None:
        returncode, result = self.run_comparison(0.5)
        self.assertEqual(returncode, 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["explanation_changes"]["evidence_change_count"], 1)

    def test_rejects_predictor_change(self) -> None:
        returncode, result = self.run_comparison(0.6)
        self.assertNotEqual(returncode, 0)
        self.assertEqual(result["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
