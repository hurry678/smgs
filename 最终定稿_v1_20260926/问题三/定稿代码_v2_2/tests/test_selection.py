from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = EXTENSION_ROOT / "scripts" / "q3_select_freeze_v2.py"
sys.path.insert(0, str(EXTENSION_ROOT / "scripts"))

from q3_select_freeze_v2 import load_candidate  # noqa: E402


def report(window: int, score: float, lower: float) -> dict:
    comparison = {
        "n_observations": 6,
        "n_groups": 3,
        "mean_difference": score,
        "bootstrap_95_ci": [lower, score + 0.1],
        "resamples": 5000,
        "seed": 2026,
    }
    return {
        "mode": "validation",
        "protocol": {
            "window_size": window,
            "stride": window,
            "n_selected": 3,
        },
        "model": {
            "freeze_manifest_check": {"checked": True, "ok": True},
        },
        "full_validation_metrics": {"accuracy": 0.6},
        "selected_sample_ids": ["a$_$0", "b$_$0", "c$_$0"],
        "local_occlusion": {
            "top_support_fidelity": {
                "comparison_schema": "q3v2.2-symmetric-top-support-v1",
                "control_selection_symmetric": True,
                "mean_budget_normalized_top_support": score,
                "mean_budget_normalized_top_random_contiguous": score - lower,
                "mean_budget_normalized_top_point_scatter": score - lower,
                "top_support_vs_random_contiguous": comparison,
                "top_support_vs_point_scatter": comparison,
            }
        },
    }


class SelectionTests(unittest.TestCase):
    def run_selector(self, lowers: dict[int, float]) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_args: list[str] = []
            for window, score in ((3, 0.4), (5, 0.5), (10, 0.9)):
                path = root / f"w{window}" / "q3_validation_report.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps(report(window, score, lowers[window])),
                    encoding="utf-8",
                )
                report_args.extend(["--report", f"{window}={path}"])
            q2_freeze = root / "q2_freeze.json"
            q2_version = root / "q2_version.json"
            q2_freeze.write_text("{}", encoding="utf-8")
            q2_version.write_text("{}", encoding="utf-8")
            output = root / "freeze.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SELECTOR),
                    *report_args,
                    "--out",
                    str(output),
                    "--expected-n",
                    "3",
                    "--q2-freeze-manifest",
                    str(q2_freeze),
                    "--q2-version-manifest",
                    str(q2_version),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(output.read_text(encoding="utf-8"))

    def test_selects_best_eligible_budget_normalized_score(self) -> None:
        result = self.run_selector({3: 0.01, 5: 0.01, 10: -0.01})
        self.assertEqual(result["schema"], "q3v2.2-explanation-freeze-v1")
        self.assertEqual(result["selected_window_size"], 5)
        self.assertEqual(result["selection_mode"], "clustered_ci_eligible_best_score")

    def test_uses_preregistered_fallback_if_no_candidate_passes(self) -> None:
        result = self.run_selector({3: -0.01, 5: -0.01, 10: -0.01})
        self.assertEqual(result["selected_window_size"], 5)
        self.assertEqual(
            result["selection_mode"],
            "preregistered_middle_scale_fallback",
        )

    def test_rejects_legacy_asymmetric_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            payload = report(3, 0.4, 0.1)
            del payload["local_occlusion"]["top_support_fidelity"]["comparison_schema"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not symmetric v2.2"):
                load_candidate(3, path, 3)


if __name__ == "__main__":
    unittest.main()
