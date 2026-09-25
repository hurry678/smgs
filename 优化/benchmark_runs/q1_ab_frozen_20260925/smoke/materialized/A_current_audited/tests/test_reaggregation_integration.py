"""Exercise a correction on a copied real fixture; never annotate the real dataset."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pipeline
import reaggregate
import validate
import package

SID = "-THoVjtIkeU$_$2"


@unittest.skipUnless((ROOT / "intermediate/samples" / SID / "native_provenance.json").exists(),
                     "Requires local native fixture; not included in diagnostic archive")
class ReaggregationIntegrationTest(unittest.TestCase):
    def test_correction_recompute_review_approval_and_stale_package_guard(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            for module in (pipeline, reaggregate, validate, package):
                stack.enter_context(patch.object(module, "ROOT", root))
            for name in ("src", "configs"):
                shutil.copytree(ROOT / name, root / name)
            for name in ("outputs/features", "outputs/details", "annotations", "logs", "reports",
                         "intermediate/probes", "intermediate/audio", "resources"):
                (root / name).mkdir(parents=True, exist_ok=True)
            for name in ("models", "input"):
                (root / "resources" / name).symlink_to(ROOT / "resources" / name, target_is_directory=True)
            for name in ("input_manifest.json", "model_sources.json", "resource_lock.json"):
                shutil.copy2(ROOT / "resources" / name, root / "resources" / name)
            shutil.copy2(ROOT / "requirements.lock.txt", root / "requirements.lock.txt")
            shutil.copy2(ROOT / "run.py", root / "run.py")
            row = next(r for r in json.loads((ROOT / "resources/manifest.json").read_text())
                       if r["sample_id"] == SID)
            (root / "resources/manifest.json").write_text(json.dumps([row]))
            for name in (f"outputs/features/{SID}.npz", f"outputs/details/{SID}.json",
                         f"intermediate/probes/{SID}.json", f"intermediate/audio/{SID}.wav"):
                shutil.copy2(ROOT / name, root / name)
            shutil.copytree(ROOT / "intermediate/samples" / SID, root / "intermediate/samples" / SID)
            original = json.loads((root / f"outputs/details/{SID}.json").read_text())
            stack.enter_context(patch.object(
                reaggregate, "ensure_mfa_alignments",
                return_value={
                    "fingerprint": original["automatic_boundary_evidence"][
                        "mfa_batch_fingerprint"]
                }))
            stack.enter_context(patch.object(
                reaggregate, "load_mfa_words",
                return_value=original["automatic_words"]))
            stack.enter_context(patch.object(
                reaggregate, "vad_intervals",
                return_value=original["automatic_boundary_evidence"]["vad_intervals_s"]))
            with np.load(root / original["feature_file"]) as z:
                before = z["features"].copy()
            phrases = [{k: p[k] for k in ("start_s", "end_s", "char_start", "char_end")}
                       for p in original["phrases"]]
            phrases[0]["end_s"] = phrases[1]["start_s"] = 1.0
            (root / "annotations/manual_phrases.jsonl").write_text(json.dumps({
                "sample_id": SID, "reviewer": "TEST_ONLY_ANNOTATOR",
                "reason": "Synthetic correction in temporary fixture, not human evidence",
                "phrases": phrases}))
            config = json.loads((root / "configs/q1.json").read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                reaggregate.main(config)
                validate.main(diagnostic=True, config=config)
            corrected = json.loads((root / f"outputs/details/{SID}.json").read_text())
            self.assertEqual(corrected["automatic_phrases"], original["automatic_phrases"])
            with np.load(root / original["feature_file"]) as z:
                self.assertFalse(np.array_equal(before, z["features"]))
                self.assertEqual(float(z["intervals"][0, 1]), 1.0)
            report = json.loads((root / "reports/validation.json").read_text())
            self.assertTrue(report["structural_passed"])
            self.assertFalse(report["strict_release_passed"])
            self.assertEqual(report["human_boundary_evaluation"]["independent_evaluation_boundaries"], 0)
            sheet = root / "annotations/boundary_review.csv"
            with sheet.open(encoding="utf-8-sig") as handle:
                reviews = list(csv.DictReader(handle))
            for state in ("reviewed", "approved"):
                for r in reviews:
                    r.update(reference_s=r["current_s"], reviewer="TEST_ONLY_REVIEWER",
                             reference_source="correction", review_status=state)
                with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
                    writer.writeheader()
                    writer.writerows(reviews)
                with contextlib.redirect_stdout(io.StringIO()):
                    validate.main(diagnostic=True, config=config)
                report = json.loads((root / "reports/validation.json").read_text())
                self.assertEqual(report["strict_release_passed"], state == "approved")
                self.assertIsNone(report["human_boundary_evaluation"]["median_abs_error_s"])
            # Packaging checks freshness before its full 100-sample coverage gate.
            with sheet.open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Validation is stale"):
                package.main(diagnostic=True, config=config)


if __name__ == "__main__":
    unittest.main()
