import copy
import csv
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acceptance import (boundary_rows, evaluate_reviews, merge_review_sheet, decide_sample,
                        release_ready, validation_fingerprint, verify_digital_silence)
from alignment import assess_alignment
from pipeline import manual_phrases, save_features
from prepare import sha256
from reaggregate import check_sealed_native
from test_alignment_quality import QUALITY


def metadata(manual=False):
    return {
        "sample_id": "s", "text": "one two", "duration_s": 10,
        "fingerprint_components": {"source_sha256": "source"},
        "alignment_quality": {"hard_passed": True},
        "phrases": [{"phrase_index": 0, "char_start": 0, "char_end": 7, "text": "one two",
                     "start_s": 1.0, "end_s": 2.0, "flags": [], "word_indices": [],
                     "alignment_source": "manual" if manual else "automatic",
                     "human_annotated": manual, "reviewer": "annotator" if manual else ""}],
    }


def reviewed_rows(meta=None, state="approved"):
    rows = boundary_rows(meta or metadata(), "feature-hash")
    for row in rows:
        row.update(reference_s=str(row["current_s"]), reviewer="reviewer", review_status=state,
                   reference_source="correction")
    return rows


class ReviewAndCacheTest(unittest.TestCase):
    def test_reviewed_does_not_approve(self):
        result = evaluate_reviews(reviewed_rows(state="reviewed"))
        self.assertTrue(result["samples"]["s"]["checked"])
        self.assertFalse(result["samples"]["s"]["approved"])

    def test_approved_mismatching_output_requires_reaggregation(self):
        rows = reviewed_rows()
        rows[1]["reference_s"] = "9"
        result = evaluate_reviews(rows)
        self.assertTrue(result["samples"]["s"]["needs_revision"])
        self.assertFalse(result["samples"]["s"]["approved"])

    def test_nonfinite_negative_and_out_of_duration_references_rejected(self):
        for bad in ("nan", "inf", "-inf", "-1", "11", "", "abc"):
            with self.subTest(reference=bad):
                rows = reviewed_rows()
                rows[0]["reference_s"] = bad
                result = evaluate_reviews(rows)
                self.assertFalse(result["samples"]["s"]["approved"])
                self.assertGreater(result["invalid_rows"], 0)

    def test_reversed_and_incomplete_reference_interval(self):
        rows = reviewed_rows()
        rows[0]["reference_s"] = "3"
        self.assertEqual(evaluate_reviews(rows)["invalid_rows"], 2)
        self.assertEqual(evaluate_reviews(reviewed_rows()[:1])["invalid_rows"], 1)

    def test_overlapping_reference_phrases(self):
        rows = reviewed_rows()
        second = copy.deepcopy(rows)
        for row in second:
            row["phrase_index"] = 1
        all_rows = rows + second
        self.assertEqual(evaluate_reviews(all_rows)["invalid_rows"], 4)

    def test_duplicate_reference_key_is_invalid(self):
        rows = reviewed_rows()
        rows.append(copy.deepcopy(rows[0]))
        self.assertFalse(evaluate_reviews(rows)["samples"]["s"]["approved"])

    def test_reference_error_metrics_do_not_include_corrections(self):
        result = evaluate_reviews(reviewed_rows(metadata(manual=True)))
        self.assertEqual(len(result["corrections"]), 2)
        self.assertEqual(result["independent"], [])

    def test_independent_reference_requires_identity_and_different_reviewer(self):
        for same_author, evidence_id in ((True, "independent-a"), (False, "")):
            rows = reviewed_rows(metadata(manual=True))
            for row in rows:
                row.update(reference_source="independent", reference_id=evidence_id,
                           reviewer="annotator" if same_author else "other")
            self.assertEqual(evaluate_reviews(rows)["independent"], [])
        rows = reviewed_rows(metadata(manual=True))
        for row in rows:
            row.update(reference_source="independent", reference_id="independent-a", reviewer="other")
        self.assertEqual(len(evaluate_reviews(rows)["independent"]), 2)

    def test_manual_import_does_not_create_reference_or_approval(self):
        rows = boundary_rows(metadata(manual=True), "file")
        self.assertTrue(all(r["reference_s"] == "" and r["review_status"] == "pending" for r in rows))
        decision = decide_sample(metadata(manual=True), True, {}, True)
        self.assertFalse(decision["release_eligible"])
        self.assertEqual(decision["status"], "manual_reaggregated_awaiting_approval")

    def test_quality_failure_requires_documented_explicit_override(self):
        meta = metadata()
        meta["alignment_quality"]["hard_passed"] = False
        rows = reviewed_rows(meta)
        review = evaluate_reviews(rows)["samples"]["s"]
        self.assertFalse(decide_sample(meta, True, review, True)["release_eligible"])
        for row in rows:
            row["decision_note"] = "Actual interval checked; rapid speech confirmed in this fixture."
        decision = decide_sample(meta, True, evaluate_reviews(rows)["samples"]["s"], True)
        self.assertTrue(decision["aligned_accepted"])
        self.assertFalse(decision["current_quality_passed"])
        self.assertTrue(release_ready([decision], 1, True))

    def test_review_activity_blocks_automatic_acceptance(self):
        rows = reviewed_rows(state="needs_revision")
        decision = decide_sample(metadata(), True, evaluate_reviews(rows)["samples"]["s"], False)
        self.assertFalse(decision["release_eligible"])
        self.assertTrue(decide_sample(metadata(), True, {}, False)["release_eligible"])

    def test_source_exception_is_neither_alignment_nor_human_approval(self):
        meta = metadata()
        meta["phrases"] = []
        decision = decide_sample(meta, True, {}, False, {"verified": True})
        self.assertTrue(release_ready([decision], 1, True))
        self.assertFalse(decision["aligned_accepted"])
        self.assertFalse(decision["human_approved"])
        self.assertFalse(decide_sample(meta, True, {}, False)["release_eligible"])
        self.assertFalse(decide_sample(meta, False, {}, False, {"verified": True})["release_eligible"])

    def test_clip_context_branches_are_automatic_final_results(self):
        for audio_observed, status in (
            (True, "clip_context_speech_unavailable_accepted"),
            (False, "clip_context_audio_missing_accepted"),
        ):
            with self.subTest(audio_observed=audio_observed):
                meta = metadata()
                meta.update(
                    speech_alignment_available=False,
                    audio_observed=audio_observed,
                    automatic_boundary_evidence={"protocol_passed": True},
                )
                decision = decide_sample(meta, True, {}, False)
                self.assertTrue(decision["release_eligible"])
                self.assertEqual(decision["status"], status)

    def test_changed_text_time_or_artifact_invalidates_approval(self):
        for change in ("text", "time", "artifact"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "review.csv"
                rows = reviewed_rows()
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                result, stale = merge_review_sheet(path, boundary_rows(metadata(), "feature-hash"))
                self.assertTrue(result["samples"]["s"]["approved"])
                self.assertEqual(stale, 0)
                changed = metadata()
                if change == "text":
                    changed["text"] = changed["phrases"][0]["text"] = "one new"
                elif change == "time":
                    changed["phrases"][0]["end_s"] = 3.0
                result, stale = merge_review_sheet(
                    path, boundary_rows(changed, "new-hash" if change == "artifact" else "feature-hash"))
                self.assertFalse(result["samples"]["s"]["approved"])
                self.assertEqual(stale, 2)
                self.assertTrue(path.with_suffix(".history.jsonl").exists())

    def test_package_fingerprint_detects_review_feature_and_policy_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("annotations/boundary_review.csv", "outputs/features/s.npz", "src/acceptance.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                before = validation_fingerprint(root, sha256)
                path.write_text("first")
                self.assertNotEqual(before, validation_fingerprint(root, sha256))
                before = validation_fingerprint(root, sha256)
                path.write_text("changed")
                self.assertNotEqual(before, validation_fingerprint(root, sha256))

    def test_stereo_cancellation_does_not_qualify_as_source_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            for nonzero in (False, True):
                data = np.zeros((160, 2), dtype="<i2")
                if nonzero:
                    data[:, 0], data[:, 1] = 100, -100
                with wave.open(str(path), "wb") as output:
                    output.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
                    output.writeframes(data.tobytes())
                result = verify_digital_silence(path, sha256(path), sha256)
                self.assertEqual(result["verified"], not nonzero)
                self.assertEqual(result["channel_counts"], [2])

    def test_manual_import_rejects_invalid_ranges_and_never_auto_approves(self):
        with tempfile.TemporaryDirectory() as directory, patch("pipeline.ROOT", Path(directory)):
            root = Path(directory)
            (root / "annotations").mkdir()
            row = {"sample_id": "s", "duration_s": 10, "text": "one two"}
            p = {"start_s": 1, "end_s": 2, "char_start": 0, "char_end": 7}
            def write(phrase):
                (root / "annotations/manual_phrases.jsonl").write_text(json.dumps(
                    {"sample_id": "s", "reviewer": "annotator", "reason": "actual correction",
                     "phrases": [phrase]}))
            write(p)
            result = manual_phrases(row)
            self.assertTrue(result[0]["human_annotated"])
            self.assertFalse(result[0]["human_reviewed"])
            for field, value in (("end_s", float("nan")), ("end_s", float("inf")), ("start_s", -1),
                                 ("char_start", 1), ("char_end", 6), ("char_end", 7.0)):
                with self.subTest(field=field, value=value):
                    write({**p, field: value})
                    with self.assertRaises(ValueError):
                        manual_phrases(row)

    def test_manual_quality_still_checks_duration_and_speaking_rate(self):
        p = metadata(manual=True)["phrases"][0]
        p["end_s"] = p["start_s"] + .01
        result = assess_alignment([], [p], QUALITY)
        self.assertFalse(result["hard_passed"])
        self.assertTrue(result["review_required"])
        self.assertIsNone(result["phrase_checks"][0]["instant_word_fraction"])

    def test_correction_actually_changes_aggregation_with_native_features(self):
        text = {"ids": np.array([1, 2]), "offsets": np.array([[0, 3], [4, 7]]),
                "embeddings": np.ones((2, 128), np.float32)}
        audio = {"values": np.array([[10.] * 15, [20.] * 15]),
                 "intervals": np.array([[0., 1.], [1., 2.]]), "f0": np.array([100., 200.]),
                 "f0_intervals": np.array([[0., 1.], [1., 2.]]), "f0_valid": np.array([True, True])}
        vision = {"values": np.ones((2, 6)), "valid": np.ones((2, 6), bool),
                  "time": np.array([.5, 1.5]), "pts": np.array([.5, 1.5]),
                  "frame_index": np.array([0, 1]), "ambiguous": np.zeros(2, bool),
                  "crop_fallback_used": np.zeros(2, bool), "geometry_rejected": np.zeros(2, bool)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.npz"
            p = metadata(manual=True)["phrases"][0]
            for start, end, expected in ((0., 1., 10.), (1., 2., 20.)):
                p.update(start_s=start, end_s=end, flags=[])
                save_features(path, metadata(), {"schema_version": "test"}, [p],
                              {"hard_passed": True}, text, audio, vision)
                with np.load(path) as data:
                    self.assertEqual(data["features"][0, 128], expected)
                    self.assertEqual(data["intervals"][0].tolist(), [start, end])

    def test_audio_missing_context_masks_all_acoustic_dimensions(self):
        text = {"ids": np.array([1]), "offsets": np.array([[0, 3]]),
                "embeddings": np.ones((1, 128), np.float32)}
        audio = {"values": np.ones((1, 15)), "intervals": np.array([[0., 1.]]),
                 "f0": np.array([100.]), "f0_intervals": np.array([[0., 1.]]),
                 "f0_valid": np.array([True])}
        vision = {"values": np.ones((1, 6)), "valid": np.ones((1, 6), bool),
                  "time": np.array([.5]), "pts": np.array([.5]),
                  "frame_index": np.array([0]), "ambiguous": np.zeros(1, bool),
                  "crop_fallback_used": np.zeros(1, bool),
                  "geometry_rejected": np.zeros(1, bool)}
        phrase = {
            "phrase_index": 0, "start_s": 0., "end_s": 1.,
            "char_start": 0, "char_end": 3, "text": "one", "word_indices": [],
            "alignment_source": "clip-context/digital_silence", "flags": [],
            "context_only": True, "audio_missing": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.npz"
            save_features(path, {"sample_id": "s", "duration_s": 1.},
                          {"schema_version": "test"}, [phrase],
                          {"hard_passed": True}, text, audio, vision)
            with np.load(path) as data:
                self.assertFalse(data["valid"][0, 128:144].any())
                self.assertTrue(np.all(data["features"][0, 128:144] == 0))
                self.assertFalse(bool(data["audio_observed"]))
                self.assertEqual(str(data["granularity"]), "clip_context")

    def test_native_dependency_and_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for name in ("text.npz", "audio.npz", "vision.npz"):
                (cache / name).write_bytes(b"original")
            from pipeline import json_hash
            seal = {"dependencies": {"text": "original"}, "automatic_words_sha256": json_hash([]),
                    "alignment_config": {}, "alignment_retry_sha256": "retry",
                    "files": {p.name: sha256(p) for p in cache.iterdir()}}
            check_sealed_native(seal, {"text": "original"}, cache, [], {"alignment": {}},
                                {"alignment_retry_sha256": "retry"})
            with self.assertRaises(ValueError):
                check_sealed_native(seal, {"text": "changed"}, cache, [], {"alignment": {}},
                                    {"alignment_retry_sha256": "retry"})
            (cache / "audio.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                check_sealed_native(seal, {"text": "original"}, cache, [], {"alignment": {}},
                                    {"alignment_retry_sha256": "retry"})


if __name__ == "__main__":
    unittest.main()
