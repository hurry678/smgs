import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from automatic_alignment import add_automatic_evidence, attach_protocol_checks
from download_resources import fetch
from prepare import sha256
from audit_boundaries import align_tokens, compare


class ResourceAndEvidenceTest(unittest.TestCase):
    def test_corrupt_existing_resource_cannot_be_reregistered(self):
        with tempfile.TemporaryDirectory() as folder, patch("download_resources.ROOT", Path(folder)):
            root = Path(folder)
            path = root / "resources/models/weights.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"trusted")
            lock = root / "resources/resource_lock.json"
            lock.write_text(json.dumps({"sha256": {"resources/models/weights.bin": sha256(path)}}))
            baseline = lock.read_bytes()
            self.assertEqual(fetch("https://example.invalid/fixed", path)["sha256"], sha256(path))
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Existing resource"):
                fetch("https://example.invalid/fixed", path)
            self.assertEqual(lock.read_bytes(), baseline)
            self.assertEqual(path.read_bytes(), b"corrupted")

    def test_corrupt_download_never_becomes_final_file(self):
        with tempfile.TemporaryDirectory() as folder, patch("download_resources.ROOT", Path(folder)):
            root = Path(folder)
            path = root / "resources/models/weights.bin"
            path.parent.mkdir(parents=True)
            (root / "resources/resource_lock.json").write_text(json.dumps(
                {"sha256": {"resources/models/weights.bin": "0" * 64}}))
            def downloaded(command, **kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"bad-download")
            with patch("download_resources.subprocess.run", side_effect=downloaded):
                with self.assertRaisesRegex(ValueError, "Downloaded resource"):
                    fetch("https://example.invalid/fixed", path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".bin.part").exists())

    def test_zero_duration_secondary_is_not_a_comparable_endpoint(self):
        phrases = [{"phrase_index": 0, "char_start": 0, "char_end": 3,
                    "start_s": 1., "end_s": 2., "alignment_source": "mfa"}]
        words = [{"char_start": 0, "char_end": 3, "start_s": .5, "end_s": .5}]
        config = {"automatic_alignment": {
            "policy": "test", "grade_a_p90_difference_s": .2, "grade_b_p90_difference_s": .5}}
        evidence = add_automatic_evidence(phrases, words, [], config)
        self.assertEqual(evidence["endpoint_count"], 0)
        self.assertEqual(evidence["secondary_unavailable_phrase_indices"], [0])
        self.assertIsNone(evidence["p90_absolute_difference_s"])
        self.assertEqual(phrases[0]["automatic_boundary_evidence"]["absolute_difference_s"], [])
        attach_protocol_checks(evidence, phrases, True)
        self.assertFalse(evidence["protocol_checks"]["secondary_endpoint_coverage"])
        self.assertTrue(evidence["protocol_checks"]["secondary_evidence_accounted"])
        self.assertTrue(evidence["protocol_passed"])

    def test_missing_secondary_accounting_cannot_pass(self):
        evidence = {"endpoint_count": 0}
        attach_protocol_checks(evidence, [{"start_s": 0, "end_s": 1}], True)
        self.assertFalse(evidence["protocol_passed"])

    def test_asr_alignment_cannot_create_matches_for_wrong_transcript(self):
        edits, matches = align_tokens(["one", "two"], ["different", "speech"])
        self.assertEqual(edits, 2)
        self.assertEqual(matches, {})
        edits, matches = align_tokens(["one", "two"], ["one", "extra", "two"])
        self.assertEqual(edits, 1)
        self.assertEqual(matches, {0: 0, 1: 2})

    def test_asr_partial_match_has_no_phrase_timing_claim(self):
        meta = {
            "sample_id": "s", "text": "one two", "audio_start_s": 0, "duration_s": 3,
            "speech_alignment_available": True, "artifact_fingerprint": "artifact",
            "fingerprint_components": {"source_sha256": "source"},
            "phrases": [{"phrase_index": 0, "text": "one two", "char_start": 0,
                         "char_end": 7, "start_s": 0, "end_s": 2}]}
        raw = {"text": "one other", "status": "decoded", "segments": [{
            "words": [{"word": "one", "start": 0., "end": 1.},
                      {"word": "other", "start": 1., "end": 2.}]}]}
        result = compare(meta, raw)
        self.assertEqual(result["phrases"][0]["exact_matched_words"], 1)
        self.assertIsNone(result["phrases"][0]["asr_s"])
        self.assertEqual(result["phrases"][0]["absolute_difference_s"], [])


if __name__ == "__main__":
    unittest.main()
