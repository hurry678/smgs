from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXTENSION_ROOT.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (EXTENSION_ROOT / "configs" / "protocol.json").read_text(encoding="utf-8")
        )

    def test_immutable_repo_hashes(self) -> None:
        for relative, expected in self.protocol["immutable_repo_files"].items():
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256_file(path), expected, relative)

    def test_validation_and_attachment_policies(self) -> None:
        validation = self.protocol["validation"]
        self.assertEqual(self.protocol["schema"], "q3v2.2-protocol-v1")
        self.assertEqual(validation["expected_samples"], 728)
        self.assertEqual(validation["window_candidates"], [3, 5, 10])
        self.assertEqual(validation["fallback_window"], 5)
        self.assertEqual(validation["bootstrap_unit"], "video_id")
        self.assertEqual(
            validation["comparison_schema"],
            "q3v2.2-symmetric-top-support-v1",
        )
        self.assertFalse(self.protocol["attachment4"]["used_for_tuning"])

    def test_alignment_claim_matches_implementation(self) -> None:
        alignment = self.protocol["alignment"]
        self.assertEqual(alignment["mfa_version"], "3.4.1")
        self.assertFalse(alignment["energy_pause_dp_fallback_implemented"])
        script = (EXTENSION_ROOT / "scripts" / "q3_prepare_alignment_v2.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\([^)]+\)|\[[^\]]+\]", script)
        self.assertIn("elif not target:", script)
        self.assertIn("approximate_proportional_token_to_video", script)
        self.assertNotIn('"energy_pause_dp_fallback"', script)


if __name__ == "__main__":
    unittest.main()
