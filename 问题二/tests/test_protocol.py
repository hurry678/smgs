"""Mechanism checks: padding, missingness, pairing, and conditional uncertainty."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from protocol_core import (interface_masks, sanitize_text, make_missing, gap_structure,
                           fit_normalizer, normalize, publish_c2)
from evaluate_predictions import evaluate, scores, statistics


def fixture():
    tb = np.zeros((2, 3, 50))
    tb[:, 0, :10] = [101, 100, 2003, 1037, 2204, 3185, 2000, 2001, 2002, 102]
    tb[:, 1, :10] = 1
    a, v = np.zeros((2, 50, 74)), np.zeros((2, 50, 35))
    a[:, 1:9], v[:, 1:9] = 1, 1
    return tb, a, v


class MasksTest(unittest.TestCase):
    def test_unknown_and_legitimate_zero_dimension(self):
        tb, a, v = fixture()
        a[0, 3, 0] = 0
        a[0, 4] = 0
        d, o = interface_masks(tb, a, v)
        self.assertTrue(o[0, 0, 1])  # UNK remains an observed unknown.
        self.assertTrue(o[0, 1, 3])
        self.assertFalse(o[0, 1, 4])
        self.assertFalse(o[:, :, [0, 9, 49]].any())

    def test_interior_text_hole_does_not_shorten_domain(self):
        tb, a, v = fixture()
        tb[0, :, 3] = 0
        d, o = interface_masks(tb, a, v)
        self.assertEqual(int(d[0].sum()), 8)
        self.assertFalse(o[0, 0, 3])
        self.assertTrue(o[0, 1, 3])

    def test_reject_nonfinite_and_ambiguous_separator(self):
        tb, a, v = fixture()
        a[0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            interface_masks(tb, a, v)
        tb, a, v = fixture()
        tb[0, 0, 4] = 102
        with self.assertRaises(ValueError):
            interface_masks(tb, a, v)

    def test_physical_window_and_preserve_natural_missing(self):
        tb, a, v = fixture()
        a[0, 4] = 0
        d, o = interface_masks(tb, a, v)
        m, rows = make_missing(d, o, ["one", "two"], (1,), .5, placement="middle")
        row = rows[0]
        self.assertEqual((row["start"], row["stop"]), (3, 7))
        self.assertFalse(m[0, 1, 3:7].any())
        self.assertFalse((m & ~o).any())
        self.assertEqual(row["original_observed"]-row["visible"], 3)
        self.assertTrue(o[0, 1, 3])  # Inputs were not mutated.

    def test_batch_order_independence(self):
        d, o = interface_masks(*fixture())
        a, _ = make_missing(d, o, ["one", "two"], (0, 1, 2), .3)
        b, _ = make_missing(d[::-1], o[::-1], ["two", "one"], (0, 1, 2), .3)
        np.testing.assert_array_equal(a, b[::-1])

    def test_synchronized_windows_match(self):
        d, o = interface_masks(*fixture())
        _, rows = make_missing(d, o, ["one", "two"], (0, 1, 2), .3, coupling="synchronized")
        for sid in ("one", "two"):
            bounds = {(r["start"], r["stop"]) for r in rows if r["sample_id"] == sid}
            self.assertEqual(len(bounds), 1)

    def test_singleton_and_empty_are_not_erased(self):
        d, o = interface_masks(*fixture())
        o[0, 1] = False
        o[0, 1, 2] = True
        o[0, 2] = False
        m, rows = make_missing(d, o, ["one", "two"], (1, 2), .5)
        np.testing.assert_array_equal(m[0], o[0])
        self.assertTrue(all(r["status"] == "skipped_lt2_observed" for r in rows[:2]))

    def test_gap_features_invariant_to_padding_extension(self):
        d, o = interface_masks(*fixture())
        o[0, :, 3:6] = False
        short = gap_structure(d[:, :10], o[:, :, :10])
        long = gap_structure(d, o)
        np.testing.assert_array_equal(short, long[:, :10])
        self.assertTrue((long[:, 10:] == 0).all())
        self.assertEqual(long.shape[-1], 26)
        self.assertAlmostEqual(float(long[0, 4, 2]), 3/8)

    def test_preencoding_text_sanitization(self):
        tb, a, v = fixture()
        d, o = interface_masks(tb, a, v)
        o[0, 0, 2:6] = False
        changed = tb.copy()
        changed[0, 0, 2:6] = 888
        changed[0, 2, 2:6] = 1
        np.testing.assert_array_equal(sanitize_text(tb, d, o[:, 0]),
                                      sanitize_text(changed, d, o[:, 0]))

    def test_normalization_excludes_unobserved(self):
        x = np.array([[[1., 0.], [3., 0.], [1000., 999.]]])
        obs = np.array([[True, True, False]])
        mean, std = fit_normalizer(x, obs)
        np.testing.assert_allclose(mean, [2., 0.])
        out = normalize(x, obs, mean, std)
        self.assertTrue((out[0, 2] == 0).all())
        self.assertTrue(np.isfinite(out).all())

    def test_publication_signs_and_clipping(self):
        cls, val = publish_c2(np.eye(3), np.array([2., 2., -2.]))
        np.testing.assert_array_equal(cls, [0, 1, 2])
        np.testing.assert_allclose(val, [-1e-4, 0., 1e-4])


class StatisticsTest(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for candidate in ("B2", "M2"):
            for seed in ("42", "43"):
                for replica in ("0", "1"):
                    for i in range(12):
                        c = i % 3
                        value = [-1., 0., 1.][c]
                        rows.append(dict(candidate=candidate, train_seed=seed, bank="confirm",
                                         condition_id="text_0.3", replica=replica, sample_id=str(i),
                                         video_id=str(i//2), y_class=str(c), y_reg=str(value),
                                         polarity=("Negative", "Neutral", "Positive")[c],
                                         intensity=str(value), mask_sha256="a"*64))
        return rows

    def test_identical_predictions_zero_paired_interval(self):
        report = evaluate(self.rows(), "B2", "M2", resamples=40)
        for m in report["metrics"].values():
            self.assertEqual(m["candidate_minus_reference"], 0)
            np.testing.assert_allclose(m["paired_95pct"], [0, 0])
        self.assertEqual(report["video_groups"], 6)
        self.assertEqual(report["samples"], 12)

    def test_known_error_reduction_direction(self):
        rows = self.rows()
        for r in rows:
            if r["candidate"] == "B2":
                r["intensity"] = str(1.5 * float(r["y_reg"]))
        report = evaluate(rows, "B2", "M2", resamples=80)
        mae = report["metrics"]["mae"]
        self.assertAlmostEqual(mae["candidate_minus_reference"], -1/3)
        self.assertLess(mae["paired_95pct"][1], 0)
        self.assertEqual(mae["better_direction"], "lower")

    def test_repeated_conditions_do_not_invent_independent_videos(self):
        rows = self.rows()
        for r in rows:
            if r["candidate"] == "B2":
                r["intensity"] = str(1.5 * float(r["y_reg"]))
        once = evaluate(rows, "B2", "M2", resamples=80)
        repeated = rows + [r | {"condition_id": r["condition_id"] + "_copy"} for r in rows]
        twice = evaluate(repeated, "B2", "M2", resamples=80)
        self.assertEqual(twice["video_groups"], once["video_groups"])
        np.testing.assert_allclose(twice["metrics"]["mae"]["paired_95pct"],
                                   once["metrics"]["mae"]["paired_95pct"])

    def test_mask_mismatch_and_duplicate_rejected(self):
        rows = self.rows()
        rows[-1]["mask_sha256"] = "b"*64
        with self.assertRaises(ValueError):
            evaluate(rows, "B2", "M2", 10)
        rows = self.rows()
        with self.assertRaises(ValueError):
            evaluate(rows+[rows[0]], "B2", "M2", 10)

    def test_missing_class_fixed_macro_and_constant_pearson(self):
        rows = [dict(video_id="v", y_class=1, polarity="Neutral", y_reg=0., intensity=0.)]
        result = scores(statistics(rows, {"v": 0}).sum(0))
        np.testing.assert_allclose(result[:3], [1., 1/3, 0.])
        self.assertTrue(np.isnan(result[3]))


if __name__ == "__main__":
    unittest.main()
