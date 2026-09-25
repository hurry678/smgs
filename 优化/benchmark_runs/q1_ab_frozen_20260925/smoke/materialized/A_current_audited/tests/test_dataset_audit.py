import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_audit import statistics, correlation_table, modality_stats, workbook_audit
from alignment import visual_observation_state


class DataQualityTest(unittest.TestCase):
    def test_numeric_source_identifiers_are_valid_and_label_contradictions_are_not(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as directory, patch("dataset_audit.DATA", Path(directory)):
            path=Path(directory)/"labels.xlsx"
            book=openpyxl.Workbook()
            sheet=book.active
            sheet.title="label"
            sheet.append(["video_id","clip_id","text","label","annotation"])
            sheet.append(["100499","5","valid numeric identifier","1.0","Positive"])
            sheet.append(["source-b","0","wrong sign","-1.0","Positive"])
            book.save(path)
            book.close()
            rows,result=workbook_audit(path)
            self.assertEqual(len(rows),2)
            self.assertEqual(result["format_or_logic_errors"],
                             [{"id":"source-b$_$0","error":"label_range_or_sign_annotation"}])

    def test_statistics_exclude_missing_but_retain_valid_zero(self):
        result = statistics([0, 2, 4, np.nan, np.inf])
        self.assertEqual(result["n_valid"], 3)
        self.assertEqual(result["n_missing"], 2)
        self.assertEqual(result["mean"], 2)
        self.assertEqual(statistics([])["median"], None)

    def test_pairwise_correlation_reports_effective_n_and_constant_undefined(self):
        rows = [{"a": i, "b": i*2, "c": 1} for i in range(4)] + [{"a": np.nan, "b": 10, "c": 1}]
        result = correlation_table(rows, ["a", "b", "c"])
        self.assertEqual(result[0]["n"], 4)
        self.assertAlmostEqual(result[0]["pearson_r"], 1)
        self.assertIsNone(result[1]["pearson_r"])

    def test_zero_padding_is_not_observed_missing(self):
        x = np.array([[[1.,2.],[0.,0.],[0.,0.]]])
        stats, _ = modality_stats(x, np.array([[True,True,False]]), ["s"], False)
        self.assertEqual(stats["valid_positions"], 2)
        self.assertEqual(stats["zero_valid_positions"], 1)
        self.assertEqual(stats["nonzero_outside_valid_positions"], 0)

    def test_visual_missing_and_ambiguity_are_distinct(self):
        vi = {"valid": np.array([[False]*6,[False]*6,[True]*6,[True]*6]),
              "geometry_rejected": np.array([False,True,False,False]),
              "ambiguous": np.array([False,True,False,True])}
        self.assertEqual(visual_observation_state(vi, []), "sampling_no_coverage")
        for i, expected in enumerate(["no_reliable_face_detected","detected_geometry_rejected",
                                      "observed","observed_with_subject_ambiguity"]):
            self.assertEqual(visual_observation_state(vi, [i]), expected)


if __name__ == "__main__":
    unittest.main()
