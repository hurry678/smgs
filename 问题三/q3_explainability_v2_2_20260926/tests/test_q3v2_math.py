from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from q3v2_math import (  # noqa: E402
    clustered_bootstrap_mean,
    coalition_subsets,
    directional_intensity_delta,
    evidence_character_span,
    exact_loo,
    exact_pairwise_interactions,
    exact_shapley,
    primary_reference,
    symmetric_top_support_fidelity,
    validate_coalition_order,
)


class CoalitionTests(unittest.TestCase):
    def test_integer_bit_mask_order(self) -> None:
        self.assertEqual(
            coalition_subsets(),
            (
                (),
                (0,),
                (1,),
                (0, 1),
                (2,),
                (0, 2),
                (1, 2),
                (0, 1, 2),
            ),
        )

    def test_legacy_order_is_rejected(self) -> None:
        legacy = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
        with self.assertRaises(ValueError):
            validate_coalition_order(legacy)

    def test_additive_shapley_and_loo_oracle(self) -> None:
        weights = np.asarray([1.25, -0.5, 2.0])
        values = np.asarray(
            [
                [
                    3.0 + sum(weights[index] for index in subset)
                    for subset in coalition_subsets()
                ]
            ]
        )
        np.testing.assert_allclose(exact_shapley(values), weights[None, :])
        np.testing.assert_allclose(exact_loo(values), weights[None, :])

    def test_pair_interaction_oracle(self) -> None:
        values = []
        for subset in coalition_subsets():
            value = sum((1.0, 2.0, 3.0)[index] for index in subset)
            if 0 in subset and 2 in subset:
                value += 4.5
            values.append(value)
        interaction = exact_pairwise_interactions(np.asarray([values]))
        np.testing.assert_allclose(interaction, [[0.0, 4.5, 0.0]])

    def test_character_span_uses_token_axis_not_sample_count(self) -> None:
        offsets = np.zeros((50, 2), dtype=np.int64)
        valid = np.zeros(50, dtype=bool)
        offsets[25] = [7, 11]
        valid[25] = True
        self.assertEqual(evidence_character_span(offsets, valid, [25]), (7, 11))


class SemanticsTests(unittest.TestCase):
    def test_directional_intensity(self) -> None:
        self.assertEqual(directional_intensity_delta(2, 1.0, 0.4), 0.4)
        self.assertEqual(directional_intensity_delta(0, -1.0, -0.4), 0.4)
        self.assertAlmostEqual(
            directional_intensity_delta(1, 0.2, -0.5),
            0.5,
        )

    def test_primary_positive_support_and_absolute_fallback(self) -> None:
        self.assertEqual(
            primary_reference([-0.8, 0.2, 0.1]),
            (1, 0, "largest_positive_class_shapley"),
        )
        self.assertEqual(
            primary_reference([-0.8, -0.2, -0.1]),
            (0, 0, "no_positive_class_shapley_fallback_to_absolute"),
        )

    def test_group_bootstrap_is_deterministic(self) -> None:
        first = clustered_bootstrap_mean(
            [1.0, 2.0, -1.0, 0.5],
            ["video_a", "video_a", "video_b", "video_c"],
            seed=2026,
            n_resamples=200,
        )
        second = clustered_bootstrap_mean(
            [1.0, 2.0, -1.0, 0.5],
            ["video_a", "video_a", "video_b", "video_c"],
            seed=2026,
            n_resamples=200,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["n_groups"], 3)
        self.assertEqual(first["n_observations"], 4)

    def test_symmetric_top_control_removes_one_sided_selection_bias(self) -> None:
        records = []
        for sample_position in range(4):
            for window_index, (continuous, control) in enumerate(
                ((0.0, 1.0), (1.0, 0.0))
            ):
                for pattern, delta in (
                    ("continuous", continuous),
                    ("random_contiguous", control),
                    ("point_scatter", control),
                ):
                    records.append(
                        {
                            "sample_position": sample_position,
                            "global_index": sample_position,
                            "modality_index": 0,
                            "window_index": window_index,
                            "pattern": pattern,
                            "positions": [window_index],
                            "class_delta": delta,
                        }
                    )
        result = symmetric_top_support_fidelity(
            records,
            [f"video_{index}$_$0" for index in range(4)],
            seed=2026,
            n_resamples=200,
        )
        self.assertTrue(result["control_selection_symmetric"])
        self.assertEqual(result["mean_budget_normalized_top_support"], 1.0)
        for pattern in ("random_contiguous", "point_scatter"):
            comparison = result[f"top_support_vs_{pattern}"]
            self.assertEqual(comparison["mean_difference"], 0.0)
            self.assertEqual(comparison["bootstrap_95_ci"], [0.0, 0.0])

    def test_symmetric_top_uses_budget_normalized_ranking(self) -> None:
        records = []
        values = {
            "continuous": ((4.0, [0, 1, 2, 3]), (3.0, [4])),
            "random_contiguous": ((2.0, [0, 1]), (1.0, [2])),
            "point_scatter": ((1.0, [0]), (1.0, [1])),
        }
        for pattern, rows in values.items():
            for window_index, (delta, positions) in enumerate(rows):
                records.append(
                    {
                        "sample_position": 0,
                        "global_index": 0,
                        "modality_index": 0,
                        "window_index": window_index,
                        "pattern": pattern,
                        "positions": positions,
                        "class_delta": delta,
                    }
                )
        result = symmetric_top_support_fidelity(
            records,
            ["video$_$0"],
            seed=2026,
            n_resamples=20,
        )
        self.assertEqual(result["mean_budget_normalized_top_support"], 3.0)
        self.assertEqual(
            result["top_support_vs_random_contiguous"]["mean_difference"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
