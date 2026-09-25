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


if __name__ == "__main__":
    unittest.main()
