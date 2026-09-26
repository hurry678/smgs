#!/usr/bin/env python3
"""Pure NumPy utilities for the Q3 v2.2 explanation protocol."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "vision")
PAIR_NAMES = ("text_audio", "text_vision", "audio_vision")


def coalition_subsets(n_players: int = 3) -> tuple[tuple[int, ...], ...]:
    """Return coalitions in integer bit-mask order."""
    return tuple(
        tuple(i for i in range(n_players) if mask & (1 << i))
        for mask in range(1 << n_players)
    )


def validate_coalition_order(
    subsets: Sequence[Sequence[int]], n_players: int = 3
) -> None:
    expected = coalition_subsets(n_players)
    observed = tuple(tuple(int(i) for i in subset) for subset in subsets)
    if observed != expected:
        raise ValueError(
            "coalitions must be stored in integer bit-mask order: "
            f"expected={expected}, observed={observed}"
        )


def exact_shapley(values: np.ndarray, n_players: int = 3) -> np.ndarray:
    """Compute exact Shapley values from columns indexed by coalition bit mask."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1 << n_players:
        raise ValueError(
            f"values must have shape [n,{1 << n_players}], got {values.shape}"
        )
    phi = np.zeros((values.shape[0], n_players), dtype=np.float64)
    denominator = math.factorial(n_players)
    for player in range(n_players):
        for mask in range(1 << n_players):
            if mask & (1 << player):
                continue
            size = int(bin(mask).count("1"))
            weight = (
                math.factorial(size)
                * math.factorial(n_players - size - 1)
                / denominator
            )
            phi[:, player] += weight * (
                values[:, mask | (1 << player)] - values[:, mask]
            )
    return phi.astype(np.float32)


def exact_loo(values: np.ndarray, n_players: int = 3) -> np.ndarray:
    """Compute leave-one-modality-out effects from bit-mask ordered values."""
    values = np.asarray(values, dtype=np.float64)
    full_mask = (1 << n_players) - 1
    if values.ndim != 2 or values.shape[1] != full_mask + 1:
        raise ValueError(
            f"values must have shape [n,{full_mask + 1}], got {values.shape}"
        )
    return np.stack(
        [
            values[:, full_mask] - values[:, full_mask ^ (1 << player)]
            for player in range(n_players)
        ],
        axis=1,
    ).astype(np.float32)


def exact_pairwise_interactions(values: np.ndarray, n_players: int = 3) -> np.ndarray:
    """Return exact Shapley pair interactions for all unordered player pairs."""
    if n_players != 3:
        raise ValueError("Q3 v2.2 currently supports exactly three modalities")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"values must have shape [n,8], got {values.shape}")
    columns = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        remaining = next(i for i in range(3) if i not in (left, right))
        base = (
            values[:, (1 << left) | (1 << right)]
            - values[:, 1 << left]
            - values[:, 1 << right]
            + values[:, 0]
        )
        with_remaining = (
            values[:, 7]
            - values[:, (1 << left) | (1 << remaining)]
            - values[:, (1 << right) | (1 << remaining)]
            + values[:, 1 << remaining]
        )
        columns.append(0.5 * (base + with_remaining))
    return np.stack(columns, axis=1).astype(np.float32)


def directional_intensity_delta(
    predicted_class: int, full_raw: float, raw_delta: float
) -> float:
    """Orient a raw-intensity deletion delta toward the published class."""
    if predicted_class == 2:
        return float(raw_delta)
    if predicted_class == 0:
        return float(-raw_delta)
    perturbed_raw = float(full_raw) - float(raw_delta)
    return abs(perturbed_raw) - abs(float(full_raw))


def primary_reference(
    class_phi: Sequence[float],
) -> tuple[int, int, str]:
    """Separate the main supporting modality from dominant absolute influence."""
    values = np.asarray(class_phi, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"class_phi must have shape [3], got {values.shape}")
    dominant = int(np.argmax(np.abs(values)))
    support = int(np.argmax(values))
    if values[support] > 0.0:
        return support, dominant, "largest_positive_class_shapley"
    return dominant, dominant, "no_positive_class_shapley_fallback_to_absolute"


def evidence_character_span(
    offsets: np.ndarray,
    valid: np.ndarray,
    positions: Sequence[int],
) -> tuple[int, int] | None:
    """Return the union of valid token character offsets for one evidence window."""
    offset_array = np.asarray(offsets, dtype=np.int64)
    valid_array = np.asarray(valid, dtype=bool)
    if offset_array.ndim != 2 or offset_array.shape[1] != 2:
        raise ValueError(
            f"offsets must have shape [tokens,2], got {offset_array.shape}"
        )
    if valid_array.shape != (offset_array.shape[0],):
        raise ValueError("valid mask shape does not match offsets")
    spans = [
        (int(offset_array[position, 0]), int(offset_array[position, 1]))
        for position in positions
        if 0 <= int(position) < offset_array.shape[0]
        and valid_array[int(position)]
        and int(offset_array[int(position), 1]) > int(offset_array[int(position), 0])
    ]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def deterministic_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def clustered_bootstrap_mean(
    differences: Sequence[float],
    groups: Sequence[str],
    seed: int,
    n_resamples: int = 5000,
) -> Mapping[str, Any]:
    """Bootstrap a paired mean by resampling independent video groups."""
    diff = np.asarray(differences, dtype=np.float64)
    group_array = np.asarray(groups, dtype=str)
    if diff.shape != group_array.shape:
        raise ValueError("differences and groups must have identical shape")
    if not len(diff):
        return {
            "n_observations": 0,
            "n_groups": 0,
            "mean_difference": 0.0,
            "bootstrap_95_ci": [0.0, 0.0],
        }
    unique_groups = np.unique(group_array)
    members = [np.flatnonzero(group_array == group) for group in unique_groups]
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_resamples), dtype=np.float64)
    for index in range(int(n_resamples)):
        selected = rng.integers(0, len(members), size=len(members))
        sampled = np.concatenate([members[int(i)] for i in selected])
        boot[index] = float(np.mean(diff[sampled]))
    return {
        "n_observations": int(len(diff)),
        "n_groups": int(len(unique_groups)),
        "mean_difference": float(np.mean(diff)),
        "bootstrap_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "resamples": int(n_resamples),
        "seed": int(seed),
    }


def _video_group(sample_id: str) -> str:
    return str(sample_id).split("$_$", 1)[0]


def _budget_normalized_support(row: Mapping[str, Any]) -> float:
    budget = len(row.get("positions", ()))
    if budget <= 0:
        raise ValueError("every local-occlusion record must mask at least one position")
    return float(row["class_delta"]) / float(budget)


def symmetric_top_support_fidelity(
    records: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
    seed: int,
    n_resamples: int = 5000,
) -> Mapping[str, Any]:
    """Compare independently selected top evidence under equal search budgets."""
    patterns = ("continuous", "random_contiguous", "point_scatter")
    grouped: dict[tuple[int, int], dict[str, list[Mapping[str, Any]]]] = {}
    for row in records:
        pattern = str(row["pattern"])
        if pattern not in patterns:
            continue
        key = (int(row["sample_position"]), int(row["modality_index"]))
        grouped.setdefault(key, {name: [] for name in patterns})[pattern].append(row)

    top_scores: dict[str, list[float]] = {name: [] for name in patterns}
    differences: dict[str, list[float]] = {
        "random_contiguous": [],
        "point_scatter": [],
    }
    groups: list[str] = []
    for key in sorted(grouped):
        pattern_rows = grouped[key]
        counts = {pattern: len(pattern_rows[pattern]) for pattern in patterns}
        if not counts["continuous"] or len(set(counts.values())) != 1:
            raise ValueError(
                "continuous and control patterns must have equal non-zero search counts: "
                f"key={key}, counts={counts}"
            )
        selected: dict[str, Mapping[str, Any]] = {}
        scores: dict[str, float] = {}
        for pattern in patterns:
            selected[pattern] = max(
                pattern_rows[pattern],
                key=lambda row: (
                    _budget_normalized_support(row),
                    -int(row["window_index"]),
                ),
            )
            scores[pattern] = _budget_normalized_support(selected[pattern])
            top_scores[pattern].append(scores[pattern])

        global_indices = {
            int(selected[pattern]["global_index"]) for pattern in patterns
        }
        if len(global_indices) != 1:
            raise ValueError(f"control sample mismatch for key={key}")
        global_index = global_indices.pop()
        if not 0 <= global_index < len(sample_ids):
            raise ValueError(f"global sample index out of range: {global_index}")
        groups.append(_video_group(sample_ids[global_index]))
        for pattern in differences:
            differences[pattern].append(scores["continuous"] - scores[pattern])

    output: dict[str, Any] = {
        "comparison_schema": "q3v2.2-symmetric-top-support-v1",
        "comparison_definition": (
            "independent per-pattern maximum of signed class delta divided by "
            "the actual masked-position budget"
        ),
        "control_selection_symmetric": True,
        "n_sample_modalities": len(groups),
        "mean_budget_normalized_top_support": (
            float(np.mean(top_scores["continuous"])) if groups else 0.0
        ),
        "mean_budget_normalized_top_random_contiguous": (
            float(np.mean(top_scores["random_contiguous"])) if groups else 0.0
        ),
        "mean_budget_normalized_top_point_scatter": (
            float(np.mean(top_scores["point_scatter"])) if groups else 0.0
        ),
    }
    for offset, (pattern, values) in enumerate(differences.items()):
        output[f"top_support_vs_{pattern}"] = clustered_bootstrap_mean(
            values,
            groups,
            seed=seed + 31001 + offset,
            n_resamples=n_resamples,
        )
    return output
