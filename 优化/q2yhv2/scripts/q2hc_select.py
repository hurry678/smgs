#!/usr/bin/env python3
"""Validation-only ensemble selection and independent confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _stable_unit(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def group_key(sample_id: str, separator: str = "$_$") -> str:
    return sample_id.split(separator, 1)[0]


def fold_indices(
    ids: Sequence[str], folds: int, fold: int, seed: int, separator: str
) -> Tuple[np.ndarray, np.ndarray]:
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError(f"invalid fold {fold}/{folds}")
    assigned = np.asarray(
        [
            min(
                folds - 1,
                int(_stable_unit(f"{seed}|fold|{group_key(sid, separator)}") * folds),
            )
            for sid in ids
        ]
    )
    holdout = np.flatnonzero(assigned == fold)
    train = np.flatnonzero(assigned != fold)
    if len(train) == 0 or len(holdout) == 0:
        raise RuntimeError("grouped fold assignment produced an empty partition")
    train_groups = {group_key(ids[i], separator) for i in train}
    holdout_groups = {group_key(ids[i], separator) for i in holdout}
    if train_groups & holdout_groups:
        raise RuntimeError("group leakage detected in cross-validation split")
    return train, holdout


def selector_indices(
    ids: Sequence[str], seed: int, fraction: float, separator: str
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("selector fraction must lie in (0, 1)")
    selector = np.asarray(
        [
            _stable_unit(f"{seed}|selector|{group_key(sid, separator)}") < fraction
            for sid in ids
        ]
    )
    left = np.flatnonzero(selector)
    right = np.flatnonzero(~selector)
    if len(left) == 0 or len(right) == 0:
        raise RuntimeError("selector/confirmation split produced an empty partition")
    left_groups = {group_key(ids[i], separator) for i in left}
    right_groups = {group_key(ids[i], separator) for i in right}
    if left_groups & right_groups:
        raise RuntimeError("group leakage detected between selector and confirmation")
    return left, right


def _macro_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    for cls in range(3):
        true_positive = int(np.sum((labels == cls) & (predicted == cls)))
        false_positive = int(np.sum((labels != cls) & (predicted == cls)))
        false_negative = int(np.sum((labels == cls) & (predicted != cls)))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(values))


def _metric_block(
    probabilities: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    regression: np.ndarray,
) -> Dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    intensity = np.zeros_like(raw, dtype=np.float32)
    intensity[predicted == 2] = np.maximum(raw[predicted == 2], 1e-4)
    intensity[predicted == 0] = np.minimum(raw[predicted == 0], -1e-4)
    if len(labels) > 1 and np.std(intensity) > 0 and np.std(regression) > 0:
        pearson = float(np.corrcoef(regression, intensity)[0, 1])
    else:
        pearson = float("nan")
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "macro_f1": _macro_f1(labels, predicted),
        "mae": float(np.mean(np.abs(intensity - regression))),
        "pearson": pearson,
        "n": int(len(labels)),
    }


def load_cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "member_id": str(archive["member_id"].item()),
            "checkpoint": str(archive["checkpoint"].item()),
            "checkpoint_sha256": str(archive["checkpoint_sha256"].item()),
            "ids": [str(value) for value in archive["ids"].tolist()],
            "condition_ids": [
                str(value) for value in archive["condition_ids"].tolist()
            ],
            "probabilities": np.asarray(archive["probabilities"], dtype=np.float32),
            "raw": np.asarray(archive["raw"], dtype=np.float32),
            "cache": str(path.resolve()),
        }


def load_caches(
    paths: Iterable[Path],
) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    members: Dict[str, Dict[str, Any]] = {}
    reference_ids: List[str] | None = None
    reference_conditions: List[str] | None = None
    for path in paths:
        cache = load_cache(path)
        member_id = cache["member_id"]
        if member_id in members:
            raise RuntimeError(f"duplicate member cache: {member_id}")
        if reference_ids is None:
            reference_ids = cache["ids"]
            reference_conditions = cache["condition_ids"]
        if cache["ids"] != reference_ids:
            raise RuntimeError(f"sample ID mismatch in {path}")
        if cache["condition_ids"] != reference_conditions:
            raise RuntimeError(f"condition mismatch in {path}")
        members[member_id] = cache
    if not members or reference_ids is None or reference_conditions is None:
        raise RuntimeError("no member caches were supplied")
    return reference_ids, reference_conditions, members


def ensemble_arrays(
    members: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("ensemble weights must have positive sum")
    probabilities = None
    raw = None
    for member_id, weight in weights.items():
        if weight <= 0:
            continue
        member = members[member_id]
        scale = float(weight) / total
        if probabilities is None:
            probabilities = scale * member["probabilities"].astype(np.float64)
            raw = scale * member["raw"].astype(np.float64)
        else:
            probabilities += scale * member["probabilities"]
            raw += scale * member["raw"]
    if probabilities is None or raw is None:
        raise ValueError("ensemble has no active members")
    return probabilities.astype(np.float32), raw.astype(np.float32)


def summarize(
    probabilities: np.ndarray,
    raw: np.ndarray,
    condition_ids: Sequence[str],
    labels: np.ndarray,
    regression: np.ndarray,
    sample_indices: np.ndarray,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    clean: Dict[str, float] | None = None
    cells: Dict[str, List[Dict[str, float]]] = {}
    for condition_index, condition_id in enumerate(condition_ids):
        block = _metric_block(
            probabilities[condition_index, sample_indices],
            raw[condition_index, sample_indices],
            labels[sample_indices],
            regression[sample_indices],
        )
        if condition_id == "clean":
            clean = block
            continue
        parts = condition_id.split("|")
        if len(parts) != 3:
            raise RuntimeError(f"invalid condition ID: {condition_id}")
        cell = "|".join(parts[:2])
        cells.setdefault(cell, []).append(block)
        rows.append({"condition_id": condition_id, **block})
    if clean is None:
        raise RuntimeError("prediction cache does not contain clean condition")
    cell_rows = []
    for cell, blocks in sorted(cells.items()):
        cell_rows.append(
            {
                "cell": cell,
                "replicas": len(blocks),
                "mae": float(np.mean([block["mae"] for block in blocks])),
                "macro_f1": float(np.mean([block["macro_f1"] for block in blocks])),
                "accuracy": float(np.mean([block["accuracy"] for block in blocks])),
            }
        )
    return {
        "n": int(len(sample_indices)),
        "clean": clean,
        "R_MAE": float(np.mean([row["mae"] for row in cell_rows])),
        "R_F1": float(np.mean([row["macro_f1"] for row in cell_rows])),
        "R_accuracy": float(np.mean([row["accuracy"] for row in cell_rows])),
        "worst_condition_MAE": float(max(row["mae"] for row in cell_rows)),
        "worst_condition_F1": float(min(row["macro_f1"] for row in cell_rows)),
        "cells": cell_rows,
        "conditions": rows,
    }


def objective(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    weights: Mapping[str, float],
) -> float:
    terms = {
        "R_MAE": float(summary["R_MAE"]) / max(1e-12, float(baseline["R_MAE"])),
        "one_minus_R_F1": (1.0 - float(summary["R_F1"]))
        / max(1e-12, 1.0 - float(baseline["R_F1"])),
        "clean_MAE": float(summary["clean"]["mae"])
        / max(1e-12, float(baseline["clean"]["mae"])),
        "one_minus_clean_F1": (
            (1.0 - float(summary["clean"]["macro_f1"]))
            / max(1e-12, 1.0 - float(baseline["clean"]["macro_f1"]))
        ),
        "worst_condition_MAE": (
            float(summary["worst_condition_MAE"])
            / max(1e-12, float(baseline["worst_condition_MAE"]))
        ),
    }
    return float(sum(float(weights[name]) * value for name, value in terms.items()))


def selector_gates(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gates: Mapping[str, float],
) -> Tuple[bool, Dict[str, float]]:
    deltas = {
        "R_MAE_increase": float(candidate["R_MAE"]) - float(baseline["R_MAE"]),
        "R_F1_drop": float(baseline["R_F1"]) - float(candidate["R_F1"]),
        "clean_MAE_increase": float(candidate["clean"]["mae"])
        - float(baseline["clean"]["mae"]),
        "clean_F1_drop": float(baseline["clean"]["macro_f1"])
        - float(candidate["clean"]["macro_f1"]),
        "worst_condition_MAE_increase": (
            float(candidate["worst_condition_MAE"])
            - float(baseline["worst_condition_MAE"])
        ),
    }
    passed = (
        deltas["R_MAE_increase"] <= float(gates["max_R_MAE_increase"])
        and deltas["R_F1_drop"] <= float(gates["max_R_F1_drop"])
        and deltas["clean_MAE_increase"] <= float(gates["max_clean_MAE_increase"])
        and deltas["clean_F1_drop"] <= float(gates["max_clean_F1_drop"])
        and deltas["worst_condition_MAE_increase"]
        <= float(gates["max_worst_condition_MAE_increase"])
    )
    return bool(passed), deltas


def confirmation_gates(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gates: Mapping[str, float],
) -> Tuple[bool, Dict[str, float]]:
    deltas = {
        "R_MAE_improvement": float(baseline["R_MAE"]) - float(candidate["R_MAE"]),
        "R_F1_drop": float(baseline["R_F1"]) - float(candidate["R_F1"]),
        "clean_MAE_increase": float(candidate["clean"]["mae"])
        - float(baseline["clean"]["mae"]),
        "clean_F1_drop": float(baseline["clean"]["macro_f1"])
        - float(candidate["clean"]["macro_f1"]),
        "worst_condition_MAE_increase": (
            float(candidate["worst_condition_MAE"])
            - float(baseline["worst_condition_MAE"])
        ),
    }
    passed = (
        deltas["R_MAE_improvement"] >= float(gates["min_R_MAE_improvement"])
        and deltas["R_F1_drop"] <= float(gates["max_R_F1_drop"])
        and deltas["clean_MAE_increase"] <= float(gates["max_clean_MAE_increase"])
        and deltas["clean_F1_drop"] <= float(gates["max_clean_F1_drop"])
        and deltas["worst_condition_MAE_increase"]
        <= float(gates["max_worst_condition_MAE_increase"])
    )
    return bool(passed), deltas


def _counts_to_weights(
    member_ids: Sequence[str], counts: Tuple[int, ...]
) -> Dict[str, float]:
    total = sum(counts)
    return {
        member_id: count / total
        for member_id, count in zip(member_ids, counts)
        if count > 0
    }


def _valid_counts(counts: Tuple[int, ...], cfg: Mapping[str, Any]) -> bool:
    slots = sum(counts)
    if slots < int(cfg["min_slots"]) or slots > int(cfg["max_slots"]):
        return False
    active = sum(count > 0 for count in counts)
    if active < int(cfg["min_slots"]):
        return False
    return max(counts, default=0) / slots <= float(cfg["max_member_weight"]) + 1e-12


def select_ensemble(
    config: Mapping[str, Any],
    cache_paths: Iterable[Path],
    data_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    ids, condition_ids, members = load_caches(cache_paths)
    with np.load(data_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["valid_labels"], dtype=np.int64)
        regression = np.asarray(archive["valid_regression"], dtype=np.float32)
        data_ids = [str(value) for value in archive["valid_ids"].tolist()]
    if ids != data_ids:
        raise RuntimeError("cache IDs do not match validation data")
    search_cfg = config["search"]
    selection_cfg = config["selection"]
    selector, confirmer = selector_indices(
        ids,
        int(selection_cfg["sample_split_seed"]),
        float(selection_cfg["selector_fraction"]),
        str(search_cfg["group_separator"]),
    )
    baseline_ids = [item["id"] for item in config["baseline"]["checkpoints"]]
    missing_baseline = [
        member_id for member_id in baseline_ids if member_id not in members
    ]
    if missing_baseline:
        raise RuntimeError(f"missing baseline member caches: {missing_baseline}")
    baseline_weights = {
        member_id: 1.0 / len(baseline_ids) for member_id in baseline_ids
    }
    baseline_arrays = ensemble_arrays(members, baseline_weights)
    baseline_summary = summarize(
        *baseline_arrays,
        condition_ids,
        labels,
        regression,
        selector,
    )
    objective_weights = selection_cfg["objective_weights"]
    baseline_objective = objective(
        baseline_summary, baseline_summary, objective_weights
    )

    individual_rows = []
    for member_id, member in members.items():
        summary = summarize(
            member["probabilities"],
            member["raw"],
            condition_ids,
            labels,
            regression,
            selector,
        )
        individual_rows.append(
            {
                "member_id": member_id,
                "objective": objective(summary, baseline_summary, objective_weights),
                "summary": summary,
            }
        )
    individual_rows.sort(key=lambda row: (row["objective"], row["member_id"]))
    screened = set(baseline_ids)
    screened.update(
        row["member_id"]
        for row in individual_rows[
            : int(selection_cfg["max_pool_after_individual_screen"])
        ]
    )
    member_ids = sorted(screened)
    current = tuple(1 if member_id in baseline_ids else 0 for member_id in member_ids)
    memo: Dict[
        Tuple[int, ...], Tuple[float, Dict[str, Any], bool, Dict[str, float]]
    ] = {}

    def evaluate_counts(counts: Tuple[int, ...]):
        if counts not in memo:
            weights = _counts_to_weights(member_ids, counts)
            arrays = ensemble_arrays(members, weights)
            summary = summarize(*arrays, condition_ids, labels, regression, selector)
            score = objective(summary, baseline_summary, objective_weights)
            passed, deltas = selector_gates(
                summary,
                baseline_summary,
                selection_cfg["gates_vs_baseline"],
            )
            memo[counts] = score, summary, passed, deltas
        return memo[counts]

    current_score, _, _, _ = evaluate_counts(current)
    trajectory = [
        {
            "round": 0,
            "operation": "baseline",
            "objective": current_score,
            "weights": _counts_to_weights(member_ids, current),
        }
    ]
    best_passing = current
    best_passing_score = current_score
    visited = {current}

    for round_index in range(1, int(selection_cfg["max_coordinate_rounds"]) + 1):
        neighbors: Dict[Tuple[int, ...], str] = {}
        slots = sum(current)
        if slots < int(selection_cfg["max_slots"]):
            for add_index, member_id in enumerate(member_ids):
                candidate = list(current)
                candidate[add_index] += 1
                state = tuple(candidate)
                if _valid_counts(state, selection_cfg):
                    neighbors[state] = f"add:{member_id}"
        if slots > int(selection_cfg["min_slots"]):
            for remove_index, member_id in enumerate(member_ids):
                if current[remove_index] == 0:
                    continue
                candidate = list(current)
                candidate[remove_index] -= 1
                state = tuple(candidate)
                if _valid_counts(state, selection_cfg):
                    neighbors[state] = f"remove:{member_id}"
        active = [i for i, count in enumerate(current) if count > 0]
        for remove_index in active:
            for add_index, member_id in enumerate(member_ids):
                if add_index == remove_index:
                    continue
                candidate = list(current)
                candidate[remove_index] -= 1
                candidate[add_index] += 1
                state = tuple(candidate)
                if _valid_counts(state, selection_cfg):
                    neighbors[state] = f"swap:{member_ids[remove_index]}->{member_id}"
        ranked = []
        for state, operation in neighbors.items():
            if state in visited:
                continue
            score, summary, passed, deltas = evaluate_counts(state)
            ranked.append((score, state, operation, summary, passed, deltas))
            if passed and score < best_passing_score:
                best_passing = state
                best_passing_score = score
        if not ranked:
            break
        ranked.sort(key=lambda row: (row[0], row[1]))
        score, state, operation, _, _, _ = ranked[0]
        if score > current_score - float(selection_cfg["min_objective_improvement"]):
            break
        current = state
        current_score = score
        visited.add(state)
        trajectory.append(
            {
                "round": round_index,
                "operation": operation,
                "objective": score,
                "weights": _counts_to_weights(member_ids, state),
            }
        )

    selected = best_passing
    selected_score, selected_summary, passed, deltas = evaluate_counts(selected)
    selected_weights = _counts_to_weights(member_ids, selected)
    member_records = {
        member_id: {
            "checkpoint": members[member_id]["checkpoint"],
            "checkpoint_sha256": members[member_id]["checkpoint_sha256"],
            "cache": members[member_id]["cache"],
        }
        for member_id in selected_weights
    }
    result = {
        "schema": "q2-ds-hc-selection-v1",
        "status": "candidate_selected_pending_confirmation",
        "partition": {
            "selector_n": int(len(selector)),
            "confirmation_n": int(len(confirmer)),
            "selector_indices_sha256": __import__("hashlib")
            .sha256(selector.tobytes())
            .hexdigest(),
            "group_separated": True,
        },
        "baseline": {
            "weights": baseline_weights,
            "objective": baseline_objective,
            "summary": baseline_summary,
        },
        "selected": {
            "weights": selected_weights,
            "objective": selected_score,
            "summary": selected_summary,
            "passes_selector_gates": passed,
            "deltas_vs_baseline": deltas,
            "members": member_records,
        },
        "screened_member_ids": member_ids,
        "individual_ranking": individual_rows,
        "trajectory": trajectory,
        "rules": selection_cfg,
        "prohibitions": [
            "confirmation partition was not used for candidate ranking",
            "official test was not read",
            "attachment 3 was not read",
        ],
    }
    write_json(output_path, result)
    return result


def confirm_ensemble(
    config: Mapping[str, Any],
    cache_paths: Iterable[Path],
    data_path: Path,
    selection_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    ids, condition_ids, members = load_caches(cache_paths)
    selection = read_json(selection_path)
    with np.load(data_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["valid_labels"], dtype=np.int64)
        regression = np.asarray(archive["valid_regression"], dtype=np.float32)
        data_ids = [str(value) for value in archive["valid_ids"].tolist()]
    if ids != data_ids:
        raise RuntimeError("cache IDs do not match validation data")
    selector, confirmer = selector_indices(
        ids,
        int(config["selection"]["sample_split_seed"]),
        float(config["selection"]["selector_fraction"]),
        str(config["search"]["group_separator"]),
    )
    baseline_weights = selection["baseline"]["weights"]
    candidate_weights = selection["selected"]["weights"]
    required = set(baseline_weights) | set(candidate_weights)
    missing = sorted(required - set(members))
    if missing:
        raise RuntimeError(f"confirmation caches are missing members: {missing}")
    baseline_arrays = ensemble_arrays(members, baseline_weights)
    candidate_arrays = ensemble_arrays(members, candidate_weights)
    baseline_summary = summarize(
        *baseline_arrays,
        condition_ids,
        labels,
        regression,
        confirmer,
    )
    candidate_summary = summarize(
        *candidate_arrays,
        condition_ids,
        labels,
        regression,
        confirmer,
    )
    passed, deltas = confirmation_gates(
        candidate_summary,
        baseline_summary,
        config["confirmation"]["gates_vs_baseline"],
    )
    baseline_all = summarize(
        *baseline_arrays,
        condition_ids,
        labels,
        regression,
        np.arange(len(ids)),
    )
    candidate_all = summarize(
        *candidate_arrays,
        condition_ids,
        labels,
        regression,
        np.arange(len(ids)),
    )
    result = {
        "schema": "q2-ds-hc-confirmation-v1",
        "status": "candidate_passed"
        if passed
        else "candidate_rejected_baseline_retained",
        "passed": bool(passed),
        "partition": {
            "confirmation_n": int(len(confirmer)),
            "selector_n_not_read_for_gate": int(len(selector)),
            "confirmation_indices_sha256": __import__("hashlib")
            .sha256(confirmer.tobytes())
            .hexdigest(),
            "group_separated": True,
        },
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "deltas_vs_baseline": deltas,
        "gates": config["confirmation"]["gates_vs_baseline"],
        "full_valid_descriptive_after_gate": {
            "candidate": candidate_all,
            "baseline": baseline_all,
        },
        "frozen_choice": "candidate" if passed else "baseline",
        "prohibitions": [
            "confirmation result may only accept the preselected candidate or retain baseline",
            "confirmation cannot change ensemble weights",
            "official test was not read",
            "attachment 3 was not read",
        ],
    }
    write_json(output_path, result)
    return result


if __name__ == "__main__":
    raise SystemExit("Use run_q2hc.py; this module is not a standalone command.")
