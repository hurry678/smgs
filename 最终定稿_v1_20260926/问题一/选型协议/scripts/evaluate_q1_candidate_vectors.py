#!/usr/bin/env python3
"""Nested grouped linear probes for standardized Q1 candidate vectors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260924, 20260925, 20260926, 20260927, 20260928])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--views", nargs="+", choices=("text", "audio", "vision", "fused"), default=["text", "audio", "vision", "fused"])
    return parser.parse_args()


def metrics(y_class: np.ndarray, class_pred: np.ndarray, y_reg: np.ndarray, reg_pred: np.ndarray) -> dict[str, float]:
    if np.std(reg_pred) > 1e-12 and np.std(y_reg) > 1e-12:
        pearson = float(np.corrcoef(y_reg, reg_pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "accuracy": float(accuracy_score(y_class, class_pred)),
        "macro_f1": float(f1_score(y_class, class_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_reg, reg_pred)),
        "pearson": pearson,
    }


def split_features(data: Any, view: str) -> np.ndarray:
    if view == "fused":
        return np.concatenate([data["x_text"], data["x_audio"], data["x_vision"], data["valid_rates"]], axis=1)
    index = {"text": 0, "audio": 1, "vision": 2}[view]
    return np.concatenate([data[f"x_{view}"], data["valid_rates"][:, index:index + 1]], axis=1)


def fit_classification(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c: float) -> np.ndarray:
    classes = np.unique(y_train)
    if len(classes) == 1:
        return np.full(len(x_test), classes[0], dtype=object)
    model = LogisticRegression(
        C=c, max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=20260924
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def choose_hyperparameters(
    x: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    groups: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[float, float]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y_class, groups))
    class_scores: dict[float, list[float]] = {value: [] for value in (0.01, 0.1, 1.0, 10.0)}
    reg_scores: dict[float, list[float]] = {value: [] for value in (0.1, 1.0, 10.0, 100.0)}
    for train, valid in splits:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_valid = scaler.transform(x[valid])
        for c in class_scores:
            prediction = fit_classification(x_train, y_class[train], x_valid, c)
            class_scores[c].append(float(f1_score(y_class[valid], prediction, average="macro", zero_division=0)))
        for alpha in reg_scores:
            model = Ridge(alpha=alpha)
            model.fit(x_train, y_reg[train])
            reg_scores[alpha].append(float(mean_absolute_error(y_reg[valid], model.predict(x_valid))))
    best_c = max(class_scores, key=lambda value: (np.mean(class_scores[value]), -value))
    best_alpha = min(reg_scores, key=lambda value: (np.mean(reg_scores[value]), value))
    return float(best_c), float(best_alpha)


def evaluate_candidate(
    candidate: str,
    data: Any,
    view: str,
    seeds: list[int],
    outer_folds: int,
    inner_folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x = np.nan_to_num(split_features(data, view).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sample_ids = data["sample_id"].astype(str)
    groups = data["video_id"].astype(str)
    y_class = data["y_class"].astype(str)
    y_reg = data["y_reg"].astype(np.float64)
    predictions: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        class_pred = np.empty(len(y_class), dtype=object)
        reg_pred = np.full(len(y_reg), np.nan, dtype=np.float64)
        fold_ids = np.full(len(y_reg), -1, dtype=np.int32)
        selected: list[dict[str, Any]] = []
        for fold, (train, test) in enumerate(outer.split(x, y_class, groups)):
            best_c, best_alpha = choose_hyperparameters(
                x[train], y_class[train], y_reg[train], groups[train],
                seed + 1000 + fold, inner_folds,
            )
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train])
            x_test = scaler.transform(x[test])
            class_pred[test] = fit_classification(x_train, y_class[train], x_test, best_c)
            model = Ridge(alpha=best_alpha)
            model.fit(x_train, y_reg[train])
            reg_pred[test] = model.predict(x_test)
            fold_ids[test] = fold
            selected.append({"fold": fold, "C": best_c, "alpha": best_alpha})
        if np.any(fold_ids < 0) or not np.isfinite(reg_pred).all():
            raise RuntimeError(f"Incomplete outer predictions: {candidate}/{view}/{seed}")
        result = metrics(y_class, class_pred, y_reg, reg_pred)
        run_metrics.append({
            "candidate": candidate, "view": view, "seed": seed,
            "samples": len(y_class), "groups": len(np.unique(groups)), "feature_dim": x.shape[1],
            **result, "selected_hyperparameters": json.dumps(selected, separators=(",", ":")),
        })
        for index, sample_id in enumerate(sample_ids):
            predictions.append({
                "candidate": candidate, "view": view, "seed": seed,
                "fold": int(fold_ids[index]), "sample_id": sample_id, "video_id": groups[index],
                "y_class": y_class[index], "pred_class": class_pred[index],
                "y_reg": float(y_reg[index]), "pred_reg": float(reg_pred[index]),
            })
    return predictions, run_metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.candidate_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No candidate vectors in {args.candidate_dir}")
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    identity: tuple[list[str], list[str], list[str], list[float]] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        current_identity = (
            data["sample_id"].astype(str).tolist(),
            data["video_id"].astype(str).tolist(),
            data["y_class"].astype(str).tolist(),
            data["y_reg"].astype(float).tolist(),
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"Candidate identity/labels differ: {path}")
        for view in args.views:
            predictions, results = evaluate_candidate(
                path.stem, data, view, args.seeds, args.outer_folds, args.inner_folds
            )
            prediction_rows.extend(predictions)
            metric_rows.extend(results)
    write_csv(args.output_dir / "probe_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "probe_metrics.csv", metric_rows)
    summary: dict[str, Any] = {
        "schema": "q1-nested-grouped-probe-1",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seeds": args.seeds,
        "group_key": "video_id",
        "standardization": "fit on each outer/inner training fold only",
        "classification_selection": "inner-fold macro-F1",
        "regression_selection": "inner-fold MAE",
        "results": [],
    }
    for candidate in sorted({row["candidate"] for row in metric_rows}):
        for view in args.views:
            selected = [row for row in metric_rows if row["candidate"] == candidate and row["view"] == view]
            record: dict[str, Any] = {"candidate": candidate, "view": view}
            for name in ("accuracy", "macro_f1", "mae", "pearson"):
                values = np.asarray([row[name] for row in selected], dtype=float)
                record[name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
            summary["results"].append(record)
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": len(paths), "metric_rows": len(metric_rows),
                      "prediction_rows": len(prediction_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
