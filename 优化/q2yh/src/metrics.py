"""Metrics used for candidate selection (server task T03)."""
from __future__ import annotations

import numpy as np

CLASSES = ("Negative", "Neutral", "Positive")


def confusion(y, p, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, q in zip(np.asarray(y, dtype=int), np.asarray(p, dtype=int)):
        cm[t, q] += 1
    return cm


def macro_f1(y, p, n_classes=3):
    cm = confusion(y, p, n_classes)
    f1 = []
    for k in range(n_classes):
        tp = cm[k, k]
        denom = cm[k, :].sum() + cm[:, k].sum()
        f1.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1)), [float(x) for x in f1]


def accuracy(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    return float((y == p).mean()) if len(y) else float("nan")


def mae(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.abs(y - p).mean()) if len(y) else float("nan")


def pearson(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < 2:
        return None
    sy, sp = y.std(), p.std()
    if sy <= 1e-12 or sp <= 1e-12:
        return None
    return float(np.corrcoef(y, p)[0, 1])


def metric_block(y_class, y_reg, logits, raw, published_class, published_intensity):
    """Four headline metrics plus the supporting evidence the protocol asks for."""
    prob = softmax(logits)
    acc = accuracy(y_class, published_class)
    f1, per_class = macro_f1(y_class, published_class)
    block = {
        "accuracy": acc,
        "macro_f1": f1,
        "mae": mae(y_reg, published_intensity),
        "pearson": pearson(y_reg, published_intensity),
        "raw_mae": mae(y_reg, raw),
        "raw_pearson": pearson(y_reg, raw),
        "per_class_f1": {"Negative": per_class[0], "Neutral": per_class[1], "Positive": per_class[2]},
        "support": {c: int((np.asarray(y_class) == i).sum()) for i, c in enumerate(CLASSES)},
        "confusion": confusion(y_class, published_class).tolist(),
        "raw_sign_conflict_rate": float(np.mean(
            np.where(np.asarray(raw) < 0, 0, np.where(np.asarray(raw) > 0, 2, 1))
            != np.asarray(published_class))) if len(raw) else float("nan"),
        "published_sign_conflict_rate": 0.0,
        "mean_projection_magnitude": float(np.mean(np.abs(
            np.asarray(published_intensity) - np.asarray(raw)))),
        "argmax_agreement_with_published": float(np.mean(
            prob.argmax(1) == np.asarray(published_class))),
    }
    return block


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_condition_matrix(values_by_condition, specs):
    """Mean over replicas within each (subset, ratio) cell, then equal-weight cells."""
    cells = {}
    for cid, value in values_by_condition.items():
        spec = specs[cid]
        key = (tuple(spec["subset"]), round(float(spec["ratio"]), 6))
        cells.setdefault(key, []).append(value)
    per_cell = {k: float(np.mean(v)) for k, v in cells.items()}
    return float(np.mean(list(per_cell.values()))), per_cell


def worst_condition_mae(values_by_condition, specs, metric="mae"):
    per_cell, _ = aggregate_condition_matrix(
        {cid: v[metric] for cid, v in values_by_condition.items()}, specs)
    return float(max(per_cell.values())) if per_cell else float("nan")