#!/usr/bin/env python3
"""Independent published metrics and paired video-cluster bootstrap.

Usage: --predictions predictions.csv --bank confirm --reference B2 --candidate M2
Both candidates must contain identical IDs/labels/masks and complete paired cells.
No model fitting, threshold search, or test-driven selection is performed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"Negative": 0, "Neutral": 1, "Positive": 2}


def statistics(rows, group_index):
    """Sufficient statistics per video: confusion[9], abs error, sum y/p/y²/p²/yp."""
    out = np.zeros((len(group_index), 15), dtype=np.float64)
    for r in rows:
        g = group_index[r["video_id"]]
        y, p = int(r["y_class"]), LABELS[r["polarity"]]
        yr, pr = float(r["y_reg"]), float(r["intensity"])
        out[g, y*3+p] += 1
        out[g, 9:] += (abs(yr-pr), yr, pr, yr*yr, pr*pr, yr*pr)
    return out


def scores(stats):
    cm = stats[..., :9].reshape(*stats.shape[:-1], 3, 3)
    n = cm.sum(axis=(-2, -1))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-2) + cm.sum(-1)
    f1 = np.divide(2*tp, denom, out=np.zeros_like(tp), where=denom > 0).mean(-1)
    acc, mae = tp.sum(-1)/n, stats[..., 9]/n
    sy, sp, syy, spp, syp = (stats[..., k] for k in range(10, 15))
    cov = syp - sy*sp/n
    var = np.maximum(syy-sy*sy/n, 0) * np.maximum(spp-sp*sp/n, 0)
    pearson = np.divide(cov, np.sqrt(var), out=np.full_like(cov, np.nan), where=var > 1e-20)
    return np.stack((acc, f1, mae, pearson), axis=-1)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def evaluate(rows, reference, candidate, resamples=10000, seed=20260925):
    needed = {"candidate", "train_seed", "bank", "condition_id", "replica", "sample_id",
              "video_id", "y_class", "y_reg", "polarity", "intensity", "mask_sha256"}
    cells = {name: defaultdict(dict) for name in (reference, candidate)}
    for r in rows:
        if not needed.issubset(r):
            raise ValueError("Missing prediction columns")
        if r["candidate"] not in cells:
            continue
        key = (r["train_seed"], r["condition_id"], r["replica"])
        if r["sample_id"] in cells[r["candidate"]][key]:
            raise ValueError("Duplicate prediction key")
        if r["polarity"] not in LABELS or int(r["y_class"]) not in (0, 1, 2):
            raise ValueError("Invalid class")
        val, truth = float(r["intensity"]), float(r["y_reg"])
        if not np.isfinite([val, truth]).all() or abs(val) > 3 or abs(truth) > 3:
            raise ValueError("Invalid intensity/label")
        if LABELS[r["polarity"]] != (0 if val < 0 else 2 if val > 0 else 1):
            raise ValueError("Published polarity/intensity conflict")
        if int(r["y_class"]) != (0 if truth < 0 else 2 if truth > 0 else 1):
            raise ValueError("Ground truth polarity/intensity conflict")
        cells[r["candidate"]][key][r["sample_id"]] = r
    ref, cand = cells[reference], cells[candidate]
    if not ref or set(ref) != set(cand):
        raise ValueError("Unpaired condition/seed/replica cells")
    keys = sorted(ref)
    ids = sorted(ref[keys[0]])
    base = ref[keys[0]]
    for name in cells:
        for key, table in cells[name].items():
            if sorted(table) != ids:
                raise ValueError("Incomplete sample coverage")
            for sid in ids:
                r = table[sid]
                if any(r[k] != base[sid][k] for k in ("video_id", "y_class", "y_reg")):
                    raise ValueError("Inconsistent identity/label across conditions")
                if name == candidate and r["mask_sha256"] != ref[key][sid]["mask_sha256"]:
                    raise ValueError("Different evaluation masks between candidates")
    # Exact mean(seed -> condition -> replica), even if repeat counts differ.
    seeds = sorted({k[0] for k in keys})
    condition_sets = [{k[1] for k in keys if k[0] == s} for s in seeds]
    if any(x != condition_sets[0] for x in condition_sets):
        raise ValueError("Different condition sets across training seeds")
    weights = []
    for s, c, _ in keys:
        nrep = sum(k[0] == s and k[1] == c for k in keys)
        weights.append(1/(len(seeds)*len(condition_sets[0])*nrep))
    weights = np.asarray(weights)
    groups = sorted({base[s]["video_id"] for s in ids})
    group_index = {g: i for i, g in enumerate(groups)}
    tables = {name: np.stack([statistics(list(cells[name][k].values()), group_index) for k in keys])
              for name in cells}
    point = {name: (scores(tab.sum(1))*weights[:, None]).sum(0) for name, tab in tables.items()}
    rng = np.random.default_rng(seed)
    boot = {name: np.empty((resamples, 4)) for name in cells}
    # Common draw weights across all candidates, conditions, replicas and seeds.
    # Sufficient statistics avoid copying entire prediction tables per resample.
    for start in range(0, resamples, 128):
        count = min(128, resamples-start)
        draws = rng.integers(len(groups), size=(count, len(groups)))
        freq = np.stack([np.bincount(d, minlength=len(groups)) for d in draws]).astype(float)
        for name, tab in tables.items():
            total = np.zeros((count, 4))
            for j in range(len(keys)):
                total += weights[j] * scores(freq @ tab[j])
            boot[name][start:start+count] = total
    metrics = ("accuracy", "macro_f1", "mae", "pearson")
    summary = {}
    for j, metric in enumerate(metrics):
        diff = boot[candidate][:, j] - boot[reference][:, j]
        good = diff[np.isfinite(diff)]
        summary[metric] = {
            "reference": point[reference][j], "candidate": point[candidate][j],
            "candidate_minus_reference": point[candidate][j]-point[reference][j],
            "paired_95pct": np.quantile(good, [0.025, 0.975]).tolist() if len(good) else None,
            "defined_bootstrap_draws": len(good), "better_direction": "lower" if metric == "mae" else "higher"
        }
    return finite_json({
        "schema": "q2-paired-video-bootstrap-1", "reference": reference, "candidate": candidate,
        "samples": len(ids), "video_groups": len(groups), "seeds": seeds, "cells_per_candidate": len(keys),
        "resamples": resamples, "seed": seed, "metrics": summary,
        "method": "resample video groups jointly across all cells; average metrics within each draw then take quantiles",
        "limitations": "Conditional on frozen models and this validation set; no refitting; not independent training-data uncertainty."
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--bank", required=True, choices=("select", "confirm", "position", "coupling", "clean"))
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.reference == args.candidate or args.resamples < 1:
        ap.error("Need distinct candidates and positive resamples")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        ap.error("Output must be a new file under 问题二")
    with args.predictions.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["bank"] == args.bank]
    result = evaluate(rows, args.reference, args.candidate, args.resamples, args.seed)
    result["bank"] = args.bank
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
