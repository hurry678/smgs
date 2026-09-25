#!/usr/bin/env python3
"""Paired statistics for the Q1 method-selection experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


REFERENCE_METHODS = {
    "alignment": "full_dp_20",
    "smoothing": "sigma_0p75",
    "text": "bert",
    "audio_features": "full_40",
    "vision_features": "full_90",
    "vision_fps": "fps_5p0",
}
METRICS = ("probe_repeat_macro_f1", "probe_repeat_pearson")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 paired comparison statistics")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def finite_pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    if left.size != right.size:
        raise ValueError(f"paired arrays differ in length: {left.size} vs {right.size}")
    valid = np.isfinite(left) & np.isfinite(right)
    return left[valid], right[valid]


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, float(p_values[int(index)]) * (count - rank))
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def paired_row(
    family: str,
    metric: str,
    candidate: str,
    reference: str,
    candidate_values: Any,
    reference_values: Any,
) -> dict[str, Any]:
    candidate_arr, reference_arr = finite_pair(candidate_values, reference_values)
    differences = candidate_arr - reference_arr
    n = int(differences.size)
    if n == 0:
        return {
            "family": family,
            "metric": metric,
            "candidate": candidate,
            "reference": reference,
            "n_pairs": 0,
            "candidate_mean": float("nan"),
            "reference_mean": float("nan"),
            "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "cohen_dz": float("nan"),
            "paired_t_p": float("nan"),
            "wilcoxon_p": float("nan"),
        }

    mean_difference = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    if n > 1:
        critical = float(stats.t.ppf(0.975, df=n - 1))
        margin = critical * sd / math.sqrt(n)
        ci_low, ci_high = mean_difference - margin, mean_difference + margin
        t_result = stats.ttest_rel(candidate_arr, reference_arr, nan_policy="omit")
        paired_t_p = float(t_result.pvalue)
    else:
        ci_low = ci_high = mean_difference
        paired_t_p = float("nan")

    tolerance = 1e-12
    wins = int(np.sum(differences > tolerance))
    ties = int(np.sum(np.abs(differences) <= tolerance))
    losses = int(n - wins - ties)

    if n > 1 and np.any(np.abs(differences) > tolerance):
        try:
            wilcoxon_p = float(stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
    else:
        wilcoxon_p = float("nan")

    return {
        "family": family,
        "metric": metric,
        "candidate": candidate,
        "reference": reference,
        "n_pairs": n,
        "candidate_mean": float(np.mean(candidate_arr)),
        "reference_mean": float(np.mean(reference_arr)),
        "mean_difference": mean_difference,
        "median_difference": float(np.median(differences)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cohen_dz": float(mean_difference / sd) if sd > 1e-12 else float("nan"),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], reference_methods: dict[str, str]) -> None:
    lines = [
        "# 问题一候选方法配对统计",
        "",
        "同一次重复内所有候选使用相同的 `video_id` 分组折分；表中只比较共享的10次重复结果。",
        "`p` 值基于重复实验的配对差，重复间并非独立样本，因此仅作方法选型证据，不作正式泛化显著性声明。",
        "",
    ]
    for metric in METRICS:
        lines.extend([
            f"## {metric}",
            "",
            "| family | candidate | reference | mean diff (candidate-reference) | 95% CI | W/T/L | paired t p | Holm p | Wilcoxon p |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ])
        selected = [row for row in rows if row["metric"] == metric and row["candidate"] != reference_methods[row["family"]]]
        selected.sort(key=lambda row: (row["family"], -row["mean_difference"]))
        for row in selected:
            lines.append(
                "| {family} | {candidate} | {reference} | {mean_difference:.4f} | [{ci95_low:.4f}, {ci95_high:.4f}] | "
                "{wins}/{ties}/{losses} | {paired_t_p:.4f} | {holm_p:.4f} | {wilcoxon_p:.4f} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    with args.metrics.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    rows_by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        family = str(row["family"])
        method = str(row["method"])
        rows_by_family.setdefault(family, {})[method] = row

    output_rows: list[dict[str, Any]] = []
    for family, reference_method in REFERENCE_METHODS.items():
        if family not in rows_by_family:
            raise KeyError(f"missing family: {family}")
        candidates = rows_by_family[family]
        if reference_method not in candidates:
            raise KeyError(f"missing reference method {reference_method} in {family}")
        reference_row = candidates[reference_method]
        for method, candidate_row in candidates.items():
            if method == reference_method:
                continue
            for metric in METRICS:
                output_rows.append(paired_row(
                    family,
                    metric,
                    method,
                    reference_method,
                    json.loads(candidate_row[metric]),
                    json.loads(reference_row[metric]),
                ))

    for family in REFERENCE_METHODS:
        for metric in METRICS:
            indices = [
                index for index, row in enumerate(output_rows)
                if row["family"] == family and row["metric"] == metric
            ]
            adjusted = holm_adjust([float(output_rows[index]["paired_t_p"]) for index in indices])
            for index, value in zip(indices, adjusted):
                output_rows[index]["holm_p"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family", "metric", "candidate", "reference", "n_pairs",
        "candidate_mean", "reference_mean", "mean_difference", "median_difference",
        "ci95_low", "ci95_high", "wins", "ties", "losses", "cohen_dz",
        "paired_t_p", "holm_p", "wilcoxon_p",
    ]
    with (args.output_dir / "pairwise_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    payload = {
        "method": "paired candidate-vs-reference comparison over repeated grouped probe folds",
        "reference_methods": REFERENCE_METHODS,
        "metrics": METRICS,
        "n_rows": len(output_rows),
        "caveats": [
            "Repeated folds are not independent observations; p-values are selection-only diagnostics.",
            "The probe is not a formal problem-1 prediction model and cannot replace alignment-quality validation.",
            "Holm correction is applied within each family and metric across non-reference candidates.",
        ],
        "rows": output_rows,
    }
    (args.output_dir / "pairwise_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.output_dir / "pairwise_statistics.md", output_rows, REFERENCE_METHODS)
    print(json.dumps({"output_dir": str(args.output_dir), "n_rows": len(output_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
