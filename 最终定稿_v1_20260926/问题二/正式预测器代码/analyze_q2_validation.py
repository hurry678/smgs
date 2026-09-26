#!/usr/bin/env python3
"""Summarize Q2 validation evaluations without changing model selection data."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "artifacts" / "q2" / "evaluation"


def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def std(values):
    values = [float(v) for v in values]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def metric_mean(rows, strategy, key):
    return mean([row[f"{strategy}_mean_{key}"] for row in rows])


def aggregate_condition(rows):
    return {
        "n_conditions": len(rows),
        "c2_mean_mae": metric_mean(rows, "c2", "mae"),
        "c3_mean_mae": metric_mean(rows, "c3", "mae"),
        "c2_worst_condition_mae": max(row["c2_mean_mae"] for row in rows),
        "c3_worst_condition_mae": max(row["c3_mean_mae"] for row in rows),
        "c2_mean_macro_f1": metric_mean(rows, "c2", "macro_f1"),
        "c3_mean_macro_f1": metric_mean(rows, "c3", "macro_f1"),
    }


def aggregate_positions(rows):
    out = []
    for modality in ("text", "audio", "vision"):
        for ratio in (0.1, 0.3, 0.5):
            sub = [r for r in rows if r["subset"] == modality and float(r["ratio"]) == ratio]
            out.append({
                "subset": modality,
                "ratio": ratio,
                "c2_mae": mean(r["c2"]["mae"] for r in sub),
                "c3_mae": mean(r["c3"]["mae"] for r in sub),
                "c2_macro_f1": mean(r["c2"]["macro_f1"] for r in sub),
                "c3_macro_f1": mean(r["c3"]["macro_f1"] for r in sub),
                "positions": {r["position"]: {"c2_mae": r["c2"]["mae"], "c3_mae": r["c3"]["mae"]} for r in sub},
            })
    return out


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cond = data["conditions"]
    by_subset = {}
    for subset in sorted({r["subset"] for r in cond}):
        rows = [r for r in cond if r["subset"] == subset]
        by_subset[subset] = aggregate_condition(rows)
    by_ratio = {}
    for ratio in sorted({float(r["ratio"]) for r in cond}):
        rows = [r for r in cond if float(r["ratio"]) == ratio]
        by_ratio[str(ratio)] = aggregate_condition(rows)
    return {
        "candidate": path.parent.name,
        "kind": data["kind"],
        "seed": int(path.parent.name.rsplit("seed", 1)[1]),
        "thresholds": data["thresholds"],
        "full_c2": data["full_c2"],
        "full_c3": data["full_c3"],
        "conditions": aggregate_condition(cond),
        "by_subset": by_subset,
        "by_ratio": by_ratio,
        "positions": aggregate_positions(data["positions"]),
        "condition_rows": cond,
    }


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{digits}f}"


def main():
    files = sorted(EVAL_ROOT.glob("*/valid_valid_evaluation.json"))
    if not files:
        raise SystemExit(f"no evaluations under {EVAL_ROOT}")
    summaries = [summarize(p) for p in files]
    by_candidate = {s["candidate"]: s for s in summaries}
    groups = defaultdict(list)
    for s in summaries:
        groups[s["kind"]].append(s)

    family = []
    for kind, rows in sorted(groups.items()):
        family.append({
            "kind": kind,
            "n_seeds": len(rows),
            "seeds": [r["seed"] for r in rows],
            "r_mae_mean": mean(r["conditions"]["c2_mean_mae"] for r in rows),
            "r_mae_std": std(r["conditions"]["c2_mean_mae"] for r in rows),
            "c3_mae_mean": mean(r["conditions"]["c3_mean_mae"] for r in rows),
            "c3_mae_std": std(r["conditions"]["c3_mean_mae"] for r in rows),
            "worst_mae_mean": mean(r["conditions"]["c2_worst_condition_mae"] for r in rows),
            "mean_macro_f1": mean(r["conditions"]["c2_mean_macro_f1"] for r in rows),
            "full_macro_f1_mean": mean(r["full_c2"]["macro_f1"] for r in rows),
            "full_mae_mean": mean(r["full_c2"]["mae"] for r in rows),
            "full_pearson_mean": mean(r["full_c2"]["pearson"] for r in rows),
        })

    paired = []
    for seed in sorted({s["seed"] for s in summaries}):
        b4 = by_candidate.get(f"B4_seed{seed}")
        m0 = by_candidate.get(f"M0_seed{seed}")
        if not b4 or not m0:
            continue
        b4_cond = {(r["subset"], float(r["ratio"])): r for r in b4["condition_rows"]}
        m0_cond = {(r["subset"], float(r["ratio"])): r for r in m0["condition_rows"]}
        keys = sorted(set(b4_cond) & set(m0_cond))
        deltas = [m0_cond[k]["c2_mean_mae"] - b4_cond[k]["c2_mean_mae"] for k in keys]
        paired.append({
            "seed": seed,
            "r_mae_b4": b4["conditions"]["c2_mean_mae"],
            "r_mae_m0": m0["conditions"]["c2_mean_mae"],
            "delta_m0_minus_b4": m0["conditions"]["c2_mean_mae"] - b4["conditions"]["c2_mean_mae"],
            "m0_wins_conditions": sum(d < 0 for d in deltas),
            "ties_conditions": sum(abs(d) <= 1e-12 for d in deltas),
            "n_conditions": len(deltas),
            "delta_mean": mean(deltas),
            "delta_std": std(deltas),
            "full_macro_f1_b4": b4["full_c2"]["macro_f1"],
            "full_macro_f1_m0": m0["full_c2"]["macro_f1"],
            "full_macro_f1_delta_m0_minus_b4": m0["full_c2"]["macro_f1"] - b4["full_c2"]["macro_f1"],
        })

    result = {"families": family, "candidates": summaries, "paired_b4_m0": paired}
    out_json = EVAL_ROOT / "q2_validation_summary.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 问题二验证集汇总（C2 主策略）", "", "## 候选族多种子结果", "", "| 模型 | 种子数 | R_MAE 均值 | R_MAE 标准差 | 最差条件 MAE 均值 | 缺失 Macro-F1 | 完整 Macro-F1 | 完整 MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in family:
        lines.append(f"| {row['kind']} | {row['n_seeds']} | {fmt(row['r_mae_mean'])} | {fmt(row['r_mae_std'])} | {fmt(row['worst_mae_mean'])} | {fmt(row['mean_macro_f1'])} | {fmt(row['full_macro_f1_mean'])} | {fmt(row['full_mae_mean'])} |")
    lines += ["", "## B4 与 M0 逐种子配对比较", "", "| 种子 | B4 R_MAE | M0 R_MAE | Δ(M0-B4) | M0 胜出条件数 | 条件总数 | 完整 Macro-F1 Δ |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['seed']} | {fmt(row['r_mae_b4'])} | {fmt(row['r_mae_m0'])} | {fmt(row['delta_m0_minus_b4'])} | {row['m0_wins_conditions']} | {row['n_conditions']} | {fmt(row['full_macro_f1_delta_m0_minus_b4'])} |")
    lines += ["", "## 各模型条件级指标", "", "| 模型 | C2 平均 MAE | C3 平均 MAE | C2 最差条件 | C3 最差条件 | C2 平均 Macro-F1 | C3 平均 Macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        c = s["conditions"]
        lines.append(f"| {s['candidate']} | {fmt(c['c2_mean_mae'])} | {fmt(c['c3_mean_mae'])} | {fmt(c['c2_worst_condition_mae'])} | {fmt(c['c3_worst_condition_mae'])} | {fmt(c['c2_mean_macro_f1'])} | {fmt(c['c3_mean_macro_f1'])} |")
    lines += ["", "## M0 seed43 按模态与缺失比例", "", "| 模态 | 比例 | C2 MAE | C3 MAE | C2 Macro-F1 | C3 Macro-F1 |", "|---|---:|---:|---:|---:|---:|"]
    m0 = by_candidate.get("M0_seed43")
    if m0:
        for r in m0["positions"]:
            lines.append(f"| {r['subset']} | {r['ratio']:.1f} | {fmt(r['c2_mae'])} | {fmt(r['c3_mae'])} | {fmt(r['c2_macro_f1'])} | {fmt(r['c3_macro_f1'])} |")
    (EVAL_ROOT / "q2_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"families": family, "paired_b4_m0": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

