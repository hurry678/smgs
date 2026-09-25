#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q1 A/B/C 表示探针：补充配对统计与聚类自助区间（显式扩展，不改动任何冻结产物）。

边界说明
--------
协议 q1-comparison-1.0 的冻结输出包含：
  * representation_probe/probe_metrics.csv  —— 每个 (候选, 视图, 种子) 的整表指标；
  * representation_probe/probe_predictions.csv —— 每个 (候选, 视图, 种子, 折) 的逐样本预测；
  * version_comparison/version_comparison.json —— A/B 两套对齐的端点分歧（模型间一致性）。
协议本身不输出 A/B/C 之间的配对检验与区间估计。本脚本在**不修改**上述冻结产物的前提下，
仅从已保留的逐样本预测重算：

  1) 每个 (候选, 视图) 的 video_id 聚类自助 95% 区间；
  2) B-A、C-A 的配对聚类自助区间（同一重采样索引同时作用于两个候选）；
  3) 逐样本胜/平/负计数与 Wilcoxon 符号秩检验（Holm 校正；仅用于可分解指标 Accuracy/MAE）；
  4) 效应量 Cohen's d_z 与中位差。

统计口径（引用结果时必须一并说明）
  * 聚类单元 = video_id（37 组 / 100 样本），重采样整组抽取，避免同一视频跨训练/测试折泄漏；
  * 自助次数默认 10000，区间取 2.5/97.5 百分位；
  * 每个种子的外层折划分不同，故区间按种子分别计算后对上下界取种子平均；
  * Wilcoxon 的输入是"种子平均后的逐样本差"，样本在视频内相关，p 值只作辅助证据；
  * 全部数字均为候选之间的表示效用比较，不涉及任何人工边界真值。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LABELS = ("Negative", "Neutral", "Positive")
METRICS = ("accuracy", "macro_f1", "mae", "pearson")
CLASSIFICATION_METRICS = ("accuracy", "macro_f1")
REGRESSION_METRICS = ("mae", "pearson")
CONTRASTS = (("B_git_frozen", "A_current_audited"), ("C_hybrid_capacity", "A_current_audited"))
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "full" / "representation_probe" / "probe_predictions.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0.0
    for label in LABELS:
        tp = float(np.count_nonzero((y_true == label) & (y_pred == label)))
        fp = float(np.count_nonzero((y_pred == label) & (y_true != label)))
        fn = float(np.count_nonzero((y_pred != label) & (y_true == label)))
        denom = 2.0 * tp + fp + fn
        total += 0.0 if denom == 0.0 else (2.0 * tp) / denom
    return total / len(LABELS)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    y = np.asarray(y, dtype=float) - float(np.mean(y))
    denom = float(np.sqrt(float(np.sum(x * x)) * float(np.sum(y * y))))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def metrics_for(idx, y_cls, p_cls, y_reg, p_reg):
    yt = y_cls[idx]
    yp = p_cls[idx]
    return {
        "accuracy": float(np.mean(yt == yp)),
        "macro_f1": macro_f1(yt, yp),
        "mae": float(np.mean(np.abs(y_reg[idx] - p_reg[idx]))),
        "pearson": pearson(y_reg[idx], p_reg[idx]),
    }


def holm(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, pvalues[i] * (m - rank))
        running = max(running, value)
        adjusted[i] = running
    return adjusted


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description="补充 A/B/C 配对统计与聚类自助区间")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=20260925)
    parser.add_argument("--views", default="text,audio,vision,fused")
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        print(f"缺少逐样本预测文件：{input_path}", file=sys.stderr)
        return 2

    rows = load_rows(input_path)
    candidates = sorted({r["candidate"] for r in rows})
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    seeds = sorted({int(r["seed"]) for r in rows})

    # 结构化：data[candidate][view][seed][sample_id] -> row
    data = {}
    for row in rows:
        data.setdefault(row["candidate"], {}).setdefault(row["view"], {}).setdefault(
            int(row["seed"]), {}
        )[row["sample_id"]] = row

    checks = {
        "input_file": str(input_path.name),
        "input_sha256": sha256_file(input_path),
        "row_count": len(rows),
        "candidates": candidates,
        "views": views,
        "seeds": seeds,
        "samples_per_cell": {},
        "target_consistency": {},
        "point_estimate_match_max_abs_diff": {},
    }

    probe_metrics_path = input_path.parent / "probe_metrics.csv"
    probe_means = {}
    if probe_metrics_path.is_file():
        for row in load_rows(probe_metrics_path):
            key = (row["candidate"], row["view"], int(row["seed"]))
            probe_means[key] = {m: float(row[m]) for m in METRICS}

    ci_rows = []
    pairwise_rows = []
    detail = {}

    for view in views:
        detail[view] = {}
        for seed in seeds:
            per_cand = {c: data[c][view][seed] for c in candidates}
            sample_ids = sorted(per_cand[candidates[0]].keys())
            checks["samples_per_cell"][f"{view}/{seed}"] = len(sample_ids)
            for cand in candidates:
                if set(per_cand[cand]) != set(sample_ids):
                    raise SystemExit(f"样本集合不一致：{view}/{seed}/{cand}")

            ref = per_cand[candidates[0]]
            y_cls = np.array([ref[s]["y_class"] for s in sample_ids])
            y_reg = np.array([float(ref[s]["y_reg"]) for s in sample_ids])
            vids = np.array([ref[s]["video_id"] for s in sample_ids])

            for cand in candidates:
                cls_ok = all(per_cand[cand][s]["y_class"] == ref[s]["y_class"] for s in sample_ids)
                reg_ok = all(
                    float(per_cand[cand][s]["y_reg"]) == float(ref[s]["y_reg"]) for s in sample_ids
                )
                checks["target_consistency"][f"{view}/{seed}/{cand}"] = bool(cls_ok and reg_ok)
                if not (cls_ok and reg_ok):
                    raise SystemExit(f"标签不一致：{view}/{seed}/{cand}")

            p_cls = {c: np.array([per_cand[c][s]["pred_class"] for s in sample_ids]) for c in candidates}
            p_reg = {c: np.array([float(per_cand[c][s]["pred_reg"]) for s in sample_ids]) for c in candidates}

            uniq_videos = sorted(set(vids.tolist()))
            cluster_idx = [np.where(vids == v)[0] for v in uniq_videos]
            rng = np.random.default_rng(args.base_seed + 1000 * views.index(view) + (seed % 100000))

            boot = {c: {m: np.empty(args.resamples) for m in METRICS} for c in candidates}
            full_idx = np.arange(len(sample_ids))
            point = {}
            for c in candidates:
                point[c] = metrics_for(full_idx, y_cls, p_cls[c], y_reg, p_reg[c])

            for r in range(args.resamples):
                pick = rng.integers(0, len(uniq_videos), len(uniq_videos))
                idx = np.concatenate([cluster_idx[p] for p in pick])
                for c in candidates:
                    mm = metrics_for(idx, y_cls, p_cls[c], y_reg, p_reg[c])
                    for m in METRICS:
                        boot[c][m][r] = mm[m]

            for c in candidates:
                for m in METRICS:
                    lo, hi = np.percentile(boot[c][m], [2.5, 97.5])
                    ci_rows.append(
                        {
                            "view": view,
                            "seed": seed,
                            "candidate": c,
                            "metric": m,
                            "point": point[c][m],
                            "ci_low": float(lo),
                            "ci_high": float(hi),
                            "clusters": len(uniq_videos),
                            "resamples": args.resamples,
                        }
                    )
                    if probe_metrics_path.is_file():
                        key = (c, view, seed)
                        if key in probe_means:
                            diff = abs(point[c][m] - probe_means[key][m])
                            k = f"{view}/{seed}/{c}/{m}"
                            checks["point_estimate_match_max_abs_diff"][k] = diff

            for (num, den) in CONTRASTS:
                for m in METRICS:
                    diff = boot[num][m] - boot[den][m]
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    pairwise_rows.append(
                        {
                            "view": view,
                            "seed": seed,
                            "metric": m,
                            "contrast": f"{num}-{den}",
                            "point_difference": point[num][m] - point[den][m],
                            "boot_mean_difference": float(np.mean(diff)),
                            "ci_low": float(lo),
                            "ci_high": float(hi),
                            "prob_num_better": float(np.mean(diff > 0)),
                        }
                    )

            detail[view][seed] = {
                "clusters": len(uniq_videos),
                "samples": len(sample_ids),
                "point": {c: point[c] for c in candidates},
            }

    # 逐样本胜/平/负 + Wilcoxon（种子平均后的逐样本差）
    per_sample_rows = []
    for view in views:
        corr = {c: [] for c in candidates}
        abserr = {c: [] for c in candidates}
        sample_ids = None
        for si, seed in enumerate(seeds):
            per_cand = {c: data[c][view][seed] for c in candidates}
            sample_ids = sorted(per_cand[candidates[0]].keys())
            ref = per_cand[candidates[0]]
            for c in candidates:
                y_cls = np.array([ref[s]["y_class"] for s in sample_ids])
                y_reg = np.array([float(ref[s]["y_reg"]) for s in sample_ids])
                p_cls = np.array([per_cand[c][s]["pred_class"] for s in sample_ids])
                p_reg = np.array([float(per_cand[c][s]["pred_reg"]) for s in sample_ids])
                corr[c].append((y_cls == p_cls).astype(float))
                abserr[c].append(np.abs(y_reg - p_reg))
        corr = {c: np.vstack(corr[c]) for c in candidates}
        abserr = {c: np.vstack(abserr[c]) for c in candidates}
        for (num, den) in CONTRASTS:
            for metric, table, better_is_lower in (
                ("accuracy", corr, False),
                ("mae", abserr, True),
            ):
                d = table[num].mean(axis=1) - table[den].mean(axis=1)
                wins = int(np.count_nonzero(d < 0) if better_is_lower else np.count_nonzero(d > 0))
                losses = int(np.count_nonzero(d > 0) if better_is_lower else np.count_nonzero(d < 0))
                ties = int(len(d) - wins - losses)
                per_sample_rows.append(
                    {
                        "view": view,
                        "metric": metric,
                        "contrast": f"{num}-{den}",
                        "n_samples": len(d),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "mean_difference": float(np.mean(d)),
                        "median_difference": float(np.median(d)),
                        "std_difference": float(np.std(d, ddof=1)),
                        "cohens_dz": float(np.mean(d) / np.std(d, ddof=1))
                        if np.std(d, ddof=1) > 0
                        else float("nan"),
                        "mean_num": float(np.mean(table[num])),
                        "mean_den": float(np.mean(table[den])),
                    }
                )

    try:
        from scipy import stats as scipy_stats

        have_scipy = True
    except Exception:  # pragma: no cover
        have_scipy = False

    if have_scipy:
        # 重新计算逐样本差用于 Wilcoxon（与上表同口径）
        for view in views:
            corr = {c: [] for c in candidates}
            abserr = {c: [] for c in candidates}
            for si, seed in enumerate(seeds):
                per_cand = {c: data[c][view][seed] for c in candidates}
                sample_ids = sorted(per_cand[candidates[0]].keys())
                ref = per_cand[candidates[0]]
                for c in candidates:
                    y_cls = np.array([ref[s]["y_class"] for s in sample_ids])
                    y_reg = np.array([float(ref[s]["y_reg"]) for s in sample_ids])
                    p_cls = np.array([per_cand[c][s]["pred_class"] for s in sample_ids])
                    p_reg = np.array([float(per_cand[c][s]["pred_reg"]) for s in sample_ids])
                    corr[c].append((y_cls == p_cls).astype(float))
                    abserr[c].append(np.abs(y_reg - p_reg))
            corr = {c: np.vstack(corr[c]) for c in candidates}
            abserr = {c: np.vstack(abserr[c]) for c in candidates}
            for metric, table in (("accuracy", corr), ("mae", abserr)):
                pvals = []
                targets = []
                for (num, den) in CONTRASTS:
                    d = table[num].mean(axis=1) - table[den].mean(axis=1)
                    if np.allclose(d, 0.0):
                        pvals.append(1.0)
                    else:
                        pvals.append(float(scipy_stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue))
                    targets.append(f"{num}-{den}")
                adjusted = holm(pvals)
                for target, pval, adj in zip(targets, pvals, adjusted):
                    for row in per_sample_rows:
                        if row["view"] == view and row["metric"] == metric and row["contrast"] == target:
                            row["wilcoxon_p"] = pval
                            row["holm_p"] = adj
                            row["wilcoxon_note"] = (
                                "样本在 video_id 内相关，p 值只作辅助；主证据为聚类自助区间"
                            )

    # 汇总：种子平均的区间
    summary = {}
    for view in views:
        summary[view] = {"confidence_intervals": {}, "pairwise": {}}
        for c in candidates:
            summary[view]["confidence_intervals"][c] = {}
            for m in METRICS:
                sel = [r for r in ci_rows if r["view"] == view and r["candidate"] == c and r["metric"] == m]
                summary[view]["confidence_intervals"][c][m] = {
                    "point_mean": float(np.mean([r["point"] for r in sel])),
                    "ci_low_mean": float(np.mean([r["ci_low"] for r in sel])),
                    "ci_high_mean": float(np.mean([r["ci_high"] for r in sel])),
                    "seed_count": len(sel),
                    "per_seed": [
                        {"seed": r["seed"], "point": r["point"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
                        for r in sorted(sel, key=lambda r: r["seed"])
                    ],
                }
        for (num, den) in CONTRASTS:
            key = f"{num}-{den}"
            summary[view]["pairwise"][key] = {}
            for m in METRICS:
                sel = [
                    r
                    for r in pairwise_rows
                    if r["view"] == view and r["metric"] == m and r["contrast"] == key
                ]
                summary[view]["pairwise"][key][m] = {
                    "point_difference_mean": float(np.mean([r["point_difference"] for r in sel])),
                    "ci_low_mean": float(np.mean([r["ci_low"] for r in sel])),
                    "ci_high_mean": float(np.mean([r["ci_high"] for r in sel])),
                    "prob_num_better_mean": float(np.mean([r["prob_num_better"] for r in sel])),
                }
            for m in CLASSIFICATION_METRICS + REGRESSION_METRICS:
                sel = [r for r in per_sample_rows if r["view"] == view and r["metric"] == m and r["contrast"] == key]
                if sel:
                    row = sel[0]
                    summary[view]["pairwise"][key][f"{m}_per_sample"] = {
                        "wins": row["wins"],
                        "ties": row["ties"],
                        "losses": row["losses"],
                        "cohens_dz": row["cohens_dz"],
                        "median_difference": row["median_difference"],
                        "wilcoxon_p": row.get("wilcoxon_p"),
                        "holm_p": row.get("holm_p"),
                    }

    worst = max(checks["point_estimate_match_max_abs_diff"].values(), default=None)
    checks["point_estimate_match_max_abs_diff_overall"] = worst

    result = {
        "schema": "q1-supplementary-pairwise-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_extension_of": "q1-comparison-1.0",
        "purpose": "协议未输出的 A/B/C 配对统计与聚类自助区间；由已保留的逐样本预测重算",
        "method": {
            "cluster_unit": "video_id",
            "cluster_count": 37,
            "sample_count": 100,
            "resamples": args.resamples,
            "interval": "2.5/97.5 百分位",
            "seed_aggregation": "每个种子单独聚类自助，再对区间上下界取种子平均",
            "paired_resampling": "同一重采样索引同时作用于对比双方",
            "per_sample_test": "Wilcoxon 符号秩（种子平均后的逐样本差，zero_method=wilcox，双侧）",
            "multiple_testing": "Holm（每个视图/指标内的 2 个对比）",
            "effect_size": "Cohen's d_z = mean(d)/std(d, ddof=1)",
            "claim_boundary": "候选之间的表示效用比较；不含人工边界真值；不替代 version_comparison 的模型间一致性证据",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__ if have_scipy else None,
            "platform": platform.platform(),
        },
        "checks": checks,
        "summary": summary,
    }

    (out_dir / "pairwise_statistics_supplementary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "pairwise_statistics_supplementary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "view", "seed", "metric", "contrast", "point_difference", "boot_mean_difference",
                "ci_low", "ci_high", "prob_num_better",
            ],
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)
    with (out_dir / "confidence_intervals.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["view", "seed", "candidate", "metric", "point", "ci_low", "ci_high", "clusters", "resamples"],
        )
        writer.writeheader()
        writer.writerows(ci_rows)
    with (out_dir / "per_sample_wtl.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "view", "metric", "contrast", "n_samples", "wins", "ties", "losses", "mean_difference",
            "median_difference", "std_difference", "cohens_dz", "mean_num", "mean_den", "wilcoxon_p", "holm_p",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_sample_rows)

    run_record = {
        "schema": "q1-supplementary-pairwise-run-1",
        "generated_at": result["generated_at"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input": {"file": input_path.name, "sha256": checks["input_sha256"], "rows": len(rows)},
        "resamples": args.resamples,
        "base_seed": args.base_seed,
        "environment": result["environment"],
        "note": "附加分析：不属于冻结的 170 文件运行清单，不修改任何冻结产物",
    }
    (out_dir / "supplementary_run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({"checks": checks, "summary": summary}, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())