#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper-ready SVG figures and CSV tables for the ds handoff package.

Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SUBMISSION = ROOT / "submission"

PALETTE = ["#3B6FB6", "#E07B39", "#4C9A6A", "#B24C63", "#7A5AA6", "#6C7A89"]
FONT = "Microsoft YaHei, SimSun, Arial, sans-serif"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def wrap_label(label: str, width: int = 12) -> list[str]:
    label = str(label)
    if len(label) <= width:
        return [label]
    parts: list[str] = []
    rest = label
    while rest:
        parts.append(rest[:width])
        rest = rest[width:]
    return parts[:3]


class SVG:
    def __init__(self, width: int, height: int, title: str, subtitle: str = "") -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<style>'
            f'text{{font-family:{FONT};fill:#1f2937}}'
            '.axis{stroke:#9aa4b2;stroke-width:1}'
            '.grid{stroke:#e5e7eb;stroke-width:1}'
            '</style>',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text x="54" y="48" font-size="26" font-weight="700">{html.escape(title)}</text>',
        ]
        if subtitle:
            self.parts.append(
                f'<text x="54" y="76" font-size="14" fill="#596579">{html.escape(subtitle)}</text>'
            )

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 0) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
            f'fill="{fill}" stroke="{stroke}" rx="{rx}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str = "axis", stroke: str | None = None, width: float = 1) -> None:
        stroke_attr = f' stroke="{stroke}"' if stroke else ""
        self.parts.append(
            f'<line class="{cls}" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke-width="{width}"{stroke_attr}/>'
        )

    def text(self, x: float, y: float, text: Any, size: int = 14, anchor: str = "start", weight: int = 400, fill: str = "#1f2937", rotate: float | None = None) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{transform}>{html.escape(str(text))}</text>'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str = "#ffffff", width: float = 1.5) -> None:
        self.parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def polyline(self, points: Sequence[tuple[float, float]], color: str, width: float = 2.5) -> None:
        value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(f'<polyline points="{value}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def draw_legend(svg: SVG, series: Sequence[tuple[str, Sequence[float], str]], x: float, y: float, columns: int = 3) -> None:
    for i, (name, _values, color) in enumerate(series):
        col = i % columns
        row = i // columns
        xx = x + col * 250
        yy = y + row * 25
        svg.rect(xx, yy - 11, 18, 12, color, rx=2)
        svg.text(xx + 25, yy, name, size=13)


def grouped_bar(path: Path, title: str, categories: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], ylabel: str, subtitle: str = "", ylim: tuple[float, float] | None = None, digits: int = 3) -> None:
    width, height = 1200, 720
    left, right, top, bottom = 105, 45, 125, 155
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [v for _name, values, _color in series for v in values]
    ymin = min(0.0, min(all_values) if all_values else 0.0)
    ymax = max(all_values) if all_values else 1.0
    if ylim:
        ymin, ymax = ylim
    elif ymin >= 0:
        ymin = 0.0
        ymax = ymax * 1.14 if ymax else 1.0
    else:
        span = ymax - ymin
        ymin -= span * 0.10
        ymax += span * 0.12
    if ymax == ymin:
        ymax = ymin + 1.0
    svg = SVG(width, height, title, subtitle)
    svg.text(18, top + plot_h / 2, ylabel, size=13, anchor="middle", rotate=-90)
    for tick in range(6):
        val = ymin + (ymax - ymin) * tick / 5
        yy = top + plot_h - (val - ymin) / (ymax - ymin) * plot_h
        svg.line(left, yy, width - right, yy, "grid")
        svg.text(left - 10, yy + 4, fmt(val, digits), size=11, anchor="end", fill="#596579")
    svg.line(left, top, left, top + plot_h, "axis", width=1.2)
    svg.line(left, top + plot_h, width - right, top + plot_h, "axis", width=1.2)
    n = len(categories)
    group_w = plot_w / max(n, 1)
    bar_gap = 4
    bar_w = min(42, max(8, (group_w * 0.76 - bar_gap * (len(series) - 1)) / max(len(series), 1)))
    for i, category in enumerate(categories):
        center = left + group_w * (i + 0.5)
        labels = wrap_label(category, 12)
        for line_i, label in enumerate(labels):
            svg.text(center, top + plot_h + 24 + line_i * 15, label, size=11, anchor="middle")
        total_w = len(series) * bar_w + (len(series) - 1) * bar_gap
        start = center - total_w / 2
        for j, (_name, values, color) in enumerate(series):
            value = values[i] if i < len(values) else 0.0
            yy = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            x = start + j * (bar_w + bar_gap)
            svg.rect(x, yy, bar_w, top + plot_h - yy, color, rx=2)
            if len(categories) <= 8:
                svg.text(x + bar_w / 2, yy - 5, fmt(value, digits), size=9, anchor="middle", fill="#4b5563")
    draw_legend(svg, series, left, height - 45, columns=min(3, max(1, len(series))))
    svg.save(path)


def simple_bar(path: Path, title: str, categories: Sequence[str], values: Sequence[float], ylabel: str, subtitle: str = "", color: str = PALETTE[0], value_suffix: str = "") -> None:
    grouped_bar(path, title, categories, [("数量", values, color)], ylabel, subtitle=subtitle, ylim=(0, max(values) * 1.18 if values else 1), digits=0)
    # Add suffix labels to the SVG for count charts is not necessary; tables retain exact values.


def line_chart(path: Path, title: str, categories: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], ylabel: str, subtitle: str = "", ylim: tuple[float, float] | None = None) -> None:
    width, height = 1200, 700
    left, right, top, bottom = 105, 50, 125, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    vals = [v for _n, values, _c in series for v in values]
    ymin, ymax = (min(vals), max(vals)) if vals else (0, 1)
    if ylim:
        ymin, ymax = ylim
    else:
        span = ymax - ymin
        ymin -= span * 0.12
        ymax += span * 0.12
    if ymax == ymin:
        ymax = ymin + 1
    svg = SVG(width, height, title, subtitle)
    svg.text(18, top + plot_h / 2, ylabel, size=13, anchor="middle", rotate=-90)
    for tick in range(6):
        val = ymin + (ymax - ymin) * tick / 5
        yy = top + plot_h - (val - ymin) / (ymax - ymin) * plot_h
        svg.line(left, yy, width - right, yy, "grid")
        svg.text(left - 10, yy + 4, fmt(val, 3), size=11, anchor="end", fill="#596579")
    svg.line(left, top, left, top + plot_h, "axis", width=1.2)
    svg.line(left, top + plot_h, width - right, top + plot_h, "axis", width=1.2)
    n = max(len(categories), 2)
    for i, category in enumerate(categories):
        x = left + plot_w * i / (n - 1)
        svg.text(x, top + plot_h + 28, category, size=13, anchor="middle")
    for name, values, color in series:
        pts = []
        labelled_points = []
        for i, value in enumerate(values):
            x = left + plot_w * i / (n - 1)
            y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            pts.append((x, y))
            labelled_points.append((x, y, value))
        svg.polyline(pts, color)
        for x, y, value in labelled_points:
            svg.circle(x, y, 5, color)
            svg.text(x, y - 12, fmt(value, 3), size=10, anchor="middle", fill="#4b5563")
    draw_legend(svg, series, left, height - 45, columns=min(3, max(1, len(series))))
    svg.save(path)


def build_q1(output: Path, catalog: list[dict[str, str]]) -> None:
    quality = read_json(ART / "q1/data_cleaning/quality_report.json")
    counts = quality["counts"]
    rows = [
        {"scope": "附件1审计样本", "count": counts["audited"]},
        {"scope": "合格样本", "count": counts["qualified"]},
        {"scope": "排除样本", "count": counts["excluded"]},
        {"scope": "完整解码检查", "count": counts["full_decode_checked"]},
    ]
    write_csv(output / "tables/q1_data_cleaning.csv", rows)
    simple_bar(output / "figures/q1_data_cleaning.svg", "问题一：数据清洗覆盖与合格情况", [r["scope"] for r in rows], [r["count"] for r in rows], "样本数", "原始样本仅审计，不修改；完整解码通过数为100", color=PALETTE[2])
    catalog.append({"figure": "figures/q1_data_cleaning.svg", "caption": "图1 附件1数据清洗覆盖与合格情况", "source": "artifacts/q1/data_cleaning/quality_report.json", "usage": "说明100条样本全部通过文本、标签、音视频流、时长和完整解码检查。"})

    metrics = read_csv(ART / "q1/comparison/server_all_100_repeat10/comparison_metrics.csv")
    align = [r for r in metrics if r.get("family") == "alignment"]
    align_rows = []
    for r in align:
        align_rows.append({
            "method": r["method"],
            "n_samples": r["n_samples"],
            "macro_f1_mean": r["probe_macro_f1"],
            "macro_f1_std": r["probe_macro_f1_std"],
            "mae_mean": r["probe_mae"],
            "pearson_mean": r["probe_pearson"],
            "duration_fit": r["mean_duration_fit"],
            "pause_quality": r["mean_pause_quality"],
            "elapsed_sec": r["elapsed_sec"],
        })
    write_csv(output / "tables/q1_alignment_methods.csv", align_rows)
    grouped_bar(
        output / "figures/q1_alignment_methods.svg",
        "问题一：时序对齐方法对比",
        [r["method"] for r in align_rows],
        [
            ("Macro-F1", [as_float(r["macro_f1_mean"]) for r in align_rows], PALETTE[0]),
            ("时长适配", [as_float(r["duration_fit"]) for r in align_rows], PALETTE[1]),
            ("停顿质量", [as_float(r["pause_quality"]) for r in align_rows], PALETTE[2]),
        ],
        "指标值",
        "按 video_id 分组重复交叉验证；仅用于方法选型，不是正式预测模型",
        ylim=(0, 1.05),
    )
    catalog.append({"figure": "figures/q1_alignment_methods.svg", "caption": "图2 时序对齐方法对比", "source": "artifacts/q1/comparison/server_all_100_repeat10/comparison_metrics.csv", "usage": "展示不同对齐方法的探针性能、时长适配与停顿质量权衡。"})

    selected = []
    wanted = {
        "text": {"bert", "hash"},
        "audio_features": {"full_40", "energy_voiced_f0_3"},
        "vision_features": {"full_90", "no_hog_54", "compact_38"},
    }
    for r in metrics:
        if r.get("method") in wanted.get(r.get("family", ""), set()):
            selected.append({
                "family": r["family"],
                "method": r["method"],
                "probe_component": r["probe_component"],
                "feature_dim_probe": r["feature_dim_probe"],
                "macro_f1_mean": r["probe_macro_f1"],
                "mae_mean": r["probe_mae"],
                "pearson_mean": r["probe_pearson"],
                "elapsed_sec": r["elapsed_sec"],
            })
    write_csv(output / "tables/q1_modality_components.csv", selected)
    grouped_bar(
        output / "figures/q1_modality_components.svg",
        "问题一：文本、语音与视觉候选特征对比",
        [f'{r["family"]}:{r["method"]}' for r in selected],
        [
            ("Macro-F1", [as_float(r["macro_f1_mean"]) for r in selected], PALETTE[0]),
            ("MAE（越低越好）", [as_float(r["mae_mean"]) for r in selected], PALETTE[3]),
        ],
        "指标值",
        "线性探针仅用于特征方案选型；正式NPZ保留完整特征超集",
    )
    catalog.append({"figure": "figures/q1_modality_components.svg", "caption": "图3 文本、语音与视觉候选特征对比", "source": "artifacts/q1/comparison/server_all_100_repeat10/comparison_metrics.csv", "usage": "支撑BERT、语音40维和视觉视图的选型讨论。"})


def build_q2(output: Path, catalog: list[dict[str, str]]) -> None:
    robust = read_json(ART / "q2/robustness_analysis_fixed5/q2_robustness_analysis.json")
    families = robust["families"]
    family_rows = [{
        "kind": r["kind"],
        "n_seeds": r["n_seeds"],
        "r_mae_mean": r["r_mae_mean"],
        "worst_condition_mae_mean": r["worst_condition_mae_mean"],
        "mean_macro_f1": r["mean_macro_f1"],
        "full_macro_f1_mean": r["full_macro_f1_mean"],
        "full_mae_mean": r["full_mae_mean"],
        "full_pearson_mean": r["full_pearson_mean"],
    } for r in families]
    write_csv(output / "tables/q2_model_comparison.csv", family_rows)
    grouped_bar(
        output / "figures/q2_model_comparison.svg",
        "问题二：候选模型族对比",
        [r["kind"] for r in family_rows],
        [
            ("R_MAE（越低越好）", [as_float(r["r_mae_mean"]) for r in family_rows], PALETTE[0]),
            ("完整输入Macro-F1", [as_float(r["full_macro_f1_mean"]) for r in family_rows], PALETTE[2]),
        ],
        "指标值",
        "固定21个验证条件；正式发布版本为M0架构的9模型集成，不是单一seed",
    )
    catalog.append({"figure": "figures/q2_model_comparison.svg", "caption": "图4 问题二候选模型族对比", "source": "artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json", "usage": "说明掩码感知模型与备选融合结构的性能差异。"})

    factor = robust["factors"]["M0"]
    subset_order = ["audio", "vision", "audio+vision", "text", "text+audio", "text+vision", "text+audio+vision"]
    subset_rows = []
    for name in subset_order:
        item = factor["by_subset"][name]
        subset_rows.append({
            "available_modalities": name,
            "n_rows": item["n_rows"],
            "r_mae_mean": item["r_mae_mean"],
            "delta_mae_vs_full_mean": item["delta_mae_vs_full_mean"],
            "macro_f1_mean": item["macro_f1_mean"],
        })
    write_csv(output / "tables/q2_missing_robustness.csv", subset_rows)
    grouped_bar(
        output / "figures/q2_missing_robustness.svg",
        "问题二：局部缺失条件下的鲁棒性",
        [r["available_modalities"] for r in subset_rows],
        [
            ("MAE增量（越低越好）", [as_float(r["delta_mae_vs_full_mean"]) for r in subset_rows], PALETTE[3]),
            ("Macro-F1", [as_float(r["macro_f1_mean"]) for r in subset_rows], PALETTE[2]),
        ],
        "指标值",
        "M0模型族、验证集；标签表示仍可用的模态集合",
        ylim=(-0.01, 0.62),
    )
    catalog.append({"figure": "figures/q2_missing_robustness.svg", "caption": "图5 问题二局部缺失条件下的鲁棒性", "source": "artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json", "usage": "分析缺失类型组合对MAE与Macro-F1的影响；不能写成整模态永久删除。"})

    candidate_rows = []
    for c in robust["candidates"]:
        if c["kind"] in {"M0", "B4"}:
            candidate_rows.append({
                "candidate": c["candidate"],
                "kind": c["kind"],
                "seed": c["seed"],
                "r_mae": c["r_mae"],
                "worst_condition_mae": c["worst_condition_mae"],
                "mean_macro_f1": c["mean_macro_f1"],
                "full_macro_f1": c["full"]["macro_f1"],
                "full_mae": c["full"]["mae"],
                "full_pearson": c["full"]["pearson"],
            })
    write_csv(output / "tables/q2_seed_stability.csv", candidate_rows)
    for kind, color in [("M0", PALETTE[0]), ("B4", PALETTE[1])]:
        rows = [r for r in candidate_rows if r["kind"] == kind]
        line_chart(
            output / f"figures/q2_seed_stability_{kind.lower()}.svg",
            f"问题二：{kind} 随机种子稳定性",
            [str(r["seed"]) for r in rows],
            [
                ("R_MAE", [as_float(r["r_mae"]) for r in rows], color),
                ("最差条件MAE", [as_float(r["worst_condition_mae"]) for r in rows], PALETTE[3]),
            ],
            "指标值",
            "同一验证协议下的3个随机种子；用于稳定性说明",
        )
        catalog.append({"figure": f"figures/q2_seed_stability_{kind.lower()}.svg", "caption": f"图6 {kind} 随机种子稳定性", "source": "artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json", "usage": "展示随机种子对鲁棒性指标的影响，支撑9模型集成决策。"})

    preds = read_csv(SUBMISSION / "q2_predictions.csv")
    polarity_order = ["Negative", "Neutral", "Positive"]
    polarity_counts = Counter(r.get("polarity", "") for r in preds)
    pred_rows = [{"polarity": p, "count": polarity_counts.get(p, 0), "proportion": polarity_counts.get(p, 0) / len(preds) if preds else 0} for p in polarity_order]
    write_csv(output / "tables/q2_prediction_distribution.csv", pred_rows)
    simple_bar(
        output / "figures/q2_prediction_distribution.svg",
        "问题二：附件3预测极性分布",
        [r["polarity"] for r in pred_rows],
        [r["count"] for r in pred_rows],
        "样本数",
        f"附件3无标签，仅展示推理结果；n={len(preds)}",
        color=PALETTE[0],
    )
    catalog.append({"figure": "figures/q2_prediction_distribution.svg", "caption": "图7 附件3预测极性分布", "source": "submission/q2_predictions.csv", "usage": "报告附件3推理结果；不得据此计算监督准确率。"})


def build_q3(output: Path, catalog: list[dict[str, str]]) -> None:
    validation = read_json(ART / "q3/final_validation/q3_validation_report.json")
    shap = validation["shapley_vs_loo"]
    shap_rows = []
    for modality in ["text", "audio", "vision"]:
        shap_rows.append({
            "modality": modality,
            "class_abs_share_mean": shap[f"mean_class_abs_share_{modality}"],
            "intensity_abs_share_mean": shap[f"mean_intensity_abs_share_{modality}"],
        })
    write_csv(output / "tables/q3_shapley_modalities.csv", shap_rows)
    grouped_bar(
        output / "figures/q3_shapley_modalities.svg",
        "问题三：模态级Shapley绝对作用份额",
        [r["modality"] for r in shap_rows],
        [
            ("类别绝对份额", [as_float(r["class_abs_share_mean"]) for r in shap_rows], PALETTE[0]),
            ("强度绝对份额", [as_float(r["intensity_abs_share_mean"]) for r in shap_rows], PALETTE[1]),
        ],
        "平均绝对份额",
        f"验证集n={shap['n']}；精确枚举8个模态子集；Shapley与LOO主模态一致率={fmt(shap['main_modality_agreement_shapley_vs_loo'], 3)}",
        ylim=(0, max([r["class_abs_share_mean"] for r in shap_rows] + [r["intensity_abs_share_mean"] for r in shap_rows]) * 1.25),
    )
    catalog.append({"figure": "figures/q3_shapley_modalities.svg", "caption": "图8 模态级Shapley绝对作用份额", "source": "artifacts/q3/final_validation/q3_validation_report.json", "usage": "量化文本、语音、视觉对类别与强度预测的平均绝对作用。"})

    sensitivity_rows = []
    for w in [3, 5, 10]:
        report = read_json(ART / f"q3/sensitivity_w{w}/q3_validation_report.json")
        loc = report["local_occlusion"]
        sensitivity_rows.append({
            "window_size": w,
            "n_samples": report["shapley_vs_loo"]["n"],
            "n_perturbations": loc["continuous"]["n_perturbations"],
            "mean_abs_class_delta": loc["continuous"]["mean_abs_class_delta"],
            "top1_concentration": loc["mean_top1_concentration_continuous"],
            "win_rate_vs_random": loc["continuous_win_rate_vs_random_contiguous"],
            "win_rate_vs_point": loc["continuous_win_rate_vs_point_scatter"],
        })
    write_csv(output / "tables/q3_sensitivity.csv", sensitivity_rows)
    line_chart(
        output / "figures/q3_sensitivity.svg",
        "问题三：解释窗口大小敏感性",
        [f"w={r['window_size']}" for r in sensitivity_rows],
        [
            ("连续窗口Top-1集中度", [as_float(r["top1_concentration"]) for r in sensitivity_rows], PALETTE[0]),
            ("对随机连续遮挡胜率", [as_float(r["win_rate_vs_random"]) for r in sensitivity_rows], PALETTE[1]),
            ("对点散遮挡胜率", [as_float(r["win_rate_vs_point"]) for r in sensitivity_rows], PALETTE[2]),
        ],
        "比例",
        "验证集固定样本；窗口越小定位更细，但连续结构优势不应被过度解释为因果",
        ylim=(0.30, 0.72),
    )
    catalog.append({"figure": "figures/q3_sensitivity.svg", "caption": "图9 解释窗口大小敏感性", "source": "artifacts/q3/sensitivity_w3/、w5/、w10/", "usage": "分析窗口尺度对证据集中度与控制组胜率的影响。"})

    loc = validation["local_occlusion"]
    loc_rows = []
    for key, label in [("continuous", "连续窗口"), ("random_contiguous", "随机连续"), ("point_scatter", "点散遮挡")]:
        loc_rows.append({
            "method": label,
            "n_perturbations": loc[key]["n_perturbations"],
            "mean_abs_class_delta": loc[key]["mean_abs_class_delta"],
            "median_abs_class_delta": loc[key]["median_abs_class_delta"],
            "mean_abs_raw_delta": loc[key]["mean_abs_raw_delta"],
        })
    write_csv(output / "tables/q3_local_occlusion.csv", loc_rows)
    grouped_bar(
        output / "figures/q3_local_occlusion.svg",
        "问题三：局部遮挡忠实性对照",
        [r["method"] for r in loc_rows],
        [
            ("平均类别绝对变化", [as_float(r["mean_abs_class_delta"]) for r in loc_rows], PALETTE[0]),
            ("平均强度绝对变化", [as_float(r["mean_abs_raw_delta"]) for r in loc_rows], PALETTE[1]),
        ],
        "平均绝对变化",
        "验证集；连续窗口、随机连续窗口与点散遮挡使用相同预算",
    )
    catalog.append({"figure": "figures/q3_local_occlusion.svg", "caption": "图10 局部遮挡忠实性对照", "source": "artifacts/q3/final_validation/q3_validation_report.json", "usage": "说明连续窗口证据与对照组的差异，不宣称物理因果。"})

    pred_path = SUBMISSION / "q3_predictions_explanations.csv"
    preds = read_csv(pred_path)
    polarity_order = ["Negative", "Neutral", "Positive"]
    modality_order = ["text", "audio", "vision"]
    pol = Counter(r.get("polarity", "") for r in preds)
    prim = Counter(r.get("primary_modality", "") for r in preds)
    attachment_rows = []
    for p in polarity_order:
        attachment_rows.append({"dimension": "polarity", "value": p, "count": pol.get(p, 0)})
    for m in modality_order:
        attachment_rows.append({"dimension": "primary_modality", "value": m, "count": prim.get(m, 0)})
    write_csv(output / "tables/q3_attachment4_distribution.csv", attachment_rows)
    grouped_bar(
        output / "figures/q3_attachment4_distribution.svg",
        "问题三：附件4预测与主模态分布",
        [f'{r["dimension"]}:{r["value"]}' for r in attachment_rows],
        [("样本数", [as_float(r["count"]) for r in attachment_rows], PALETTE[0])],
        "样本数",
        f"附件4无标签，仅展示20条推理与解释结果",
        ylim=(0, max([as_float(r["count"]) for r in attachment_rows] + [1]) * 1.25),
        digits=0,
    )
    catalog.append({"figure": "figures/q3_attachment4_distribution.svg", "caption": "图11 附件4预测与主模态分布", "source": "submission/q3_predictions_explanations.csv", "usage": "报告附件4推理输出；不得据此计算监督指标。"})

    evidence_counts: Counter[tuple[str, str]] = Counter()
    mapping_counts: Counter[tuple[str, str]] = Counter()
    for row in preds:
        for modality in modality_order:
            raw = row.get(f"key_evidence_{modality}_json", "")
            if not raw:
                continue
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for item in items:
                evidence_counts[(modality, str(item.get("evidence_level", "unknown")))] += 1
                mapping_counts[(modality, str(item.get("mapping_status", "unknown")))] += 1
    evidence_rows = []
    for modality in modality_order:
        for level in ["high", "medium", "low"]:
            evidence_rows.append({"modality": modality, "evidence_level": level, "count": evidence_counts.get((modality, level), 0)})
    write_csv(output / "tables/q3_evidence_levels.csv", evidence_rows)
    grouped_bar(
        output / "figures/q3_evidence_levels.svg",
        "问题三：附件4关键证据等级分布",
        modality_order,
        [
            ("high", [evidence_counts.get((m, "high"), 0) for m in modality_order], PALETTE[2]),
            ("medium", [evidence_counts.get((m, "medium"), 0) for m in modality_order], PALETTE[1]),
            ("low", [evidence_counts.get((m, "low"), 0) for m in modality_order], PALETTE[3]),
        ],
        "证据片段数",
        "按样本×模态×窗口配对统计；等级由排名和控制组胜率共同确定",
        ylim=(0, max([evidence_counts.get((m, level), 0) for m in modality_order for level in ["high", "medium", "low"]] + [1]) * 1.2),
        digits=0,
    )
    catalog.append({"figure": "figures/q3_evidence_levels.svg", "caption": "图12 附件4关键证据等级分布", "source": "submission/q3_predictions_explanations.csv", "usage": "展示证据等级分布，并区分解释等级与预测正确性。"})

    mapping_rows = [{"modality": m, "mapping_status": status, "count": count} for (m, status), count in sorted(mapping_counts.items())]
    write_csv(output / "tables/q3_evidence_mapping.csv", mapping_rows)


def write_catalog(output: Path, catalog: list[dict[str, str]]) -> None:
    lines = ["# 论文图表目录（自动生成）", "", "运行 `python paper/generate_paper_figures.py` 可重建本目录。", "", "| 图片 | 建议图注 | 数据来源 | 写作用途 |", "|---|---|---|---|"]
    for row in catalog:
        lines.append(f'| `{row["figure"]}` | {row["caption"]} | `{row["source"]}` | {row["usage"]} |')
    lines += ["", "## 表格", "", "所有 CSV 位于 `paper/tables/`，字段名和数值保留原始精度，可直接用于论文表格或进一步绘图。", ""]
    output.mkdir(parents=True, exist_ok=True)
    (output / "FIGURE_CATALOG.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate E-problem paper figures and tables using only the standard library.")
    parser.add_argument("--output", type=Path, default=ROOT / "paper", help="Output directory (default: ds/paper)")
    parser.add_argument("--list", action="store_true", help="List planned outputs without writing files")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.list:
        print(output / "figures")
        print(output / "tables")
        print(output / "FIGURE_CATALOG.md")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, str]] = []
    build_q1(output, catalog)
    build_q2(output, catalog)
    build_q3(output, catalog)
    write_catalog(output, catalog)
    print(f"Generated {len(catalog)} SVG figures and paper tables in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
