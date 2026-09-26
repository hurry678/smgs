#!/usr/bin/env python3
"""Build a balanced Q2 robustness report from validation-only evaluations.

This script intentionally uses only the Python standard library.  It does not
train models, inspect the official test split, or modify submission files.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("text", "audio", "vision")
SUBSETS = (
    "text", "audio", "vision", "text+audio", "text+vision",
    "audio+vision", "text+audio+vision",
)
RATIOS = (0.1, 0.3, 0.5)
POSITIONS = ("start", "middle", "end")
EPS = 1e-12


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pop_std(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def candidate_key(payload: dict[str, Any], path: Path) -> str:
    kind = str(payload.get("kind") or path.parent.name.split("_seed", 1)[0])
    seed = int(payload.get("seed", path.parent.name.rsplit("seed", 1)[1]))
    return f"{kind}_seed{seed}"


def load_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    found: dict[str, tuple[float, dict[str, Any]]] = {}
    sources: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        sources.append(str(root.resolve()))
        for path in sorted(root.glob("*/valid_valid_evaluation.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = candidate_key(payload, path)
            mtime = path.stat().st_mtime
            if key not in found or mtime > found[key][0]:
                found[key] = (mtime, payload)
    candidates = []
    for key in sorted(found):
        payload = found[key][1]
        payload = dict(payload)
        payload["_candidate"] = key
        payload["_source_mtime"] = found[key][0]
        candidates.append(payload)
    return candidates


def aggregate_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    conditions = list(payload.get("conditions", []))
    positions = list(payload.get("positions", []))
    r_maes = [float(row["c2_mean_mae"]) for row in conditions]
    r_f1s = [float(row["c2_mean_macro_f1"]) for row in conditions]
    full = payload.get("full_c2", {})
    return {
        "candidate": payload["_candidate"],
        "kind": str(payload["kind"]),
        "seed": int(payload["seed"]),
        "n_conditions": len(conditions),
        "n_position_conditions": len(positions),
        "r_mae": mean(r_maes),
        "r_mae_std_over_conditions": pop_std(r_maes),
        "worst_condition_mae": max(r_maes) if r_maes else float("nan"),
        "mean_macro_f1": mean(r_f1s),
        "full": {
            "accuracy": float(full.get("accuracy", float("nan"))),
            "macro_f1": float(full.get("macro_f1", float("nan"))),
            "mae": float(full.get("mae", float("nan"))),
            "pearson": float(full.get("pearson", float("nan"))),
            "n": int(full.get("n", 0)),
        },
        "thresholds": payload.get("thresholds"),
        "_conditions": conditions,
        "_positions": positions,
    }


def summarize_families(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["kind"]].append(candidate)
    output = []
    for kind, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["seed"])
        r_maes = [row["r_mae"] for row in rows]
        worst = [row["worst_condition_mae"] for row in rows]
        macro = [row["mean_macro_f1"] for row in rows]
        full_macro = [row["full"]["macro_f1"] for row in rows]
        full_mae = [row["full"]["mae"] for row in rows]
        full_pearson = [row["full"]["pearson"] for row in rows]
        seeds = [row["seed"] for row in rows]
        output.append({
            "kind": kind,
            "n_seeds": len(rows),
            "seeds": seeds,
            "complete_3_seed": set(seeds) == {42, 43, 44},
            "r_mae_mean": mean(r_maes),
            "r_mae_std_sample": sample_std(r_maes),
            "r_mae_std_population": pop_std(r_maes),
            "r_mae_min": min(r_maes) if r_maes else float("nan"),
            "r_mae_max": max(r_maes) if r_maes else float("nan"),
            "worst_condition_mae_mean": mean(worst),
            "worst_condition_mae_std_sample": sample_std(worst),
            "mean_macro_f1": mean(macro),
            "mean_macro_f1_std_sample": sample_std(macro),
            "full_macro_f1_mean": mean(full_macro),
            "full_macro_f1_std_sample": sample_std(full_macro),
            "full_mae_mean": mean(full_mae),
            "full_mae_std_sample": sample_std(full_mae),
            "full_pearson_mean": mean(full_pearson),
            "full_pearson_std_sample": sample_std(full_pearson),
        })
    return output


def exact_sign_test(a_wins: int, b_wins: int) -> float:
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def paired_comparisons(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_kind: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in candidates:
        by_kind[row["kind"]][row["seed"]] = row
    kinds = sorted(by_kind)
    output = []
    for i, a_kind in enumerate(kinds):
        for b_kind in kinds[i + 1:]:
            common = sorted(set(by_kind[a_kind]) & set(by_kind[b_kind]))
            if not common:
                continue
            deltas = [by_kind[a_kind][s]["r_mae"] - by_kind[b_kind][s]["r_mae"] for s in common]
            macro_deltas = [
                by_kind[a_kind][s]["full"]["macro_f1"] - by_kind[b_kind][s]["full"]["macro_f1"]
                for s in common
            ]
            a_wins = sum(delta < -EPS for delta in deltas)
            b_wins = sum(delta > EPS for delta in deltas)
            output.append({
                "family_a": a_kind,
                "family_b": b_kind,
                "common_seeds": common,
                "delta_r_mae_a_minus_b_mean": mean(deltas),
                "delta_r_mae_a_minus_b_std_sample": sample_std(deltas),
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": len(deltas) - a_wins - b_wins,
                "sign_test_p": exact_sign_test(a_wins, b_wins),
                "delta_full_macro_f1_a_minus_b_mean": mean(macro_deltas),
                "per_seed": [
                    {
                        "seed": seed,
                        "r_mae_a": by_kind[a_kind][seed]["r_mae"],
                        "r_mae_b": by_kind[b_kind][seed]["r_mae"],
                        "delta_a_minus_b": by_kind[a_kind][seed]["r_mae"] - by_kind[b_kind][seed]["r_mae"],
                    }
                    for seed in common
                ],
            })
    return output


def factor_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["kind"]].append(row)
    output: dict[str, Any] = {}
    for kind, rows in sorted(grouped.items()):
        full_mae = mean(row["full"]["mae"] for row in rows)
        full_macro = mean(row["full"]["macro_f1"] for row in rows)

        def agg_condition(selector) -> dict[str, Any]:
            values = []
            macros = []
            deltas = []
            for row in rows:
                for condition in row["_conditions"]:
                    if selector(condition):
                        mae = float(condition["c2_mean_mae"])
                        values.append(mae)
                        macros.append(float(condition["c2_mean_macro_f1"]))
                        deltas.append(mae - row["full"]["mae"])
            return {
                "n_rows": len(values),
                "r_mae_mean": mean(values),
                "r_mae_std_over_rows": pop_std(values),
                "delta_mae_vs_full_mean": mean(deltas),
                "macro_f1_mean": mean(macros),
            }

        def agg_position(selector) -> dict[str, Any]:
            values = []
            macros = []
            deltas = []
            for row in rows:
                for condition in row["_positions"]:
                    if selector(condition):
                        mae = float(condition["c2"]["mae"])
                        values.append(mae)
                        macros.append(float(condition["c2"]["macro_f1"]))
                        deltas.append(mae - row["full"]["mae"])
            return {
                "n_rows": len(values),
                "r_mae_mean": mean(values),
                "r_mae_std_over_rows": pop_std(values),
                "delta_mae_vs_full_mean": mean(deltas),
                "macro_f1_mean": mean(macros),
            }

        by_subset = {
            subset: agg_condition(lambda row, subset=subset: row["subset"] == subset)
            for subset in SUBSETS
        }
        by_ratio = {
            str(ratio): agg_condition(lambda row, ratio=ratio: float(row["ratio"]) == ratio)
            for ratio in RATIOS
        }
        by_position = {
            position: agg_position(lambda row, position=position: row["position"] == position)
            for position in POSITIONS
        }
        subset_ratio = {}
        for subset in SUBSETS:
            subset_ratio[subset] = {
                str(ratio): agg_condition(
                    lambda row, subset=subset, ratio=ratio:
                    row["subset"] == subset and float(row["ratio"]) == ratio
                )
                for ratio in RATIOS
            }
        output[kind] = {
            "full_mae_mean": full_mae,
            "full_macro_f1_mean": full_macro,
            "by_subset": by_subset,
            "by_ratio": by_ratio,
            "by_position": by_position,
            "subset_ratio": subset_ratio,
        }
    return output


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# 问题二鲁棒性验证增强报告",
        "",
        "> 本报告只使用验证集结果；不包含官方test重评，不改变附件3正式提交。",
        "",
        "## 1. 平衡种子矩阵",
        "",
        "| 模型 | 种子 | R_MAE | 最差条件MAE | 缺失Macro-F1 | 完整MAE | 完整Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['kind']} | {row['seed']} | {fmt(row['r_mae'])} | "
            f"{fmt(row['worst_condition_mae'])} | {fmt(row['mean_macro_f1'])} | "
            f"{fmt(row['full']['mae'])} | {fmt(row['full']['macro_f1'])} |"
        )
    lines += [
        "",
        "## 2. 族级均值与稳定性",
        "",
        "| 模型 | 种子数 | 种子 | R_MAE均值±样本标准差 | 最差条件MAE | 缺失Macro-F1 | 完整Macro-F1 | 三 seed 完整 |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in result["families"]:
        lines.append(
            f"| {row['kind']} | {row['n_seeds']} | {','.join(map(str, row['seeds']))} | "
            f"{fmt(row['r_mae_mean'])}±{fmt(row['r_mae_std_sample'])} | "
            f"{fmt(row['worst_condition_mae_mean'])} | {fmt(row['mean_macro_f1'])} | "
            f"{fmt(row['full_macro_f1_mean'])} | {'是' if row['complete_3_seed'] else '否'} |"
        )
    lines += [
        "",
        "## 3. 配对比较",
        "",
        "Δ = A - B；ΔR_MAE为负表示A优于B。符号检验只作小样本描述，不据此宣称显著性。",
        "",
        "| A vs B | 共同种子 | ΔR_MAE均值±样本标准差 | A胜 | B胜 | 平 | 符号检验p | Δ完整Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["pairwise"]:
        lines.append(
            f"| {row['family_a']} vs {row['family_b']} | "
            f"{','.join(map(str, row['common_seeds']))} | "
            f"{fmt(row['delta_r_mae_a_minus_b_mean'])}±{fmt(row['delta_r_mae_a_minus_b_std_sample'])} | "
            f"{row['a_wins']} | {row['b_wins']} | {row['ties']} | "
            f"{fmt(row['sign_test_p'])} | {fmt(row['delta_full_macro_f1_a_minus_b_mean'])} |"
        )
    lines += ["", "## 4. 缺失因素主效应", ""]
    for kind, data in result["factors"].items():
        lines += [
            f"### {kind}",
            "",
            f"完整输入MAE={fmt(data['full_mae_mean'])}。",
            "",
            "| 缺失模态 | R_MAE | ΔMAE相对完整 | Macro-F1 |",
            "|---|---:|---:|---:|",
        ]
        for subset in SUBSETS:
            row = data["by_subset"][subset]
            lines.append(
                f"| {subset} | {fmt(row['r_mae_mean'])} | "
                f"{fmt(row['delta_mae_vs_full_mean'])} | {fmt(row['macro_f1_mean'])} |"
            )
        lines += ["", "| 缺失比例 | R_MAE | ΔMAE相对完整 | Macro-F1 |", "|---|---:|---:|---:|"]
        for ratio in RATIOS:
            row = data["by_ratio"][str(ratio)]
            lines.append(
                f"| {ratio:.1f} | {fmt(row['r_mae_mean'])} | "
                f"{fmt(row['delta_mae_vs_full_mean'])} | {fmt(row['macro_f1_mean'])} |"
            )
        lines += ["", "| 连续缺失位置 | R_MAE | ΔMAE相对完整 | Macro-F1 |", "|---|---:|---:|---:|"]
        for position in POSITIONS:
            row = data["by_position"][position]
            lines.append(
                f"| {position} | {fmt(row['r_mae_mean'])} | "
                f"{fmt(row['delta_mae_vs_full_mean'])} | {fmt(row['macro_f1_mean'])} |"
            )
        lines.append("")

    lines += [
        "## 5. 缺失模态×缺失比例交互",
        "",
        "表中数值为 C2 强度 MAE；行内升高表示该模态组合对缺失比例更敏感。",
        "",
    ]
    for kind, data in result["factors"].items():
        lines += [
            f"### {kind}",
            "",
            "| 缺失模态 | 10% | 30% | 50% |",
            "|---|---:|---:|---:|",
        ]
        for subset in SUBSETS:
            row = data["subset_ratio"][subset]
            lines.append(
                f"| {subset} | {fmt(row['0.1']['r_mae_mean'])} | "
                f"{fmt(row['0.3']['r_mae_mean'])} | {fmt(row['0.5']['r_mae_mean'])} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root", type=Path, action="append",
        default=None,
        help="Directory containing <candidate>/valid_valid_evaluation.json. Repeatable.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "artifacts" / "q2" / "robustness_analysis",
    )
    args = parser.parse_args()
    roots = args.eval_root or [
        ROOT / "artifacts" / "q2" / "evaluation",
        ROOT / "artifacts" / "q2" / "robustness_eval",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_candidates = load_candidates(roots)
    candidates = [aggregate_candidate(row) for row in raw_candidates]
    if not candidates:
        raise SystemExit("No validation evaluations found")
    result = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "policy": {
            "data": "validation only",
            "official_test_rerun": False,
            "submission_changed": False,
            "pairwise_note": "With only three paired seeds, the exact sign test is descriptive and low-power.",
        },
        "sources": [str(root.resolve()) for root in roots if root.exists()],
        "candidates": [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in candidates
        ],
        "families": summarize_families(candidates),
        "pairwise": paired_comparisons(candidates),
        "factors": factor_summary(candidates),
    }
    (args.out_dir / "q2_robustness_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.out_dir / "q2_robustness_analysis.md", result)
    print(json.dumps({
        "candidates": len(candidates),
        "families": len(result["families"]),
        "out_dir": str(args.out_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()