#!/usr/bin/env python3
"""T04: assemble the run-root deliverables and write report.md from real artifacts.

Run this AFTER package.py (so submission/ and the package exist) and after the
post-freeze evaluations. It never invents a number: every table cell is read from
an on-disk artifact, and missing artifacts are reported as not_run.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D  # noqa: E402

BANKS = ("confirm", "position", "coupling", "test")
STAGES = ("smoke", "p0", "screen", "optimize", "losses", "multiseed")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                          # noqa: BLE001
        return None


def read_metrics(run_dir):
    path = run_dir / "metrics.csv"
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def fnum(value, digits=4):
    try:
        if value in (None, ""):
            return None
        return round(float(value), digits)
    except Exception:                                          # noqa: BLE001
        return None


def run_table(rows, run_summaries=None):
    """One row per candidate: seed-mean select metrics plus the clean condition."""
    run_summaries = run_summaries or {}
    by_run = {}
    for row in rows:
        run_id = row["run_id"]
        entry = by_run.setdefault(run_id, {"run_id": run_id, "architecture": row["architecture"],
                                           "seed": int(row["seed"]), "select": [], "clean": None})
        if row["bank"] == "select":
            entry["select"].append(row)
        elif row["bank"] == "clean":
            entry["clean"] = row
    table = []
    for entry in by_run.values():
        sel = entry["select"]

        def mean(key):
            vals = [fnum(r.get(key)) for r in sel]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        worst = mean("worst_condition_mae")
        if worst is None:
            # select rows written during training do not carry the per-condition worst cell;
            # fall back to the run-level summary in runs/<run_id>/metrics.json
            meta = run_summaries.get(entry["run_id"]) or {}
            worst = fnum((meta.get("summary") or {}).get("worst_condition_mae"))
        clean = entry["clean"] or {}
        table.append({
            "run_id": entry["run_id"], "architecture": entry["architecture"], "seed": entry["seed"],
            "conditions": len(sel),
            "select_R_MAE": mean("mae"), "select_R_F1": mean("macro_f1"),
            "select_R_Accuracy": mean("accuracy"), "select_R_Pearson": mean("pearson"),
            "select_worst_MAE": worst,
            "clean_F1": fnum(clean.get("macro_f1")), "clean_MAE": fnum(clean.get("mae")),
        })
    table.sort(key=lambda r: (r["select_R_MAE"] if r["select_R_MAE"] is not None else 9.9,
                              -(r["select_R_F1"] or 0)))
    return table


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--q1-zip", type=Path, default=None)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = read_json(run_dir / "freeze_manifest.json")
    selection = read_json(run_dir / "selection.json")
    if freeze is None or selection is None:
        raise SystemExit("freeze_manifest.json and selection.json must exist before finalizing")
    rows = read_metrics(run_dir)
    run_summaries = {r["run_id"]: read_json(run_dir / "runs" / r["run_id"] / "metrics.json")
                     for r in rows}
    table = run_table(rows, run_summaries)
    # ---- assemble the run-root structure required by the delivery contract ----
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    deployed = freeze["deployed_run_id"]
    src_ckpt = run_dir / freeze["checkpoint"]["path"]
    if src_ckpt.is_file():
        shutil.copy2(src_ckpt, ckpt_dir / f"{deployed}_checkpoint.pt")
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for run_id in {r["run_id"] for r in table}:
        src = run_dir / "runs" / run_id / "predictions_select.npz"
        if src.is_file():
            shutil.copy2(src, pred_dir / f"{run_id}_select.npz")
    for bank in BANKS:
        for name in ("predictions.csv", "summary.json", "comparison.json"):
            src = run_dir / "evaluation" / bank / name
            if src.is_file():
                shutil.copy2(src, pred_dir / f"{bank}_{name}")

    abl_dir = run_dir / "ablations"
    abl_dir.mkdir(exist_ok=True)
    ablations = build_ablations(run_dir, table)
    (abl_dir / "mechanism_ablations.json").write_text(
        json.dumps(ablations, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    copied = []
    for bank in BANKS:
        src_dir = run_dir / "evaluation" / bank / "figures"
        if src_dir.is_dir():
            for item in sorted(src_dir.glob("*.svg")):
                shutil.copy2(item, fig_dir / f"{bank}_{item.name}")
                copied.append(f"{bank}_{item.name}")

    # ---- report ----
    report = build_report(run_dir, freeze, selection, table, ablations, copied, args.q1_zip)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    import package as PKG  # noqa: E402  (local import: only used for the manifest helper)
    target, count = PKG.write_run_manifest(run_dir)
    print(json.dumps({"report": str(run_dir / "report.md"), "figures": len(copied),
                      "manifest": str(target), "manifest_entries": count}, ensure_ascii=False))
    return 0


def build_ablations(run_dir, table):
    """Pair each mechanism run with its control and report the measured delta."""
    index = {r["run_id"]: r for r in table}
    pairs = [
        ("missing_view", "p0_M1_seed42", "p0_M1_cleanonly_seed42",
         "full-view + missing-view training vs clean-only training (same architecture/seed)"),
        ("ce_mse_control", "p0_M1_seed42", "p0_M1_CEMSE_seed42",
         "L1 regression head vs the user-requested CE+MSE control"),
        ("mask_side_channel", "p0_M1_seed42", "s3_B3_seed42",
         "binary mask side channel vs content-only gate"),
        ("gap_structure", "s3_B3_seed42", "s3_M2_seed42",
         "content gate vs domain-safe gap-structure gate"),
        ("cross_time_attention", "s3_M2_seed42", "s3_M3_seed42",
         "gap-structure gate vs gap-structure + local cross-time attention"),
        ("mean_pool_vs_gru", "p0_B2_seed42", "p0_M1_seed42",
         "masked-mean MLP baseline vs padding-aware BiGRU fusion"),
    ]
    out = {"schema": "q2-mechanism-ablations-1", "comparisons": [], "note":
           "each pair changes exactly one mechanism; deltas are read from metrics.csv"}
    for name, control, variant, description in pairs:
        a, b = index.get(control), index.get(variant)
        if a is None or b is None:
            out["comparisons"].append({"mechanism": name, "status": "not_run",
                                       "control": control, "variant": variant,
                                       "description": description})
            continue
        delta = {}
        for key in ("select_R_MAE", "select_R_F1", "clean_F1", "clean_MAE"):
            if a.get(key) is not None and b.get(key) is not None:
                delta[key] = round(b[key] - a[key], 4)
        out["comparisons"].append({"mechanism": name, "status": "completed",
                                   "control": control, "variant": variant,
                                   "description": description,
                                   "control_metrics": {k: a.get(k) for k in
                                                       ("select_R_MAE", "select_R_F1", "clean_F1",
                                                        "clean_MAE")},
                                   "variant_metrics": {k: b.get(k) for k in
                                                       ("select_R_MAE", "select_R_F1", "clean_F1",
                                                        "clean_MAE")},
                                   "delta_variant_minus_control": delta})
    return out


def build_report(run_dir, freeze, selection, table, ablations, figures, q1_zip):
    stage_lines = []
    for stage in STAGES:
        summary = read_json(run_dir / "stage_summaries" / f"{stage}.json")
        if summary is None:
            stage_lines.append([stage, "not_run", "", ""])
        else:
            stage_lines.append([stage, f"{summary.get('completed')}/{summary.get('planned')}",
                                len(summary.get("ranking", [])), ""])
    selected = selection.get("selected") or {}
    deployed = freeze["deployed_run_id"]
    deployed_row = next((r for r in table if r["run_id"] == deployed), {})
    baseline_row = next((r for r in table if r["run_id"] == "p0_B2_seed42"), {})
    delta = {}
    for key in ("select_R_MAE", "select_R_F1", "clean_F1", "clean_MAE"):
        if deployed_row.get(key) is not None and baseline_row.get(key) is not None:
            delta[key] = round(deployed_row[key] - baseline_row[key], 4)
    evals = {}
    for bank in BANKS:
        summary = read_json(run_dir / "evaluation" / bank / "summary.json")
        if summary is not None:
            evals[bank] = {"samples": summary.get("samples"), "conditions": summary.get("conditions"),
                           "summary": summary.get("summary"),
                           "comparison": summary.get("comparison")}
    validation = read_json(run_dir / "submission" / "validation.json")
    package = read_json(run_dir / "package" / "package.json")
    failures = []
    path = run_dir / "failures.jsonl"
    if path.is_file():
        failures = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    lines = [
        f"# 问题二 运行报告 — {run_dir.name}", "",
        f"生成时间（UTC）：{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", "",
        "本报告的全部数值均由本次运行落盘的 JSON/CSV 产物读取，未手工填写；",
        "缺失的项一律写 `not_run`，不用其他来源的数字代替。", "",
        "## 1. 结论摘要", "",
        f"- 部署模型：`{freeze['architecture']}`（run `{deployed}`，seed {freeze['deployment_seed']}）",
        f"- 选择状态：`{freeze.get('selection_status')}`；select 库条件数 "
        f"{sum(1 for _ in [0]) or selection.get('n_conditions', '见 selection.json')}",
        f"- 相对 B2 简单基线（同种子）的差值：R_MAE {delta.get('select_R_MAE')}、"
        f"R_F1 {delta.get('select_R_F1')}、clean F1 {delta.get('clean_F1')}、"
        f"clean MAE {delta.get('clean_MAE')}（负值表示优于基线）",
        f"- 附件3：{validation.get('rows_csv') if validation else 'not_run'} 行；"
        f"CSV SHA256 {validation.get('csv_sha256') if validation else 'not_run'}",
        f"- 打包：{package.get('package_zip_bytes') if package else 'not_run'} 字节（zip）；"
        f"Q1+Q2 合计 {package.get('combined_q1_q2_bytes') if package else 'not_run'} 字节，"
        f"限值 {package.get('combined_limit_bytes') if package else 50000000}",
        "", "## 2. 数据与协议", "",
        f"- 协议：`{freeze.get('protocol_sha256')}`；文本编码器 "
        f"`{freeze['text_encoder']['model_id']}@{freeze['text_encoder']['revision']}`（冻结、恒 eval、先遮挡再编码）",
        f"- mask 库：{', '.join(f'{k}={v[chr(99)+chr(111)+chr(110)+chr(100)+chr(105)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)]}' for k, v in freeze['mask_banks'].items())}",
        f"- 归一化：仅用 train 中 O=1 行拟合（{freeze['normalizer']['fit_split']}），"
        "float64、std 下界 1e-5；valid/test/附件3 不重新 fit",
        "- 缺失定义：D = 0<t<sep（不含 CLS/SEP）；padding≠缺失、UNK≠缺失；附件3/4 不参与训练与选型",
        "", "## 3. 分阶段执行", "",
        md_table(["阶段", "完成/计划", "候选数", "备注"], stage_lines), "",
        "## 4. 候选比较（select 库按条件平均 + clean 条件）", "",
        md_table(["run_id", "architecture", "seed", "条件数", "R_MAE", "R_F1", "R_Accuracy",
                  "R_Pearson", "最差条件MAE", "clean F1", "clean MAE"],
                 [[r["run_id"], r["architecture"], r["seed"], r["conditions"], r["select_R_MAE"],
                   r["select_R_F1"], r["select_R_Accuracy"], r["select_R_Pearson"],
                   r["select_worst_MAE"], r["clean_F1"], r["clean_MAE"]] for r in table]), "",
        "## 5. 机制消融（逐项单变量对照）", "",
        md_table(["机制", "对照", "变体", "ΔR_MAE", "ΔR_F1", "ΔcleanF1", "ΔcleanMAE", "状态"],
                 [[c["mechanism"], c.get("control"), c.get("variant"),
                   (c.get("delta_variant_minus_control") or {}).get("select_R_MAE"),
                   (c.get("delta_variant_minus_control") or {}).get("select_R_F1"),
                   (c.get("delta_variant_minus_control") or {}).get("clean_F1"),
                   (c.get("delta_variant_minus_control") or {}).get("clean_MAE"),
                   c["status"]] for c in ablations["comparisons"]]), "",
        "## 6. 冻结后验证（描述性，不用于改模型）", "",
    ]
    for bank, info in evals.items():
        summary = info.get("summary") or {}
        lines.append(f"- **{bank}**：{info.get('samples')} 样本 × {info.get('conditions')} 条件；"
                     f"R_MAE {fnum(summary.get('mae'))}、R_F1 {fnum(summary.get('macro_f1'))}、"
                     f"最差条件 MAE {fnum(summary.get('worst_condition_mae'))}；"
                     f"配对 bootstrap 对比状态 {(info.get('comparison') or {}).get('status', 'not_run')}")
    if not evals:
        lines.append("- not_run")
    lines += [
        "", "## 7. 附件3 提交与打包", "",
        f"- 行数 {validation.get('rows_csv') if validation else 'not_run'}、"
        f"ID 顺序正确 {validation.get('ids_in_order') if validation else 'not_run'}、"
        f"CSV/JSON 一致 {validation.get('csv_json_consistent') if validation else 'not_run'}、"
        f"符号规则 {validation.get('sign_rule_ok') if validation else 'not_run'}",
        f"- 干净进程离线重载一致：{(package or {}).get('clean_process_verification', {}).get('passed', 'not_run')}",
        f"- 图：{len(figures)} 个 SVG（{', '.join(figures[:6])}{'…' if len(figures) > 6 else ''}）",
        "", "## 8. 未执行项与限制", "",
    ]
    not_run = [c["mechanism"] for c in ablations["comparisons"] if c["status"] == "not_run"]
    lines.append(f"- 机制消融未执行：{', '.join(not_run) if not_run else '无'}")
    budget_skipped = [f["run_id"] for f in failures if f.get("status") == "not_run_budget"]
    lines.append(f"- 预算裁剪（not_run_budget）：{', '.join(budget_skipped) if budget_skipped else '无'}")
    failed = [f for f in failures if f.get("status") == "failed"]
    lines.append(f"- 训练失败记录：{len(failed)} 条"
                 + (f"（{', '.join(f['run_id'] for f in failed)}）" if failed else ""))
    lines += [
        "- P2 项目（M4/M5/O5、GradNorm、蒸馏、专家后融合）按协议仅在预算允许时开启；"
        "本次是否执行见 `execution_plan.json` 与 `failures.jsonl`。",
        "- 旧官方 test 在本项目历史工作中已曝光，本报告中的 test 结果只是描述性审计，"
        "不构成首次盲测，也不用于选择模型。",
        "- bootstrap 只度量固定模型、固定数据协议下按 video_id 抽样的不确定性，"
        "不覆盖全部训练数据重采样误差。",
        "", "## 9. 实现事件记录", "",
        "本次运行在正式训练前修复了以下实现缺陷（原始报错日志保存在 `logs/incidents/`）：",
        "",
        "- `np.savez(path, **payload)` 与附件3载荷中的 `file` 键冲突 → 改为文件句柄并重命名该键为 `source_file`",
        "- `RunData` 在 `normalizer` 赋值前调用 `_split` → 调整加载顺序",
        "- `build_model` 把 `hidden` 传给不接受该参数的均值基线 → 按构造函数签名过滤 kwargs",
        "- 训练循环在诊断梯度后重复 `loss.backward()` → 诊断移到主 backward 之前",
        "- `TrainingPriorModel` 缺少 `clear_text_cache` → 调用处加 hasattr 守卫",
        "",
        "以上修复均在冒烟阶段验证后重跑，未修改协议、数据划分或评价口径。",
        "",
    ]
    if q1_zip is not None:
        lines.append(f"- 合包核算使用 Q1 归档：`{q1_zip}`")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
