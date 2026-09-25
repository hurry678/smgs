#!/usr/bin/env python3
"""T03: apply the frozen selection rules and write selection.json.

Only select-bank and clean validation evidence is used. confirm/test are never read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D  # noqa: E402


def group_key(record):
    return json.dumps({"architecture": record["architecture"], "config": record["config"]},
                      sort_keys=True, ensure_ascii=False)


def collect(run_dir):
    groups = {}
    for metrics_file in sorted((run_dir / "runs").glob("*/metrics.json")):
        record = json.loads(metrics_file.read_text())
        key = group_key(record)
        groups.setdefault(key, {"architecture": record["architecture"],
                                "config": record["config"], "runs": []})
        groups[key]["runs"].append(record)
    return groups




def aggregate(entry, cfg):
    runs = entry["runs"]
    seeds = sorted({r["seed"] for r in runs})
    per_seed = {}
    for seed in seeds:
        rs = [r for r in runs if r["seed"] == seed]
        per_seed[seed] = {
            "R_MAE": float(sum(r["summary"]["mae"] for r in rs) / len(rs)),
            "R_F1": float(sum(r["summary"]["macro_f1"] for r in rs) / len(rs)),
            "R_Accuracy": float(sum(r["summary"]["accuracy"] for r in rs) / len(rs)),
            "worst_condition_MAE": float(max(r["summary"].get("worst_condition_mae", 9.9) for r in rs)),
            "clean_F1": float(sum(r["clean"]["macro_f1"] for r in rs) / len(rs)),
            "clean_MAE": float(sum(r["clean"]["mae"] for r in rs) / len(rs)),
        }
    keys = ("R_MAE", "R_F1", "R_Accuracy", "worst_condition_MAE", "clean_F1", "clean_MAE")
    mean = {k: float(sum(per_seed[s][k] for s in seeds) / len(seeds)) for k in keys}
    missing = [r["run_id"] for r in runs
               if not (Path(run_dir_of(r)) / "checkpoint.pt").exists()]
    if missing:
        raise RuntimeError(f"completed runs without a deployable checkpoint: {missing}")
    size = sum((Path(run_dir_of(r)) / "checkpoint.pt").stat().st_size for r in runs)
    return {"seeds": seeds, "per_seed": per_seed, "mean": mean, "checkpoint_bytes": int(size),
            "run_ids": [r["run_id"] for r in runs],
            "non_encoder_params": runs[0]["train"]["non_encoder_params"],
            "peak_gpu_bytes": max(r["train"]["peak_gpu_bytes"] for r in runs)}


def run_dir_of(record):
    return Path(record["_dir"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
    sel = cfg["selection"]
    groups = collect(run_dir)
    for key, entry in groups.items():
        for r in entry["runs"]:
            r["_dir"] = str(run_dir / "runs" / r["run_id"])
    baseline_key = None
    for key, entry in groups.items():
        if entry["architecture"] == "three_modality_masked_mean_concat_mlp" and \
                any(r["run_id"] == "p0_B2_seed42" for r in entry["runs"]):
            baseline_key = key
    if baseline_key is None:
        raise SystemExit("baseline B2 (p0_B2_seed42) not found; cannot apply selection rules")
    table = {}
    for key, entry in groups.items():
        agg = aggregate(entry, cfg)
        table[key] = {"architecture": entry["architecture"], "config": entry["config"], **agg}
    base = table[baseline_key]["mean"]
    decisions = []
    for key, info in table.items():
        if key == baseline_key:
            continue
        mean = info["mean"]
        gates = {
            "clean_macro_f1_drop": base["clean_F1"] - mean["clean_F1"],
            "clean_mae_increase": mean["clean_MAE"] - base["clean_MAE"],
            "missing_f1_drop": base["R_F1"] - mean["R_F1"],
            "worst_condition_mae_increase": mean["worst_condition_MAE"] - base["worst_condition_MAE"],
        }
        passed = (gates["clean_macro_f1_drop"] <= sel["max_clean_f1_drop_vs_B2"] and
                  gates["clean_mae_increase"] <= sel["max_clean_mae_increase_vs_B2"] and
                  gates["missing_f1_drop"] <= sel["max_missing_f1_drop_vs_B2"] and
                  gates["worst_condition_mae_increase"] <= sel["max_worst_mae_increase_vs_B2"])
        decisions.append({
            "key": key, "architecture": info["architecture"], "config": info["config"],
            "seeds": info["seeds"], "run_ids": info["run_ids"],
            "complete_three_seed": len(info["seeds"]) >= 3,
            "R_MAE": mean["R_MAE"], "R_F1": mean["R_F1"],
            "worst_condition_MAE": mean["worst_condition_MAE"],
            "clean_F1": mean["clean_F1"], "clean_MAE": mean["clean_MAE"],
            "checkpoint_bytes": info["checkpoint_bytes"],
            "gates_vs_B2": gates, "passes_hard_gates": bool(passed),
            "status": "qualified" if passed else "tradeoff_only",
            "reject_reason": None if passed else "fails the pre-registered relative-to-B2 gate",
        })
    base_decision = {
        "key": baseline_key, "architecture": table[baseline_key]["architecture"],
        "config": table[baseline_key]["config"], "seeds": table[baseline_key]["seeds"],
        "run_ids": table[baseline_key]["run_ids"],
        "complete_three_seed": len(table[baseline_key]["seeds"]) >= 3,
        "R_MAE": base["R_MAE"], "R_F1": base["R_F1"],
        "worst_condition_MAE": base["worst_condition_MAE"], "clean_F1": base["clean_F1"],
        "clean_MAE": base["clean_MAE"], "checkpoint_bytes": table[baseline_key]["checkpoint_bytes"],
        "gates_vs_B2": None, "passes_hard_gates": True, "status": "fallback_baseline",
        "reject_reason": None,
    }
    qualified = [d for d in decisions if d["passes_hard_gates"] and d["complete_three_seed"]]
    qualified.sort(key=lambda d: (d["R_MAE"], -d["R_F1"], d["worst_condition_MAE"],
                                  d["checkpoint_bytes"]))
    selected = qualified[0] if qualified else base_decision
    tie = [d for d in qualified if abs(d["R_MAE"] - selected["R_MAE"]) <= sel["mae_tie_tolerance"]]
    tie.sort(key=lambda d: (-d["R_F1"], d["worst_condition_MAE"], d["checkpoint_bytes"]))
    if tie and tie[0] is not selected:
        selected = tie[0]
    status = "selected_three_seed" if selected["complete_three_seed"] else "provisional_single_seed"
    provisional = [d for d in decisions if not d["complete_three_seed"]]
    result = {
        "schema": "q2-selection-1", "bank": "select", "baseline": base_decision,
        "candidates": decisions, "engineering_tie_set": [d["key"] for d in tie],
        "selected": selected, "status": status,
        "default_deployment_seed": sel["default_deployment_seed"],
        "provisional_candidates": [d["key"] for d in provisional],
        "rules_applied": {
            "primary": sel["primary"],
            "mae_tie_tolerance": sel["mae_tie_tolerance"],
            "f1_tie_tolerance": sel["f1_tie_tolerance"],
            "worst_mae_tie_tolerance": sel["worst_mae_tie_tolerance"],
            "gates": {k: sel[k] for k in sel if k.startswith("max_")},
        },
        "limitations": "Selection uses only the select bank and clean valid condition; "
                       "confirm/test are reported after freezing and cannot change this choice.",
    }
    (run_dir / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"selected": selected["key"], "status": status,
                      "R_MAE": selected["R_MAE"], "R_F1": selected["R_F1"],
                      "seeds": selected["seeds"],
                      "selection": str(run_dir / "selection.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())