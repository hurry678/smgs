#!/usr/bin/env python3
"""T03: staged experiment driver (smoke / p0 / screen / optimize / losses / multiseed).

Every candidate shares the frozen encoder, the train-only normalizer, the same mask
banks and the same evaluation protocol. Only the select bank and the clean condition
are read during training or screening; confirm is evaluated after freezing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                                              # noqa: E402
import engine as E                                            # noqa: E402
from protocol_core import publish_c2                          # noqa: E402

METRIC_KEYS = ("accuracy", "macro_f1", "mae", "pearson", "raw_mae", "raw_pearson",
               "worst_condition_mae", "worst_condition_macro_f1", "actual_missing_ratio",
               "samples_with_new_masking", "raw_sign_conflict_rate",
               "mean_projection_magnitude", "argmax_agreement_with_published")


def environment_block(device):
    block = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(), "amp": False, "tf32": False,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cwd": os.getcwd(),
    }
    try:
        block["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             capture_output=True, text=True,
                                             check=True).stdout.strip()
    except Exception:
        block["git_commit"] = None
    return block


def stage_plan(stage, cfg):
    """Pre-registered run list per stage; written to execution_plan.json before training."""
    train = cfg["training"]
    seed = 42
    if stage == "smoke":
        return [
            {"run_id": "smoke_B2_seed42", "architecture": "three_modality_masked_mean_concat_mlp",
             "seed": seed, "config": {}, "train_limit": 128, "max_epochs": 2, "valid_limit": 32},
            {"run_id": "smoke_M1_seed42", "architecture": "bigru_binary_mask_gate_time_pool",
             "seed": seed, "config": {}, "train_limit": 128, "max_epochs": 2, "valid_limit": 32},
        ]
    if stage == "p0":
        base = {"loss": "L1", "optimizer": "O1", "regression_weight": 1.0, "huber_beta": 0.5}
        return [
            {"run_id": "p0_B0_seed42", "architecture": "training_prior", "seed": seed,
             "config": base, "not_trained": True},
            {"run_id": "p0_B2_seed42", "architecture": "three_modality_masked_mean_concat_mlp",
             "seed": seed, "config": base},
            {"run_id": "p0_M1_seed42", "architecture": "bigru_binary_mask_gate_time_pool",
             "seed": seed, "config": base},
            {"run_id": "p0_M1_CEMSE_seed42", "architecture": "bigru_binary_mask_gate_time_pool",
             "seed": seed, "config": {**base, "loss": "L0"}},
            {"run_id": "p0_M1_cleanonly_seed42", "architecture": "bigru_binary_mask_gate_time_pool",
             "seed": seed, "config": {**base, "missing_view": False}},
        ]
    if stage == "screen":
        base = {"loss": "L1", "optimizer": "O1", "regression_weight": 1.0, "huber_beta": 0.5}
        return [
            {"run_id": "s3_B1_seed42", "architecture": "tinybert_masked_mean_linear_dual_head",
             "seed": seed, "config": base},
            {"run_id": "s3_B3_seed42", "architecture": "bigru_content_gate_time_pool",
             "seed": seed, "config": base},
            {"run_id": "s3_M2_seed42", "architecture": "bigru_domain_safe_gap_structure_gate",
             "seed": seed, "config": base},
            {"run_id": "s3_M3_seed42", "architecture": "M2_plus_local_cross_time_attention",
             "seed": seed, "config": base},
        ]
    return []


def load_stage_summary(run_dir, stage):
    path = run_dir / "stage_summaries" / f"{stage}.json"
    if not path.exists():
        raise SystemExit(f"missing previous stage summary: {path}")
    return json.loads(path.read_text())


def ranking(records):
    """Sort by seed-mean R_MAE, then R_F1, then worst-condition MAE (protocol rule 3/4)."""
    def key(item):
        m = item["summary"]
        return (m.get("mae") if m.get("mae") is not None else 9.9,
                -(m.get("macro_f1") or 0.0),
                m.get("worst_condition_mae") if m.get("worst_condition_mae") is not None else 9.9)
    return sorted(records, key=key)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("smoke", "p0", "screen", "optimize", "losses", "multiseed"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-hours", type=float, default=12.0)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        ap.error("run-dir must live under 问题二")
    cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
    cands = json.loads((ROOT / "configs" / "candidates.json").read_text())
    device = torch.device(args.device)
    run_data = E.RunData(run_dir, device=str(device))
    select_bank = D.load_bank(run_dir / "masks" / "select.npz")
    clean_bank = D.load_bank(run_dir / "masks" / "clean.npz")
    specs = {c: s for c, s in zip(select_bank["condition_ids"], select_bank["specs"])}
    plan = stage_plan(args.stage, cfg)
    if args.stage in ("optimize", "losses", "multiseed"):
        plan = derive_stage_plan(args.stage, run_dir, cfg, plan)
    if not plan:
        raise SystemExit(f"empty plan for stage {args.stage}")
    plan_path = run_dir / "stage_summaries" / f"{args.stage}_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({
        "stage": args.stage, "runs": len(plan), "max_hours": args.max_hours,
        "selection_tolerance": {k: cfg["selection"][k] for k in cfg["selection"]
                                if k.endswith("tolerance") or k.startswith("max_")},
        "environment": environment_block(device),
        "note": "written before the first parameter update of this stage",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.stage == "p0":
        (run_dir / "environment.json").write_text(
            json.dumps(environment_block(device), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record_execution_plan(run_dir, args.stage, plan, args.max_hours, str(device), cfg)
    write_environment_lock(run_dir)
    metrics_path = run_dir / "metrics.csv"
    history_path = run_dir / "training_history.jsonl"
    failures_path = run_dir / "failures.jsonl"
    resource_path = run_dir / "resource_usage.csv"
    records = []
    started = time.time()
    for item in plan:
        if (time.time() - started) / 3600.0 > args.max_hours:
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": item["run_id"], "status": "not_run_budget"}) + "\n")
            continue
        try:
            record = execute_run(run_dir, run_data, select_bank, clean_bank, specs, item,
                                 device, cfg, metrics_path, history_path, resource_path)
            records.append(record)
            print(json.dumps({"run_id": item["run_id"], "select_R_MAE": record["summary"]["mae"],
                              "select_R_F1": record["summary"]["macro_f1"],
                              "clean_F1": record["clean"]["macro_f1"],
                              "clean_MAE": record["clean"]["mae"],
                              "best_epoch": record["train"]["best_epoch"]}, ensure_ascii=False),
                  flush=True)
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": item["run_id"], "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
            print(json.dumps({"run_id": item["run_id"], "status": "failed", "error": str(exc)},
                             ensure_ascii=False), flush=True)
    ordered = ranking(records)
    summary = {"stage": args.stage, "completed": len(records), "planned": len(plan),
               "ranking": [{"run_id": r["run_id"], "architecture": r["architecture"],
                            "seed": r["seed"], "config": r["config"],
                            "summary": r["summary"], "clean": r["clean"]} for r in ordered]}
    (run_dir / "stage_summaries" / f"{args.stage}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": args.stage, "completed": len(records),
                      "summary": str(run_dir / "stage_summaries" / f"{args.stage}.json")},
                     ensure_ascii=False))
    return 0


def derive_stage_plan(stage, run_dir, cfg, _unused):
    if stage == "optimize":
        screen = load_stage_summary(run_dir, "screen")
        top = [r for r in screen["ranking"][:2]]
        plan = []
        for rank, entry in enumerate(top):
            arch = entry["architecture"]
            tag = entry["run_id"].split("_")[1]
            base = dict(entry["config"])
            plan.append({"run_id": f"s4_{tag}_O2lr0.01_seed42", "architecture": arch, "seed": 42,
                         "config": {**base, "optimizer": "O2", "lr": 0.01}})
            plan.append({"run_id": f"s4_{tag}_O2lr0.03_seed42", "architecture": arch, "seed": 42,
                         "config": {**base, "optimizer": "O2", "lr": 0.03}})
            plan.append({"run_id": f"s4_{tag}_O3rho0.02_seed42", "architecture": arch, "seed": 42,
                         "config": {**base, "optimizer": "O3", "rho": 0.02, "lr": 5e-4}})
            plan.append({"run_id": f"s4_{tag}_O3rho0.05_seed42", "architecture": arch, "seed": 42,
                         "config": {**base, "optimizer": "O3", "rho": 0.05, "lr": 5e-4}})
            if rank == 0:
                plan.append({"run_id": f"s4_{tag}_O4_seed42", "architecture": arch, "seed": 42,
                             "config": {**base, "optimizer": "O4"}})
        return plan
    if stage == "losses":
        optimize = load_stage_summary(run_dir, "optimize")
        screen = load_stage_summary(run_dir, "screen")
        best_opt = optimize["ranking"][0]
        tag = best_opt["run_id"].split("_")[1]
        arch = best_opt["architecture"]
        base = {**best_opt["config"]}
        return [
            {"run_id": f"s5_{tag}_L1w0.5_seed42", "architecture": arch, "seed": 42,
             "config": {**base, "loss": "L1", "regression_weight": 0.5}},
            {"run_id": f"s5_{tag}_L1w2_seed42", "architecture": arch, "seed": 42,
             "config": {**base, "loss": "L1", "regression_weight": 2.0}},
            {"run_id": f"s5_{tag}_L2_seed42", "architecture": arch, "seed": 42,
             "config": {**base, "loss": "L2"}},
            {"run_id": f"s5_{tag}_L3_seed42", "architecture": arch, "seed": 42,
             "config": {**base, "loss": "L3"}},
            {"run_id": f"s5_{tag}_EMA_seed42", "architecture": arch, "seed": 42,
             "config": {**base, "ema": True, "ema_decay": 0.99}},
        ]
    if stage == "multiseed":
        losses = load_stage_summary(run_dir, "losses")
        optimize = load_stage_summary(run_dir, "optimize")
        best_loss = losses["ranking"][0]
        best_opt = optimize["ranking"][0]
        # the hard gates are relative to B2, so the fallback baseline must carry the same
        # 42/43/44 seed budget as every candidate; otherwise a single-seed baseline would be
        # compared against three-seed candidates
        baseline = {"run_id": "p0_B2_seed42",
                    "architecture": "three_modality_masked_mean_concat_mlp",
                    "config": {"loss": "L1", "optimizer": "O1", "regression_weight": 1.0,
                               "huber_beta": 0.5}}
        pairs = [("loss", best_loss), ("optimizer", best_opt), ("baseline", baseline)]
        plan, seen = [], set()
        for label, entry in pairs:
            tag = entry["run_id"].split("_")[1]
            for seed in (43, 44):
                fingerprint = (entry["architecture"],
                               json.dumps(entry["config"], sort_keys=True), seed)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                plan.append({"run_id": f"s6_{tag}_{label}_seed{seed}",
                             "architecture": entry["architecture"], "seed": seed,
                             "config": entry["config"]})
        return plan
    return []


def record_execution_plan(run_dir, stage, plan, max_hours, device, cfg):
    """Append this stage to run-root execution_plan.json before the first update of the stage."""
    path = run_dir / "execution_plan.json"
    try:
        plan_doc = json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        plan_doc = {"schema": "q2-execution-plan-1", "protocol": cfg.get("protocol"),
                    "stages": {}}
    plan_doc.setdefault("stages", {})[stage] = {
        "runs": len(plan), "run_ids": [item["run_id"] for item in plan],
        "max_hours": max_hours, "device": device,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorded_before_first_update": True,
    }
    plan_doc["budget"] = {"total_neural_runs_cap": cfg["training"].get("max_neural_runs", 36),
                          "wall_clock_hours_cap": cfg["training"].get("max_wall_clock_hours", 12.0),
                          "reserve_hours": cfg["training"].get("reserve_hours", 2.0)}
    plan_doc["selection_tolerance"] = {k: v for k, v in cfg["selection"].items()
                                       if k.endswith("tolerance") or k.startswith("max_")}
    path.write_text(json.dumps(plan_doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_environment_lock(run_dir):
    """Write a pip freeze lock of the exact interpreter once; never overwrite an existing lock."""
    path = run_dir / "environment.lock.txt"
    if path.exists():
        return
    lines = ["# pip freeze of the exact training interpreter",
             f"# executable: {sys.executable}",
             f"# generated_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"# python: {sys.version.split()[0]}", ""]
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, check=False)
        lines.append(freeze.stdout.strip())
        if freeze.returncode != 0:
            lines.append(f"# pip freeze exit={freeze.returncode}: {freeze.stderr.strip()}")
    except Exception as exc:                                   # noqa: BLE001
        lines.append(f"# pip freeze unavailable: {type(exc).__name__}: {exc}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def execute_run(run_dir, run_data, select_bank, clean_bank, specs, item, device, cfg,
                metrics_path, history_path, resource_path):
    out_dir = run_dir / "runs" / item["run_id"]
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {out_dir}")
    out_dir.mkdir(parents=True)
    history_file = out_dir / "history.jsonl"
    config = {"max_epochs": cfg["training"]["max_epochs"], "patience": cfg["training"]["patience"],
              "batch_size": cfg["training"]["batch_size"], "min_delta": cfg["training"]["min_delta"],
              "gradient_clip": cfg["training"]["gradient_clip"],
              "model_kwargs": {"hidden": cfg["training"]["hidden"],
                               "dropout": cfg["training"]["dropout"]}}
    config.update(item.get("config", {}))
    model = E.make_model(item["architecture"], run_data, **config.get("model_kwargs", {})).to(device)
    if item.get("not_trained"):
        train_info = {"architecture": item["architecture"], "seed": item["seed"],
                      "config": config, "epochs_run": 0, "best_epoch": -1,
                      "best_select_R_MAE": None, "history": [], "not_trained": True,
                      "deterministic": False, "tf32": False, "amp": False,
                      "trainable_params": E.trainable_parameter_count(model),
                      "non_encoder_params": E.non_encoder_parameter_count(model),
                      "total_seconds": 0.0, "peak_gpu_bytes": 0}
    else:
        model, train_info = E.train_one(
            run_dir, item["architecture"], config, run_data, select_bank, device=str(device),
            log_path=history_file, seed=item["seed"],
            max_epochs=item.get("max_epochs"), train_limit=item.get("train_limit"),
            valid_limit=item.get("valid_limit"))
    limit = item.get("valid_limit")
    per_condition, predictions = E.evaluate_bank(model, run_data, select_bank, split="valid",
                                                 limit=limit)
    summary = E.summarise_bank(per_condition, specs)
    clean_per, clean_pred = E.evaluate_bank(model, run_data, clean_bank, split="valid", limit=limit)
    clean_summary = next(iter(clean_per.values()))
    checkpoint_sha = E.save_checkpoint(out_dir / "checkpoint.pt", model, {
        "run_id": item["run_id"], "architecture": item["architecture"], "seed": item["seed"],
        "config": config, "best_epoch": train_info["best_epoch"],
        "protocol_sha256": D.sha256_file(ROOT / "configs" / "protocol.json"),
    })
    np.savez_compressed(out_dir / "predictions_select.npz",
                        condition_ids=np.asarray(select_bank["condition_ids"]),
                        logits=np.stack([predictions[c]["logits"] for c in select_bank["condition_ids"]]),
                        raw=np.stack([predictions[c]["raw"] for c in select_bank["condition_ids"]]),
                        published_class=np.stack([predictions[c]["published_class"]
                                                  for c in select_bank["condition_ids"]]),
                        published_intensity=np.stack([predictions[c]["published_intensity"]
                                                      for c in select_bank["condition_ids"]]))
    write_header = (not metrics_path.exists()) or metrics_path.stat().st_size == 0
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "architecture", "seed", "bank", "condition_id"] +
                            list(METRIC_KEYS) + ["per_class_f1", "confusion"])
            f.flush()
        for bank_name, per in (("select", per_condition), ("clean", clean_per)):
            for cid, block in per.items():
                writer.writerow([item["run_id"], item["architecture"], item["seed"], bank_name, cid] +
                                [block.get(k) for k in METRIC_KEYS] +
                                [json.dumps(block.get("per_class_f1"), ensure_ascii=False),
                                 json.dumps(block.get("confusion"))])
    with history_path.open("a", encoding="utf-8") as f:
        for entry in train_info.get("history", []):
            f.write(json.dumps({"run_id": item["run_id"], "architecture": item["architecture"],
                                "seed": item["seed"], **entry}, ensure_ascii=False) + "\n")
    write_resource_header = (not resource_path.exists()) or resource_path.stat().st_size == 0
    with resource_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_resource_header:
            writer.writerow(["run_id", "architecture", "seed", "epochs_run", "best_epoch",
                             "total_seconds", "peak_gpu_bytes", "trainable_params",
                             "non_encoder_params"])
            f.flush()
        writer.writerow([item["run_id"], item["architecture"], item["seed"],
                         train_info["epochs_run"], train_info["best_epoch"],
                         round(train_info["total_seconds"], 3), train_info["peak_gpu_bytes"],
                         train_info["trainable_params"], train_info["non_encoder_params"]])
    record = {"run_id": item["run_id"], "architecture": item["architecture"], "seed": item["seed"],
              "config": item.get("config", {}), "summary": summary, "clean": clean_summary,
              "train": {k: train_info[k] for k in ("best_epoch", "epochs_run", "total_seconds",
                                                   "peak_gpu_bytes", "trainable_params",
                                                   "non_encoder_params")},
              "checkpoint_sha256": checkpoint_sha}
    (out_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return record


if __name__ == "__main__":
    raise SystemExit(main())