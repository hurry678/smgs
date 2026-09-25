#!/usr/bin/env python3
"""Re-run a single arm of an already-executed stage.

Used when a mechanism arm is found to be invalid because of a real implementation bug
(recorded as such in the report). The invalidated arm directory is *archived*, never
deleted, so the audit trail keeps the original (wrong) numbers; the re-run then writes a
fresh ``runs/<run_id>/metrics.json`` and appends to the run-level
metrics.csv / resource_usage.csv / training_history.jsonl exactly like run_experiments.py.
Finally the stage summary is regenerated so downstream reports read the corrected ranking.

Archived arms are moved to ``runs/_invalidated/<run_id>__<UTC stamp>/``. That path is two
levels deep, so the ``runs/*/metrics.json`` globs used by the stage summaries and by
finalize_run.py never pick them up again.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D          # noqa: E402
import engine as E        # noqa: E402
import run_experiments as RX  # noqa: E402

STAGE_PREFIX = {"smoke": ("smoke_",), "p0": ("p0_",), "screen": ("s3_",),
                "optimize": ("s4_",), "losses": ("s5_",), "multiseed": ("s6_",)}


def archive_invalidated_arm(run_dir, run_id, reason):
    """Move an existing (known-bad) arm aside and log why. Returns the archive path."""
    out_dir = run_dir / "runs" / run_id
    if not out_dir.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = run_dir / "runs" / "_invalidated" / f"{run_id}__{stamp}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir), str(dest))
    entry = {"run_id": run_id, "status": "invalidated_rerun", "reason": reason,
             "archived_to": str(dest.relative_to(run_dir)), "archived_at_utc": stamp}
    with (run_dir / "failures.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return dest


def refresh_stage_summary(run_dir, stage):
    prefixes = STAGE_PREFIX[stage]
    records = []
    for metrics_file in sorted((run_dir / "runs").glob("*/metrics.json")):
        run_id = metrics_file.parent.name
        if any(run_id.startswith(p) for p in prefixes):
            records.append(json.loads(metrics_file.read_text()))
    ordered = RX.ranking(records)
    plan = {}
    plan_file = run_dir / "stage_summaries" / f"{stage}_plan.json"
    if plan_file.is_file():
        plan = json.loads(plan_file.read_text())
    summary = {"stage": stage, "completed": len(records),
               "planned": plan.get("runs", len(records)),
               "ranking": [{"run_id": r["run_id"], "architecture": r["architecture"],
                            "seed": r["seed"], "config": r["config"],
                            "summary": r["summary"], "clean": r["clean"]} for r in ordered]}
    path = run_dir / "stage_summaries" / f"{stage}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stage", required=True, choices=tuple(STAGE_PREFIX))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--architecture", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True, help="JSON object with the arm config")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--reason", default="unspecified implementation defect in the first run")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
    archived = archive_invalidated_arm(run_dir, args.run_id, args.reason)
    if archived is not None:
        print(json.dumps({"archived_invalidated_arm": str(archived)}, ensure_ascii=False), flush=True)
    run_data = E.RunData(run_dir, device=str(device))
    select_bank = D.load_bank(run_dir / "masks" / "select.npz")
    clean_bank = D.load_bank(run_dir / "masks" / "clean.npz")
    specs = {c: s for c, s in zip(select_bank["condition_ids"], select_bank["specs"])}
    item = {"run_id": args.run_id, "architecture": args.architecture, "seed": args.seed,
            "config": json.loads(args.config)}
    record = RX.execute_run(run_dir, run_data, select_bank, clean_bank, specs, item, device, cfg,
                            run_dir / "metrics.csv", run_dir / "training_history.jsonl",
                            run_dir / "resource_usage.csv")
    path, summary = refresh_stage_summary(run_dir, args.stage)
    print(json.dumps({"run_id": record["run_id"], "select_R_MAE": record["summary"]["mae"],
                      "select_R_F1": record["summary"]["macro_f1"],
                      "clean_F1": record["clean"]["macro_f1"],
                      "clean_MAE": record["clean"]["mae"],
                      "stage_summary": str(path), "stage_completed": summary["completed"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
