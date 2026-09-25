#!/usr/bin/env python3
"""T04: freeze the selected configuration and the published policy into a manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D  # noqa: E402


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


STAGE_KEYS = (("S1_smoke", "smoke"), ("S2_p0", "p0"), ("S3_screen", "screen"),
              ("S4_optimize", "optimize"), ("S5_losses", "losses"), ("S6_multiseed", "multiseed"))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:                                          # noqa: BLE001
        return None


def stage_status(run_dir):
    """Derive stage completion from on-disk evidence; never assume a stage ran."""
    status = {}
    preflight_path = next((p for p in (run_dir / "preflight.json",
                                       ROOT / "runs" / "preflight_server" / "preflight.json")
                           if Path(p).is_file()), None)
    preflight = _read_json(preflight_path) if preflight_path else None
    if preflight is None:
        status["S0_preflight"] = "missing"
    else:
        status["S0_preflight"] = ("passed" if preflight.get("passed") else
                                  str(preflight.get("status") or "failed"))
        status["S0_preflight_source"] = str(Path(preflight_path).relative_to(ROOT))
    for key, stage in STAGE_KEYS:
        summary = _read_json(run_dir / "stage_summaries" / f"{stage}.json")
        if summary is None:
            status[key] = "not_run"
            continue
        completed, planned = summary.get("completed"), summary.get("planned")
        if completed is None:
            status[key] = "unknown"
        elif planned is not None and completed < planned:
            status[key] = f"partial ({completed}/{planned})"
        else:
            status[key] = f"completed ({completed}/{planned})"
    done = [b for b in ("confirm", "position", "coupling", "test")
            if (run_dir / "evaluation" / b / "summary.json").is_file()]
    status["S7_post_freeze"] = f"completed: {', '.join(done)}" if done else "not_run"
    status["S8_attachment3"] = ("completed" if (run_dir / "submission" / "q2_predictions.csv").is_file()
                                else "not_run")
    failures = run_dir / "failures.jsonl"
    if failures.is_file() and failures.stat().st_size:
        counts = {}
        for line in failures.read_text(encoding="utf-8").splitlines():
            if line.strip():
                key = json.loads(line).get("status", "unknown")
                counts[key] = counts.get(key, 0) + 1
        status["failures_jsonl"] = counts
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    selected = selection["selected"]
    seed = args.seed or selection["default_deployment_seed"]
    members = []
    for run_id in selected.get("run_ids", []):
        metrics_file = run_dir / "runs" / run_id / "metrics.json"
        if metrics_file.exists():
            members.append({"run_id": run_id, "seed": json.loads(metrics_file.read_text())["seed"]})
    chosen = [m for m in members if m["seed"] == seed]
    if not chosen:
        raise SystemExit(f"no completed run for deployment seed {seed}")
    chosen = chosen[0]
    run_out = run_dir / "runs" / chosen["run_id"]
    checkpoint = run_out / "checkpoint.pt"
    mask_manifest = json.loads((run_dir / "mask_bank_manifest.json").read_text())
    data_manifest = json.loads((run_dir / "data_manifest.json").read_text())
    resources = json.loads((ROOT / "resources" / "manifest.json").read_text())
    manifest = {
        "schema": "q2-freeze-manifest-1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "protocol_sha256": D.sha256_file(ROOT / "configs" / "protocol.json"),
        "architecture": selected["architecture"], "config": selected["config"],
        "selection_status": selection["status"], "selection_bank": "select",
        "deployment_seed": seed, "member_runs": members,
        "deployed_run_id": chosen["run_id"],
        "checkpoint": {"path": str(Path("runs") / chosen["run_id"] / "checkpoint.pt"),
                       "sha256": D.sha256_file(checkpoint),
                       "bytes": checkpoint.stat().st_size,
                       "contains_encoder": False,
                       "encoder_shared": "resources/bert_shared"},
        "text_encoder": {
            "model_id": cfg["text_encoder"]["model_id"], "revision": cfg["text_encoder"]["revision"],
            "resource_dir": cfg["text_encoder"]["resource_dir"],
            "files": {item["path"]: item["sha256"] for item in resources["files"]},
            "frozen": True, "always_eval": True, "mask_before_encoding": True,
        },
        "normalizer": {"path": "data/normalizer.npz",
                       "sha256": D.sha256_file(run_dir / "data" / "normalizer.npz"),
                       "fit_split": data_manifest["normalization_fit_split"]},
        "input_protocol": {
            "mask_protocol": cfg["mask"]["protocol"],
            "domain": cfg["mask"]["domain"],
            "unobserved": cfg["mask"]["audio_vision_unobserved"],
            "text_input": cfg["data"]["text_input"],
            "batch_size": cfg["training"]["batch_size"],
        },
        "mask_banks": {name: {"bank_sha256": info["bank_sha256"],
                              "conditions": info["conditions"]}
                       for name, info in mask_manifest["banks"].items()},
        "publication": {
            "policy": cfg["publication"]["default_policy"],
            "intensity_range": cfg["publication"]["intensity_range"],
            "nonneutral_epsilon": cfg["publication"]["nonneutral_epsilon"],
            "csv_decimals": cfg["publication"]["csv_decimals"],
            "csv_columns": cfg["publication"]["csv_columns"],
            "rule": "argmax class; neutral intensity 0; positive max(clip(r,-3,3),1e-4); "
                    "negative min(clip(r,-3,3),-1e-4)",
        },
        "stage_status": stage_status(run_dir),
        "notes": [
            "confirm/position/coupling/test are evaluated only after this manifest is written",
            "no attachment-3 or attachment-4 data was used for training, selection or thresholds",
            "the historical official test split was already exposed in earlier project work; "
            "the new test report is descriptive, not a first blind test",
        ],
    }
    (run_dir / "freeze_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"freeze_manifest": str(run_dir / "freeze_manifest.json"),
                      "deployed_run_id": chosen["run_id"], "seed": seed,
                      "checkpoint_sha256": manifest["checkpoint"]["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())