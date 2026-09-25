#!/usr/bin/env python3
"""T01: build the pre-registered select/confirm/position/coupling mask banks.

The banks are functions of (domain, observed, sample_id) only: re-running them in a
different sample order reproduces identical per-sample digests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="run_dir/data produced by prepare_data.py")
    ap.add_argument("--protocol", type=Path, default=ROOT / "configs" / "protocol.json")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    data_dir = args.data.resolve()
    if not run_dir.is_relative_to(ROOT):
        ap.error("run-dir must live under 问题二")
    cfg = json.loads(args.protocol.read_text())
    valid = D.load_npz(data_dir / "valid.npz")
    domain, observed = valid["domain"].astype(bool), valid["observed"].astype(bool)
    ids = [str(x) for x in valid["id"]]
    manifest = {"schema": "q2-mask-bank-manifest-1", "protocol_sha256": D.sha256_file(args.protocol),
                "data_dir": str(data_dir), "n_samples": len(ids), "T": int(domain.shape[1]),
                "banks": {}}
    for bank in ("select", "confirm", "position", "coupling"):
        built = D.build_bank(bank, domain, observed, ids, cfg)
        path = D.save_bank(run_dir, built, ids)
        rec_path = run_dir / "masks" / f"{bank}_intervals.jsonl"
        with rec_path.open("w", encoding="utf-8") as f:
            for rec in built["records"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = {}
        for rec in built["records"]:
            status[rec["status"]] = status.get(rec["status"], 0) + 1
        manifest["banks"][bank] = {
            "conditions": int(len(built["condition_ids"])),
            "bank_sha256": built["bank_sha256"],
            "npz_sha256": path,
            "npz": str(Path("masks") / f"{bank}.npz"),
            "intervals": str(Path("masks") / f"{bank}_intervals.jsonl"),
            "condition_sha256": list(built["condition_sha256"]),
            "condition_ids": list(built["condition_ids"]),
            "status_counts": status,
        }
        print(json.dumps({"bank": bank, "conditions": len(built["condition_ids"]),
                          "bank_sha256": built["bank_sha256"]}, ensure_ascii=False), flush=True)
    clean = D.clean_bank(observed)
    manifest["banks"]["clean"] = {"conditions": 1, "bank_sha256": clean["bank_sha256"],
                                 "condition_sha256": list(clean["condition_sha256"]),
                                 "condition_ids": list(clean["condition_ids"])}
    D.save_bank(run_dir, clean, ids)
    (run_dir / "mask_bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mask_bank_manifest": str(run_dir / "mask_bank_manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())