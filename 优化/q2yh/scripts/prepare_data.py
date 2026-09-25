#!/usr/bin/env python3
"""T01: split the competition pickles into per-split archives plus a train-fitted normalizer.

Writes only inside the new run directory. The training entry point never receives the
test or Attachment-3 path, and the normalizer is fitted on train observations only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                                    # noqa: E402
from protocol_core import interface_masks           # noqa: E402


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def add_interface(payload):
    domain, observed = interface_masks(payload["text_bert"], payload["audio"], payload["vision"])
    payload["domain"] = domain
    payload["observed"] = observed
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--attachment3-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT) or run_dir == ROOT:
        ap.error("run-dir must be a new directory under 问题二")
    data_dir = run_dir / "data"
    if data_dir.exists():
        ap.error("data directory already exists; use a new run id")
    data_dir.mkdir(parents=True)
    protocol_sha = D.sha256_file(ROOT / "configs" / "protocol.json")
    cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
    manifest = {
        "schema": "q2-data-manifest-1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(), "protocol_sha256": protocol_sha,
        "aligned_source": str(args.aligned), "aligned_sha256": D.sha256_file(args.aligned),
        "attachment3_dir": str(args.attachment3_dir),
        "normalization_fit_split": "train",
        "text_input": "text_bert (int64, 3x50 token channels); precomputed `text` never used",
        "splits": {}, "files": {},
    }
    raw = D.load_aligned(args.aligned)
    for name in ("train", "valid", "test"):
        payload = split = D.split_payload(raw[name], name, with_labels=True)
        add_interface(split)
        counts = {str(k): int((payload["classification_labels"] == k).sum()) for k in range(3)}
        if name == "test":
            manifest["splits"][name] = {
                "count": int(len(payload["id"])),
                "scope": "labels stored for the post-freeze descriptive audit only; "
                         "never read by train.py or run_experiments.py",
                "class_counts": counts,
            }
        else:
            manifest["splits"][name] = {
                "count": int(len(payload["id"])),
                "video_groups": int(len(set(payload["video_id"]))),
                "class_counts": counts,
                "semantic_length_min": int(split["domain"].sum(1).min()),
                "semantic_length_max": int(split["domain"].sum(1).max()),
                "native_empty_modalities": (~split["observed"].any(-1)).sum(0).tolist(),
                "label_dtype": "int8 classification / float32 regression (not float16)",
            }
        path = data_dir / f"{name}.npz"
        manifest["files"][f"{name}.npz"] = D.save_npz(path, payload)
    a3 = D.load_attachment3(args.attachment3_dir)
    domain, observed = interface_masks(a3["text_bert"], a3["audio"], a3["vision"])
    a3["domain"], a3["observed"] = domain, observed
    a3["split"] = np.asarray("attachment3")
    manifest["files"]["attachment3.npz"] = D.save_npz(data_dir / "attachment3.npz", a3)
    manifest["attachment3"] = {
        "count": int(len(a3["id"])), "ids": [str(x) for x in a3["id"]],
        "has_precomputed_text": False,
        "role": "frozen inference only; never used for training, selection or threshold tuning",
    }
    train_raw = D.load_npz(data_dir / "train.npz")
    stats = D.fit_normalizer(train_raw["audio"], train_raw["vision"], train_raw["observed"])
    provenance = {
        "fit_split": "train", "rows": int(train_raw["observed"][:, 1].sum()),
        "rule": "per-dimension mean/std over O=1 rows, float64 accumulation, std floor 1e-5",
        "source_sha256": manifest["aligned_sha256"],
    }
    manifest["files"]["normalizer.npz"] = D.save_normalizer(data_dir / "normalizer.npz", stats, provenance)
    manifest["normalizer_provenance"] = provenance
    ids = {name: set(map(str, D.load_npz(data_dir / f"{name}.npz")["id"])) for name in ("train", "valid", "test")}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"id overlap between {a} and {b}")
    manifest["checks"] = ["no_id_overlap_across_splits", "normalizer_train_only",
                          "domain_and_observed_stored_per_split"]
    (run_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "splits": {k: v["count"] for k, v in manifest["splits"].items()},
                      "attachment3": manifest["attachment3"]["count"],
                      "data_manifest": str(run_dir / "data_manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())