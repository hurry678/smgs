#!/usr/bin/env python3
"""T04: inference-only prediction for the 30 Attachment-3 samples under a frozen manifest."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                       # noqa: E402
import engine as E                     # noqa: E402
from metrics import softmax            # noqa: E402
from protocol_core import publish_c2   # noqa: E402


def load_frozen(run_dir, manifest, device):
    config = {"model_kwargs": {"hidden": 64, "dropout": manifest["config"].get("dropout", 0.15)}}
    config.update(manifest["config"])
    if manifest["architecture"] == "training_prior":
        # B0 carries no checkpoint: it is the train-split class prior plus class-conditional
        # mean intensity, rebuilt exactly as evaluate.build_model_for does
        from models import build_model
        train = D.load_npz(run_dir / "data" / "train.npz")
        cls = train["classification_labels"].astype(int)
        reg = train["regression_labels"].astype(float)
        counts = np.asarray([(cls == k).sum() for k in range(3)], dtype=np.float64)
        means = np.asarray([reg[cls == k].mean() if (cls == k).any() else 0.0 for k in range(3)],
                           dtype=np.float64)
        model = build_model(manifest["architecture"], class_counts=counts,
                            class_intensity_mean=means)
    else:
        model = E.make_model(manifest["architecture"], None, **config.get("model_kwargs", {}))
        E.load_checkpoint(run_dir / manifest["checkpoint"]["path"], model)
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run_dir = args.freeze.resolve().parent
    manifest = json.loads(args.freeze.read_text())
    device = torch.device(args.device)
    model = load_frozen(run_dir, manifest, device)
    a3 = D.load_npz(args.input)
    normalizer = D.load_normalizer(run_dir / "data" / "normalizer.npz")
    audio = D.apply_normalizer(a3["audio"], a3["observed"][:, 1],
                               normalizer["audio"]["mean"], normalizer["audio"]["std"])
    vision = D.apply_normalizer(a3["vision"], a3["observed"][:, 2],
                                normalizer["vision"]["mean"], normalizer["vision"]["std"])
    tensors = {
        "text_bert": torch.from_numpy(a3["text_bert"].astype(np.int64)).to(device),
        "audio": torch.from_numpy(audio).to(device),
        "vision": torch.from_numpy(vision).to(device),
        "domain": torch.from_numpy(a3["domain"].astype(bool)).to(device),
        "observed": torch.from_numpy(a3["observed"].astype(bool)).to(device),
    }
    with torch.no_grad():
        logits, raw = E.forward_condition(model, tensors, tensors["observed"], batch_size=32)
    prob = softmax(logits)
    pub_class, pub_intensity = publish_c2(prob, raw,
                                          epsilon=manifest["publication"]["nonneutral_epsilon"])
    labels = ("Negative", "Neutral", "Positive")
    ids = [str(x) for x in a3["id"]]
    files = [str(x) for x in a3["source_file"]]
    out_dir = run_dir / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "q2_predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(manifest["publication"]["csv_columns"])
        for i, sid in enumerate(ids):
            writer.writerow([sid, files[i], labels[int(pub_class[i])], f"{pub_intensity[i]:.6f}"])
    json_path = out_dir / "q2_predictions.json"
    payload = {
        "schema": "q2-attachment3-predictions-1",
        "freeze_manifest_sha256": D.sha256_file(args.freeze),
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "input_sha256": D.sha256_file(args.input),
        "publication_policy": manifest["publication"]["policy"],
        "rows": [{"sample_id": sid, "原文件名": files[i], "polarity": labels[int(pub_class[i])],
                  "intensity": float(pub_intensity[i]), "raw_intensity": float(raw[i]),
                  "probabilities": [float(x) for x in prob[i]]}
                 for i, sid in enumerate(ids)],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = {
        "schema": "q2-submission-validation-1",
        "rows_csv": sum(1 for _ in csv_path.open(encoding="utf-8-sig")) - 1,
        "rows_json": len(payload["rows"]),
        "ids_in_order": ids == [f"附件3_{i:02d}" for i in range(1, 31)],
        "ids": ids,
        "filenames_match": all(files[i] == f"附件3_{i + 1:02d}.pkl" for i in range(30)),
        "csv_json_consistent": True,
        "sign_rule_ok": bool(np.all((pub_class == 1) == (np.abs(pub_intensity) < 1e-9))),
        "finite": bool(np.isfinite(pub_intensity).all() and np.isfinite(prob).all()),
        "intensity_in_range": bool((np.abs(pub_intensity) <= 3).all()),
        "csv_sha256": D.sha256_file(csv_path), "json_sha256": D.sha256_file(json_path),
        "decimals": manifest["publication"]["csv_decimals"],
    }
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    for row, item in zip(rows, payload["rows"]):
        if row["sample_id"] != item["sample_id"] or row["polarity"] != item["polarity"]:
            validation["csv_json_consistent"] = False
        if abs(float(row["intensity"]) - item["intensity"]) > 5e-7:
            validation["csv_json_consistent"] = False
        if abs(float(row["intensity"])) > 1e-9:
            if (row["polarity"] == "Positive") != (float(row["intensity"]) > 0):
                validation["sign_rule_ok"] = False
    (out_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({k: validation[k] for k in ("rows_csv", "rows_json", "ids_in_order",
                                                 "filenames_match", "csv_json_consistent",
                                                 "sign_rule_ok", "finite", "intensity_in_range")},
                                                 ensure_ascii=False))
    return 0 if all([validation["ids_in_order"], validation["filenames_match"],
                     validation["csv_json_consistent"], validation["sign_rule_ok"],
                     validation["finite"], validation["intensity_in_range"]]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
