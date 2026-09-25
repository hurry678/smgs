#!/usr/bin/env python3
"""T03/T04: post-freeze evaluation of the confirm / position / coupling / test banks.

The frozen manifest is read first; nothing here can change the selection. Every
condition keeps per-sample predictions (published class, raw and projected
intensity, mask SHA) so the independent paired video-cluster bootstrap in
``evaluate_predictions.py`` can re-derive the reported numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                       # noqa: E402
import engine as E                     # noqa: E402
import evaluate_predictions as EP      # noqa: E402
from metrics import metric_block, softmax   # noqa: E402
from models import build_model         # noqa: E402
from protocol_core import publish_c2   # noqa: E402

LABELS = ("Negative", "Neutral", "Positive")
METRIC_KEYS = ("accuracy", "macro_f1", "mae", "pearson", "raw_mae", "raw_pearson",
               "worst_condition_mae", "worst_condition_macro_f1", "actual_missing_ratio",
               "samples_with_new_masking", "raw_sign_conflict_rate",
               "mean_projection_magnitude", "argmax_agreement_with_published")
PREDICTION_COLUMNS = ("candidate", "train_seed", "bank", "condition_id", "replica", "sample_id",
                      "video_id", "y_class", "y_reg", "polarity", "intensity", "mask_sha256")


# --------------------------------------------------------------------------
# frozen model loading
# --------------------------------------------------------------------------

def build_model_for(run_dir, architecture, config, device, checkpoint=None):
    """Rebuild one candidate exactly as frozen; B0 is reconstructed from train priors."""
    if architecture == "training_prior":
        train = D.load_npz(run_dir / "data" / "train.npz")
        cls = train["classification_labels"].astype(int)
        reg = train["regression_labels"].astype(float)
        counts = np.asarray([(cls == k).sum() for k in range(3)], dtype=np.float64)
        means = np.asarray([reg[cls == k].mean() if (cls == k).any() else 0.0 for k in range(3)],
                           dtype=np.float64)
        model = build_model(architecture, class_counts=counts, class_intensity_mean=means)
    else:
        model = build_model(architecture, **config.get("model_kwargs", {}))
        if checkpoint is None:
            raise ValueError("neural candidates require a checkpoint")
        E.load_checkpoint(run_dir / checkpoint, model)
    model.to(device).eval()
    return model


def load_frozen(run_dir, manifest, device):
    config = dict(manifest["config"])
    model = build_model_for(run_dir, manifest["architecture"], config, device,
                            checkpoint=manifest["checkpoint"]["path"])
    return model, config


def load_baseline(run_dir, run_id, device):
    record = json.loads((run_dir / "runs" / run_id / "metrics.json").read_text())
    checkpoint = f"runs/{run_id}/checkpoint.pt"
    model = build_model_for(run_dir, record["architecture"], record["config"], device,
                            checkpoint=checkpoint)
    return model, record


# --------------------------------------------------------------------------
# split tensors with the frozen normalizer
# --------------------------------------------------------------------------

def load_split(run_dir, split, device):
    raw = D.load_npz(run_dir / "data" / f"{split}.npz")
    normalizer = D.load_normalizer(run_dir / "data" / "normalizer.npz")
    tensors = {
        "text_bert": torch.from_numpy(raw["text_bert"].astype(np.int64)).to(device),
        "audio": torch.from_numpy(D.apply_normalizer(
            raw["audio"], raw["observed"][:, 1],
            normalizer["audio"]["mean"], normalizer["audio"]["std"])).to(device),
        "vision": torch.from_numpy(D.apply_normalizer(
            raw["vision"], raw["observed"][:, 2],
            normalizer["vision"]["mean"], normalizer["vision"]["std"])).to(device),
        "domain": torch.from_numpy(raw["domain"].astype(bool)).to(device),
        "observed": torch.from_numpy(raw["observed"].astype(bool)).to(device),
    }
    meta = {
        "id": [str(x) for x in raw["id"]],
        "video_id": [str(x) for x in raw["video_id"]],
        "classification_labels": raw["classification_labels"].astype(np.int64),
        "regression_labels": raw["regression_labels"].astype(np.float32),
        "domain": raw["domain"].astype(bool),
        "observed": raw["observed"].astype(bool),
    }
    return tensors, meta


def bank_for(run_dir, bank, meta):
    """Read a stored bank, or regenerate the select protocol on the test split."""
    path = run_dir / "masks" / f"{bank}.npz"
    if path.exists():
        stored = D.load_bank(path)
        if stored["N"] != len(meta["id"]):
            raise SystemExit(f"{path} covers {stored['N']} samples, split has {len(meta['id'])}")
        return stored
    if bank != "test":
        raise SystemExit(f"missing mask bank: {path}")
    cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
    rebuilt = D.build_bank("select", meta["domain"], meta["observed"], meta["id"], cfg)
    rebuilt["bank"] = "test"
    # build_bank serialises specs to a JSON string (mirroring save_bank/load_bank);
    # normalise it here so downstream consumers always see list[dict].
    if isinstance(rebuilt.get("specs"), str):
        rebuilt["specs"] = json.loads(rebuilt["specs"])
    return rebuilt


# --------------------------------------------------------------------------
# condition evaluation
# --------------------------------------------------------------------------

def evaluate_bank(model, tensors, meta, bank, device, epsilon, batch_size=256, limit=None):
    n = len(meta["id"]) if limit is None else int(limit)
    sub = {k: v[:n] for k, v in tensors.items()}
    labels_class = meta["classification_labels"][:n]
    labels_reg = meta["regression_labels"][:n]
    observed = meta["observed"][:n]
    domain = meta["domain"][:n]
    model.eval()
    per_condition, per_sample = {}, {}
    with torch.no_grad():
        for ci, cid in enumerate(bank["condition_ids"]):
            vis = D.visibility_of(bank, ci)[:n]
            logits, raw = E.forward_condition(model, sub, torch.from_numpy(vis).to(device),
                                              batch_size=batch_size)
            prob = softmax(logits)
            pub_class, pub_intensity = publish_c2(prob, raw, epsilon=epsilon)
            block = metric_block(labels_class, labels_reg, logits, raw, pub_class, pub_intensity)
            block["actual_missing_ratio"] = float(
                np.mean(1 - vis.sum(-1).sum(1) / (3 * domain.sum(1))))
            block["samples_with_new_masking"] = int((vis.sum(-1) < observed.sum(-1)).any(1).sum())
            per_condition[cid] = block
            per_sample[cid] = {
                "logits": logits, "raw": raw, "published_class": pub_class,
                "published_intensity": pub_intensity, "probabilities": prob,
                "mask_sha256": [D.sample_mask_sha(vis[j]) for j in range(n)],
            }
    return per_condition, per_sample


def spec_index(bank):
    out = {}
    for cid, spec in zip(bank["condition_ids"], bank["specs"]):
        out[cid] = spec
    return out


def write_prediction_rows(path, candidate, seed, bank_name, bank, per_sample, meta, limit=None):
    n = len(meta["id"]) if limit is None else int(limit)
    specs = spec_index(bank)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PREDICTION_COLUMNS)
        for cid in bank["condition_ids"]:
            spec = specs[cid]
            block = per_sample[cid]
            for j in range(n):
                writer.writerow([
                    candidate, seed, bank_name, cid, int(spec.get("replica", 0)),
                    meta["id"][j], meta["video_id"][j], int(meta["classification_labels"][j]),
                    f"{float(meta['regression_labels'][j]):.6f}",
                    LABELS[int(block["published_class"][j])],
                    f"{float(block['published_intensity'][j]):.6f}",
                    block["mask_sha256"][j],
                ])


def save_npz(path, bank, per_sample):
    np.savez_compressed(
        path,
        condition_ids=np.asarray(bank["condition_ids"]),
        condition_sha256=np.asarray(bank["condition_sha256"]),
        logits=np.stack([per_sample[c]["logits"] for c in bank["condition_ids"]]),
        raw=np.stack([per_sample[c]["raw"] for c in bank["condition_ids"]]),
        published_class=np.stack([per_sample[c]["published_class"] for c in bank["condition_ids"]]),
        published_intensity=np.stack([per_sample[c]["published_intensity"]
                                      for c in bank["condition_ids"]]),
        probabilities=np.stack([per_sample[c]["probabilities"] for c in bank["condition_ids"]]),
    )
    return D.sha256_file(path)


def write_metrics_csv(path, run_id, candidate, bank_name, per_condition):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "candidate", "bank", "condition_id"] + list(METRIC_KEYS) +
                        ["per_class_f1", "confusion"])
        for cid, block in per_condition.items():
            writer.writerow([run_id, candidate, bank_name, cid] +
                            [block.get(k) for k in METRIC_KEYS] +
                            [json.dumps(block.get("per_class_f1"), ensure_ascii=False),
                             json.dumps(block.get("confusion"))])


def failure_rows(bank, per_condition, per_sample, meta, limit=None, top=25):
    """Largest absolute errors inside the worst (subset, ratio) cell."""
    n = len(meta["id"]) if limit is None else int(limit)
    specs = spec_index(bank)
    worst_cid = max(per_condition, key=lambda c: per_condition[c]["mae"])
    block = per_sample[worst_cid]
    err = np.abs(meta["regression_labels"][:n] - block["published_intensity"])
    order = np.argsort(-err)[:top]
    rows = []
    for j in order:
        rows.append({
            "condition_id": worst_cid, "subset": list(specs[worst_cid]["subset"]),
            "ratio": specs[worst_cid]["ratio"], "sample_id": meta["id"][j],
            "video_id": meta["video_id"][j], "y_class": int(meta["classification_labels"][j]),
            "y_reg": float(meta["regression_labels"][j]),
            "published_class": int(block["published_class"][j]),
            "raw_intensity": float(block["raw"][j]),
            "published_intensity": float(block["published_intensity"][j]),
            "abs_error": float(err[j]), "mask_sha256": block["mask_sha256"][j],
        })
    return worst_cid, rows


# --------------------------------------------------------------------------
# figures (SVG, generated from the CSV metrics only)
# --------------------------------------------------------------------------

def make_figures(out_dir, bank, per_condition, meta, limit=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # ``bank`` is the bank dict (it carries condition_ids/specs); its display name lives
    # under the "bank" key. Accept a plain string too so the helper stays usable directly.
    bank_name = bank["bank"] if isinstance(bank, dict) else bank
    specs = spec_index(bank)
    made = []

    def save(fig, name):
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)
        made.append(name)

    for metric, label, fname in (("mae", "R_MAE (published intensity)", "mae_by_ratio.svg"),
                                 ("macro_f1", "macro-F1", "f1_by_ratio.svg")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for subset in D.SUBSETS:
            xs, ys = [], []
            for ratio in D.RATIOS:
                vals = [per_condition[cid][metric] for cid, spec in specs.items()
                        if tuple(spec["subset"]) == tuple(subset) and abs(spec["ratio"] - ratio) < 1e-9]
                vals = [v for v in vals if v is not None]
                if vals:
                    xs.append(ratio)
                    ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, marker="o", label="+".join(D.MODALITIES[m] for m in subset))
        ax.set_xlabel("requested missing ratio (share of the valid domain)")
        ax.set_ylabel(label)
        ax.set_title(f"{bank_name} bank: {label} vs missing ratio")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        save(fig, fname)

    fig, ax = plt.subplots(figsize=(6, 4))
    matrix = np.full((len(D.SUBSETS), len(D.RATIOS)), np.nan)
    for i, subset in enumerate(D.SUBSETS):
        for j, ratio in enumerate(D.RATIOS):
            vals = [per_condition[cid]["mae"] for cid, spec in specs.items()
                    if tuple(spec["subset"]) == tuple(subset) and abs(spec["ratio"] - ratio) < 1e-9]
            if vals:
                matrix[i, j] = float(np.mean(vals))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(D.RATIOS)), [f"{r:.1f}" for r in D.RATIOS])
    ax.set_yticks(range(len(D.SUBSETS)),
                  ["+".join(D.MODALITIES[m] for m in s) for s in D.SUBSETS], fontsize=7)
    ax.set_xlabel("requested missing ratio")
    ax.set_title(f"{bank_name}: missing type x ratio (R_MAE)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6,
                        color="white")
    fig.colorbar(im, ax=ax, label="R_MAE")
    save(fig, "type_ratio_heatmap.svg")

    placements = [p for p in D.POSITION_PLACEMENTS
                  if any(spec.get("placement") == p for spec in specs.values())]
    if placements:
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.8 / max(1, len(placements))
        for k, placement in enumerate(placements):
            xs, ys = [], []
            for ratio in D.RATIOS:
                vals = [per_condition[cid]["mae"] for cid, spec in specs.items()
                        if spec.get("placement") == placement and abs(spec["ratio"] - ratio) < 1e-9]
                if vals:
                    xs.append(ratio)
                    ys.append(float(np.mean(vals)))
            ax.bar(np.asarray(xs) + (k - (len(placements) - 1) / 2) * width, ys, width,
                   label=placement)
        ax.set_xlabel("requested missing ratio")
        ax.set_ylabel("R_MAE")
        ax.set_title(f"{bank_name}: missing position x ratio")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
        save(fig, "position_by_ratio.svg")

    worst_cid = max(per_condition, key=lambda c: per_condition[c]["mae"])
    cm = np.asarray(per_condition[worst_cid]["confusion"])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS, fontsize=8)
    ax.set_yticks(range(3), LABELS, fontsize=8)
    ax.set_xlabel("published")
    ax.set_ylabel("ground truth")
    ax.set_title(f"worst condition ({worst_cid})", fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    save(fig, "worst_confusion_matrix.svg")
    return made


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def evaluate_candidate(run_dir, out_dir, name, model, tensors, meta, bank, bank_name,
                       device, epsilon, limit, seed):
    started = time.time()
    per_condition, per_sample = evaluate_bank(model, tensors, meta, bank, device, epsilon,
                                              limit=limit)
    specs = spec_index(bank)
    summary = E.summarise_bank(per_condition, specs)
    pred_csv = out_dir / f"predictions_{name}.csv"
    write_prediction_rows(pred_csv, name, seed, bank_name, bank, per_sample, meta, limit=limit)
    npz_sha = save_npz(out_dir / f"predictions_{name}.npz", bank, per_sample)
    worst_cid, failures = failure_rows(bank, per_condition, per_sample, meta, limit=limit)
    (out_dir / f"failures_{name}.json").write_text(
        json.dumps({"worst_condition": worst_cid, "rows": failures}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {"per_condition": per_condition, "summary": summary, "csv": pred_csv,
            "npz_sha256": npz_sha, "seconds": time.time() - started,
            "worst_condition": worst_cid,
            "worst_condition_mae": per_condition[worst_cid]["mae"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--bank", required=True,
                    choices=("confirm", "position", "coupling", "test", "select", "clean"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compare", default=None,
                    help="run_id of a second frozen candidate (default: p0_B2_seed42)")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=None, help="smoke only: first N valid samples")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args()
    run_dir = args.freeze.resolve().parent
    manifest = json.loads(args.freeze.read_text())
    device = torch.device(args.device)
    split = "test" if args.bank == "test" else "valid"
    tensors, meta = load_split(run_dir, split, device)
    bank = bank_for(run_dir, args.bank, meta)
    out_dir = (args.output_root or (run_dir / "evaluation" / args.bank))
    out_dir.mkdir(parents=True, exist_ok=True)
    epsilon = manifest["publication"]["nonneutral_epsilon"]
    frozen_model, config = load_frozen(run_dir, manifest, device)
    frozen = evaluate_candidate(run_dir, out_dir, "frozen", frozen_model, tensors, meta, bank,
                                args.bank, device, epsilon, args.limit,
                                manifest["deployment_seed"])
    result = {
        "schema": "q2-post-freeze-evaluation-1", "bank": args.bank, "split": split,
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "freeze_manifest_sha256": D.sha256_file(args.freeze),
        "deployed_run_id": manifest["deployed_run_id"],
        "architecture": manifest["architecture"], "config": config,
        "samples": len(meta["id"]) if args.limit is None else int(args.limit),
        "conditions": len(bank["condition_ids"]), "bank_sha256": bank["bank_sha256"],
        "limit": args.limit,
        "note": ("smoke subset of the first N valid rows; never used for ranking"
                 if args.limit is not None else
                 "full split; test is a previously exposed descriptive audit"),
        "summary": frozen["summary"], "worst_condition": frozen["worst_condition"],
        "per_condition": frozen["per_condition"],
        "predictions": {"csv": frozen["csv"].name, "npz_sha256": frozen["npz_sha256"],
                        "seconds": round(frozen["seconds"], 2)},
    }
    write_metrics_csv(out_dir / "metrics.csv", manifest["deployed_run_id"], "frozen", args.bank,
                      frozen["per_condition"])
    compare_id = None if args.no_compare else (args.compare or "p0_B2_seed42")
    if compare_id and (run_dir / "runs" / compare_id / "checkpoint.pt").exists() and \
            compare_id != manifest["deployed_run_id"]:
        baseline_model, record = load_baseline(run_dir, compare_id, device)
        baseline = evaluate_candidate(run_dir, out_dir, compare_id, baseline_model, tensors, meta,
                                      bank, args.bank, device, epsilon, args.limit,
                                      record["seed"])
        result["reference"] = {"run_id": compare_id, "architecture": record["architecture"],
                               "config": record["config"], "summary": baseline["summary"],
                               "worst_condition": baseline["worst_condition"]}
        # independent re-derivation with the provided paired video-cluster bootstrap
        rows = []
        for path in (baseline["csv"], frozen["csv"]):
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows.extend(csv.DictReader(f))
        try:
            paired = EP.evaluate(rows, compare_id, "frozen", resamples=args.resamples,
                                 seed=20260925)
            paired["bank"] = args.bank
            paired["status"] = "completed"
        except Exception as exc:                                        # noqa: BLE001
            paired = {"schema": "q2-paired-video-bootstrap-1", "bank": args.bank,
                      "status": "not_run", "error": f"{type(exc).__name__}: {exc}",
                      "reference": compare_id, "candidate": "frozen",
                      "note": "independent recomputation refused the inputs; see error"}
        (out_dir / "comparison.json").write_text(
            json.dumps(paired, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        result["comparison"] = {"status": paired.get("status"), "path": "comparison.json",
                                "metrics": paired.get("metrics")}
    else:
        result["reference"] = None
    if args.limit is None:
        result["figures"] = make_figures(out_dir, bank, frozen["per_condition"], meta)
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"bank": args.bank, "summary": frozen["summary"],
                      "worst_condition": frozen["worst_condition"],
                      "reference": result["reference"]["run_id"] if result["reference"] else None,
                      "comparison_status": (result.get("comparison") or {}).get("status"),
                      "output": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
