#!/usr/bin/env python3
"""Evaluate a frozen Q2 checkpoint or predict attachment 3.

This script is deliberately separate from training.  It can evaluate the
official test split once after model selection, compare C2/C3 publishing
strategies on validation, and generate the attachment-3 submission CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from q2_models import Q2Model
from q2_train import (
    MODALITIES, MODES, RATIOS, SUBSETS, apply_normalization, load_split,
    make_missing, metrics, predict_model, publish_c2, publish_c3, search_c3,
    subset_name,
)

POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}


def choose_device(name: str) -> torch.device:
    return torch.device(name if (name == "cpu" or torch.cuda.is_available()) else "cpu")


def load_checkpoint(path: Path, model_dir: Path, device: torch.device) -> Tuple[Q2Model, Dict[str, Any], Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(str(path), map_location="cpu")
    kind = str(ckpt["kind"])
    cfg = dict(ckpt.get("model_config", {}))
    hidden = int(cfg.get("hidden", 128))
    dropout = float(cfg.get("dropout", 0.15))
    model = Q2Model(kind, model_dir, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in ckpt.get("norm", {}).items()}
    return model, ckpt, norm, cfg


def infer(model: Q2Model, data: Dict[str, Any], obs: torch.Tensor, indices: torch.Tensor | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if indices is None:
        indices = torch.arange(data["labels"].numel(), device=data["labels"].device)
    model.eval()
    with torch.no_grad():
        out = model(data["text_bert"][indices, 0], obs, data["audio"][indices], data["vision"][indices])
    return out["logits"].detach().float().cpu().numpy(), out["raw_intensity"].detach().float().cpu().numpy()


def condition_obs(data: Dict[str, Any], subset: Tuple[int, ...], ratio: float, replica: int, mode: str, seed: int) -> torch.Tensor:
    return make_missing(
        data["obs"], data["ids"], data["maskable_obs"],
        subset=subset, ratio=ratio, replica=replica, mode=mode, seed=seed,
    )


def eval_split(model: Q2Model, data: Dict[str, Any], seed: int, thresholds: Tuple[float, float] | None) -> Dict[str, Any]:
    n = len(data["ids"])
    indices = torch.arange(n, device=data["labels"].device)
    full_obs = data["obs"]
    full_logits, full_raw = infer(model, data, full_obs, indices)
    y = data["labels"].detach().cpu().numpy()
    r = data["regression"].detach().cpu().numpy()
    if thresholds is None:
        dn, dp, score = search_c3(full_raw, y)
        thresholds = (dn, dp)
        c3_search = {"dn": dn, "dp": dp, "score": score}
    else:
        c3_search = None
    full_c2 = metrics(full_logits, full_raw, y, r, "c2")
    full_c3 = metrics(full_logits, full_raw, y, r, "c3", thresholds)
    conditions: List[Dict[str, Any]] = []
    for subset in SUBSETS:
        for ratio in RATIOS:
            reps_c2 = []
            reps_c3 = []
            for replica in range(3):
                obs = condition_obs(data, subset, ratio, replica, "random", seed)
                logits, raw = infer(model, data, obs, indices)
                reps_c2.append(metrics(logits, raw, y, r, "c2"))
                reps_c3.append(metrics(logits, raw, y, r, "c3", thresholds))
            conditions.append({
                "subset": subset_name(subset),
                "ratio": ratio,
                "c2_replicas": reps_c2,
                "c3_replicas": reps_c3,
                "c2_mean_mae": float(np.mean([x["mae"] for x in reps_c2])),
                "c3_mean_mae": float(np.mean([x["mae"] for x in reps_c3])),
                "c2_mean_macro_f1": float(np.mean([x["macro_f1"] for x in reps_c2])),
                "c3_mean_macro_f1": float(np.mean([x["macro_f1"] for x in reps_c3])),
            })
    positions: List[Dict[str, Any]] = []
    for m in range(3):
        for ratio in RATIOS:
            for mode in MODES:
                obs = condition_obs(data, (m,), ratio, 0, mode, seed)
                logits, raw = infer(model, data, obs, indices)
                positions.append({
                    "subset": MODALITIES[m], "ratio": ratio, "position": mode,
                    "c2": metrics(logits, raw, y, r, "c2"),
                    "c3": metrics(logits, raw, y, r, "c3", thresholds),
                })
    return {
        "n": int(n), "thresholds": list(thresholds), "c3_search": c3_search,
        "full_c2": full_c2, "full_c3": full_c3,
        "conditions": conditions,
        "positions": positions,
        "full_logits": full_logits.tolist(),
        "full_raw": full_raw.tolist(),
    }


def attachment3_predict(model: Q2Model, path: Path, norm: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    text_bert = torch.from_numpy(np.asarray(z["text_bert"], dtype=np.int64)).to(device)
    audio = torch.from_numpy(np.asarray(z["audio"], dtype=np.float32)).to(device)
    vision = torch.from_numpy(np.asarray(z["vision"], dtype=np.float32)).to(device)
    text_obs = text_bert[:, 1] > 0
    audio_obs = audio.abs().amax(dim=-1) > 1e-8
    vision_obs = vision.abs().amax(dim=-1) > 1e-8
    obs = torch.stack([text_obs, audio_obs, vision_obs], dim=1)
    # Only original observations are present in attachment 3; apply training
    # statistics and preserve the missing mask exactly.
    audio = (audio - norm["audio_mean"]) / norm["audio_std"]
    vision = (vision - norm["vision_mean"]) / norm["vision_std"]
    audio = torch.where(audio_obs.unsqueeze(-1), audio, torch.zeros_like(audio))
    vision = torch.where(vision_obs.unsqueeze(-1), vision, torch.zeros_like(vision))
    with torch.no_grad():
        out = model(text_bert[:, 0], obs, audio, vision)
    logits = out["logits"].detach().float().cpu().numpy()
    raw = out["raw_intensity"].detach().float().cpu().numpy()
    return {"filenames": [str(x) for x in z["filenames"].tolist()], "logits": logits, "raw": raw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test", "attachment3"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--publish-strategy", choices=["c2", "c3"], default="c2")
    ap.add_argument("--thresholds", type=float, nargs=2, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, ckpt, norm, cfg = load_checkpoint(args.checkpoint, args.model_dir, device)
    if args.mode == "attachment3":
        pred = attachment3_predict(model, args.data, norm, device)
        raw = pred["raw"]
        logits = pred["logits"]
        if args.publish_strategy == "c2":
            cls, intensity = publish_c2(logits, raw)
            publish_thresholds = None
        else:
            if args.thresholds is None:
                raise SystemExit("C3 attachment3 mode requires --thresholds frozen on validation")
            cls, intensity = publish_c3(raw, args.thresholds[0], args.thresholds[1])
            publish_thresholds = list(args.thresholds)
        rows = []
        for i, name in enumerate(pred["filenames"]):
            rows.append({
                "sample_id": name.rsplit(".", 1)[0],
                "原文件名": name,
                "polarity": POLARITY[int(cls[i])],
                "intensity": float(intensity[i]),
                "logits": logits[i].tolist(),
                "raw_intensity": float(raw[i]),
            })
        (args.out_dir / "attachment3_predictions.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "publish_strategy": args.publish_strategy.upper(), "thresholds": publish_thresholds, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        # CSV with a stable, documented column order.  Intensity is kept at
        # six decimal places, so a non-zero sign is never rounded to zero.
        import csv
        with (args.out_dir / "attachment3_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "原文件名", "polarity", "intensity"])
            w.writeheader()
            for row in rows:
                w.writerow({k: (f"{row[k]:.6f}" if k == "intensity" else row[k]) for k in w.fieldnames})
        print(json.dumps({"done": True, "n": len(rows), "out_dir": str(args.out_dir)}, ensure_ascii=False), flush=True)
        return

    z = np.load(args.data, allow_pickle=False)
    data = load_split(z, args.split, device)
    # Checkpoint normalization is authoritative; it was estimated on train only.
    apply_normalization(data, norm)
    thresholds = tuple(args.thresholds) if args.thresholds is not None else None
    result = eval_split(model, data, args.seed, thresholds)
    result.update({"mode": args.mode, "split": args.split, "checkpoint": str(args.checkpoint), "kind": ckpt["kind"], "seed": args.seed})
    (args.out_dir / f"{args.mode}_{args.split}_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / f"{args.mode}_{args.split}_full.npz", logits=result["full_logits"], raw=result["full_raw"])
    print(json.dumps({"done": True, "mode": args.mode, "split": args.split, "thresholds": result["thresholds"], "full_c2": result["full_c2"], "full_c3": result["full_c3"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
