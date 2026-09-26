#!/usr/bin/env python3
"""Frozen M3 ensemble inference for official test and attachment 3.

This script never participates in model selection.  It is intended to be run
once after the validation-only freeze of an ensemble version.  All outputs are
written to the caller-provided directory; no existing submission is replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from q2_eval import POLARITY, choose_device, infer, load_checkpoint
from q2_train import apply_normalization, load_split, metrics, publish_c2, publish_c3


ENSEMBLE_RULE = "mean softmax probabilities; mean raw intensity"
C3_REFERENCE_THRESHOLDS = (-0.20, 0.35)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(paths: List[Path], model_dir: Path, device: torch.device):
    models, checkpoints, norms = [], [], []
    for path in paths:
        model, checkpoint, norm, _ = load_checkpoint(path, model_dir, device)
        models.append(model)
        checkpoints.append(checkpoint)
        norms.append(norm)
    reference = norms[0]
    for norm in norms[1:]:
        for key in reference:
            if not torch.allclose(reference[key], norm[key], atol=1e-6, rtol=1e-6):
                raise SystemExit(f"normalization mismatch in {key}")
    kinds = {str(item["kind"]) for item in checkpoints}
    if len(kinds) != 1:
        raise SystemExit(f"checkpoint kind mismatch: {sorted(kinds)}")
    seeds = [int(item["seed"]) for item in checkpoints]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds: {seeds}")
    return models, checkpoints, reference, kinds.pop(), seeds


def ensemble_infer_with_members(
    models: List[Any],
    data: Dict[str, Any],
    obs: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    member_probs: List[np.ndarray] = []
    member_raws: List[np.ndarray] = []
    for model in models:
        logits, raw = infer(model, data, obs, indices)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        member_probs.append(probs.astype(np.float32))
        member_raws.append(raw.astype(np.float32))
    prob_stack = np.stack(member_probs, axis=0)
    raw_stack = np.stack(member_raws, axis=0)
    mean_prob = prob_stack.mean(axis=0)
    logits = np.log(np.clip(mean_prob, 1e-9, 1.0)).astype(np.float32)
    raw = raw_stack.mean(axis=0).astype(np.float32)
    return logits, raw, prob_stack, raw_stack


def common_metadata(checkpoint_paths: List[Path], checkpoints: List[Dict[str, Any]], kind: str, seeds: List[int]) -> Dict[str, Any]:
    return {
        "model_version": "M3-ensemble9",
        "kind": kind,
        "seeds": seeds,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_sha256": [sha256_file(path) for path in checkpoint_paths],
        "ensemble_rule": ENSEMBLE_RULE,
        "publish_strategy": "C2",
        "selection_split": "valid_only",
        "selection_metric": "R_MAE over 7 non-empty subsets x 3 missing ratios, 5 fixed replicas",
        "official_test_evaluated_once_after_freeze": True,
    }


def run_test(models, data, checkpoint_paths, checkpoints, kind, seeds, out_dir: Path) -> None:
    indices = torch.arange(len(data["ids"]), device=data["labels"].device)
    logits, raw, member_probs, member_raws = ensemble_infer_with_members(
        models, data, data["obs"], indices
    )
    y = data["labels"].detach().cpu().numpy()
    r = data["regression"].detach().cpu().numpy()
    result = {
        "mode": "test",
        "split": "test",
        "n": int(len(y)),
        "ids": [str(x) for x in data["ids"]],
        "full_c2": metrics(logits, raw, y, r, "c2"),
        "full_c3_reference": metrics(logits, raw, y, r, "c3", C3_REFERENCE_THRESHOLDS),
        "c3_reference_thresholds": list(C3_REFERENCE_THRESHOLDS),
        "member_probabilities": member_probs.tolist(),
        "member_raw_intensity": member_raws.tolist(),
        "ensemble_logits": logits.tolist(),
        "ensemble_raw_intensity": raw.tolist(),
        "protocol_note": "Official test evaluated once after validation-only freeze; no test-based tuning.",
    }
    result.update(common_metadata(checkpoint_paths, checkpoints, kind, seeds))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_test_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "test_test_full.npz",
        logits=logits,
        raw=raw,
        member_probs=member_probs,
        member_raws=member_raws,
    )
    print(json.dumps({
        "done": True,
        "mode": "test",
        "n": int(len(y)),
        "full_c2": result["full_c2"],
        "full_c3_reference": result["full_c3_reference"],
        "out_dir": str(out_dir),
    }, ensure_ascii=False), flush=True)


def load_attachment3(path: Path, norm: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        text_bert = torch.from_numpy(np.asarray(archive["text_bert"], dtype=np.int64)).to(device)
        audio = torch.from_numpy(np.asarray(archive["audio"], dtype=np.float32)).to(device)
        vision = torch.from_numpy(np.asarray(archive["vision"], dtype=np.float32)).to(device)
        filenames = [str(x) for x in archive["filenames"].tolist()]
    text_obs = text_bert[:, 1] > 0
    audio_obs = audio.abs().amax(dim=-1) > 1e-8
    vision_obs = vision.abs().amax(dim=-1) > 1e-8
    obs = torch.stack([text_obs, audio_obs, vision_obs], dim=1)
    audio = (audio - norm["audio_mean"]) / norm["audio_std"]
    vision = (vision - norm["vision_mean"]) / norm["vision_std"]
    audio = torch.where(audio_obs.unsqueeze(-1), audio, torch.zeros_like(audio))
    vision = torch.where(vision_obs.unsqueeze(-1), vision, torch.zeros_like(vision))
    return {
        "filenames": filenames,
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "obs": obs,
    }


def run_attachment3(models, data, norm, checkpoint_paths, checkpoints, kind, seeds, out_dir: Path) -> None:
    attachment = load_attachment3(data, norm, next(models[0].parameters()).device)
    indices = torch.arange(len(attachment["filenames"]), device=next(models[0].parameters()).device)
    logits, raw, member_probs, member_raws = ensemble_infer_with_members(
        models, attachment, attachment["obs"], indices
    )
    classes, intensity = publish_c2(logits, raw)
    rows = []
    for i, filename in enumerate(attachment["filenames"]):
        rows.append({
            "sample_id": filename.rsplit(".", 1)[0],
            "原文件名": filename,
            "polarity": POLARITY[int(classes[i])],
            "intensity": float(intensity[i]),
            "ensemble_logits": logits[i].tolist(),
            "ensemble_raw_intensity": float(raw[i]),
            "member_probabilities": member_probs[:, i, :].tolist(),
            "member_raw_intensity": member_raws[:, i].tolist(),
        })
    result = {
        "mode": "attachment3",
        "n": len(rows),
        "c3_reference_thresholds": list(C3_REFERENCE_THRESHOLDS),
        "rows": rows,
        "protocol_note": "Attachment 3 used only for frozen inference; no tuning or selection.",
    }
    result.update(common_metadata(checkpoint_paths, checkpoints, kind, seeds))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attachment3_predictions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "attachment3_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "原文件名", "polarity", "intensity"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "原文件名": row["原文件名"],
                "polarity": row["polarity"],
                "intensity": f"{row['intensity']:.6f}",
            })
    print(json.dumps({
        "done": True,
        "mode": "attachment3",
        "n": len(rows),
        "polarity_counts": {label: sum(row["polarity"] == label for row in rows) for label in POLARITY.values()},
        "out_dir": str(out_dir),
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "attachment3"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = choose_device(args.device)
    models, checkpoints, norm, kind, seeds = load_models(args.checkpoints, args.model_dir, device)
    if args.mode == "attachment3":
        run_attachment3(models, args.data, norm, args.checkpoints, checkpoints, kind, seeds, args.out_dir)
        return
    with np.load(args.data, allow_pickle=False) as archive:
        data = load_split(archive, args.split, device)
    apply_normalization(data, norm)
    run_test(models, data, args.checkpoints, checkpoints, kind, seeds, args.out_dir)


if __name__ == "__main__":
    main()
