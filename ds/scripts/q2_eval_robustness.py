#!/usr/bin/env python3
"""Validation-only Q2 robustness evaluation with fixed missing masks.

This script extends the frozen Q2 evaluation without changing training,
the official test evaluation, or the attachment-3 submission.  Training seeds
control model weights only.  All checkpoints are evaluated with the same mask
seed, the same replicas, and therefore the same synthetic missing conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from q2_eval import choose_device, infer, load_checkpoint
from q2_train import (
    MODALITIES, MODES, RATIOS, SUBSETS, apply_normalization, load_split,
    make_missing, metrics, search_c3, subset_name,
)


def mask_sha256(mask: torch.Tensor) -> str:
    arr = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def condition_obs(
    data: Dict[str, Any], subset: Tuple[int, ...], ratio: float,
    replica: int, mode: str, mask_seed: int,
) -> torch.Tensor:
    return make_missing(
        data["obs"], data["ids"], data["maskable_obs"],
        subset=subset, ratio=ratio, replica=replica, mode=mode,
        seed=mask_seed,
    )


def evaluate(
    model: Any,
    data: Dict[str, Any],
    mask_seed: int,
    replicas: int,
) -> Dict[str, Any]:
    n = len(data["ids"])
    indices = torch.arange(n, device=data["labels"].device)
    y = data["labels"].detach().cpu().numpy()
    r = data["regression"].detach().cpu().numpy()

    full_logits, full_raw = infer(model, data, data["obs"], indices)
    dn, dp, score = search_c3(full_raw, y)
    thresholds = (dn, dp)
    full_c2 = metrics(full_logits, full_raw, y, r, "c2")
    full_c3 = metrics(full_logits, full_raw, y, r, "c3", thresholds)

    conditions: List[Dict[str, Any]] = []
    for subset in SUBSETS:
        for ratio in RATIOS:
            reps_c2: List[Dict[str, float]] = []
            reps_c3: List[Dict[str, float]] = []
            hashes: List[str] = []
            for replica in range(replicas):
                obs = condition_obs(data, subset, ratio, replica, "random", mask_seed)
                hashes.append(mask_sha256(obs))
                logits, raw = infer(model, data, obs, indices)
                reps_c2.append(metrics(logits, raw, y, r, "c2"))
                reps_c3.append(metrics(logits, raw, y, r, "c3", thresholds))
            conditions.append({
                "subset": subset_name(subset),
                "ratio": ratio,
                "replicas": replicas,
                "mask_sha256": hashes,
                "c2_replicas": reps_c2,
                "c3_replicas": reps_c3,
                "c2_mean_mae": float(np.mean([x["mae"] for x in reps_c2])),
                "c3_mean_mae": float(np.mean([x["mae"] for x in reps_c3])),
                "c2_mean_macro_f1": float(np.mean([x["macro_f1"] for x in reps_c2])),
                "c3_mean_macro_f1": float(np.mean([x["macro_f1"] for x in reps_c3])),
            })

    positions: List[Dict[str, Any]] = []
    for modality_index in range(3):
        for ratio in RATIOS:
            for mode in MODES:
                obs = condition_obs(data, (modality_index,), ratio, 0, mode, mask_seed)
                logits, raw = infer(model, data, obs, indices)
                positions.append({
                    "subset": MODALITIES[modality_index],
                    "ratio": ratio,
                    "position": mode,
                    "mask_sha256": mask_sha256(obs),
                    "c2": metrics(logits, raw, y, r, "c2"),
                    "c3": metrics(logits, raw, y, r, "c3", thresholds),
                })

    return {
        "n": int(n),
        "thresholds": list(thresholds),
        "c3_search": {"dn": dn, "dp": dp, "score": score},
        "full_c2": full_c2,
        "full_c3": full_c3,
        "conditions": conditions,
        "positions": positions,
        "mask_seed": int(mask_seed),
        "replicas": int(replicas),
        "evaluation_protocol": {
            "split": "valid",
            "primary_strategy": "C2",
            "mask_seed_fixed_across_checkpoints": True,
            "random_condition_count": len(conditions),
            "position_condition_count": len(positions),
            "mask_hash": "sha256 of boolean [N,3,50] visibility tensor",
        },
        "full_logits": full_logits.tolist(),
        "full_raw": full_raw.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.replicas < 1:
        raise SystemExit("--replicas must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, checkpoint, norm, _ = load_checkpoint(args.checkpoint, args.model_dir, device)
    with np.load(args.data, allow_pickle=False) as archive:
        data = load_split(archive, args.split, device)
    apply_normalization(data, norm)

    result = evaluate(model, data, args.mask_seed, args.replicas)
    result.update({
        "mode": "valid",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "kind": str(checkpoint["kind"]),
        "seed": int(args.train_seed),
        "train_seed": int(args.train_seed),
        "mask_seed": int(args.mask_seed),
        "evaluation_script": "scripts/q2_eval_robustness.py",
    })
    output = args.out_dir / "valid_valid_evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "valid_valid_full.npz",
        logits=np.asarray(result["full_logits"], dtype=np.float32),
        raw=np.asarray(result["full_raw"], dtype=np.float32),
    )
    print(json.dumps({
        "done": True,
        "kind": result["kind"],
        "train_seed": result["train_seed"],
        "mask_seed": result["mask_seed"],
        "replicas": result["replicas"],
        "R_MAE": float(np.mean([x["c2_mean_mae"] for x in result["conditions"]])),
        "full_c2": result["full_c2"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
