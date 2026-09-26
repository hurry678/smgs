#!/usr/bin/env python3
"""Validation-only ensemble evaluation with fixed synthetic missing masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from q2_eval import choose_device, infer, load_checkpoint
from q2_eval_robustness import condition_obs, mask_sha256
from q2_train import MODALITIES, MODES, RATIOS, SUBSETS, apply_normalization, load_split, metrics, search_c3, subset_name


def ensemble_infer(models: List[Any], data: Dict[str, Any], obs: torch.Tensor, indices: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    probs: List[np.ndarray] = []
    raws: List[np.ndarray] = []
    for model in models:
        logits, raw = infer(model, data, obs, indices)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs.append(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))
        raws.append(raw)
    mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
    logits = np.log(np.clip(mean_prob, 1e-9, 1.0))
    return logits.astype(np.float32), np.mean(np.stack(raws, axis=0), axis=0).astype(np.float32)


def evaluate(models: List[Any], data: Dict[str, Any], mask_seed: int, replicas: int) -> Dict[str, Any]:
    n = len(data["ids"])
    indices = torch.arange(n, device=data["labels"].device)
    y = data["labels"].detach().cpu().numpy()
    r = data["regression"].detach().cpu().numpy()
    full_logits, full_raw = ensemble_infer(models, data, data["obs"], indices)
    dn, dp, score = search_c3(full_raw, y)
    thresholds = (dn, dp)
    conditions: List[Dict[str, Any]] = []
    for subset in SUBSETS:
        for ratio in RATIOS:
            reps_c2: List[Dict[str, float]] = []
            reps_c3: List[Dict[str, float]] = []
            hashes: List[str] = []
            for replica in range(replicas):
                obs = condition_obs(data, subset, ratio, replica, "random", mask_seed)
                hashes.append(mask_sha256(obs))
                logits, raw = ensemble_infer(models, data, obs, indices)
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
                logits, raw = ensemble_infer(models, data, obs, indices)
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
        "full_c2": metrics(full_logits, full_raw, y, r, "c2"),
        "full_c3": metrics(full_logits, full_raw, y, r, "c3", thresholds),
        "conditions": conditions,
        "positions": positions,
        "mask_seed": int(mask_seed),
        "replicas": int(replicas),
        "evaluation_protocol": {
            "split": "valid",
            "primary_strategy": "C2",
            "mask_seed_fixed_across_checkpoints": True,
            "ensemble_rule": "mean softmax probabilities and mean raw intensity",
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
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.replicas < 1:
        raise SystemExit("--replicas must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    models = []
    checkpoints = []
    norms = []
    for checkpoint_path in args.checkpoints:
        model, checkpoint, norm, _ = load_checkpoint(checkpoint_path, args.model_dir, device)
        models.append(model)
        checkpoints.append(checkpoint)
        norms.append(norm)
    reference = norms[0]
    for norm in norms[1:]:
        for key in reference:
            if not torch.allclose(reference[key], norm[key], atol=1e-6, rtol=1e-6):
                raise SystemExit(f"normalization mismatch in {key}")

    with np.load(args.data, allow_pickle=False) as archive:
        data = load_split(archive, "valid", device)
    apply_normalization(data, reference)

    result = evaluate(models, data, args.mask_seed, args.replicas)
    result.update({
        "mode": "valid",
        "split": "valid",
        "checkpoints": [str(x) for x in args.checkpoints],
        "kinds": [str(x["kind"]) for x in checkpoints],
        "seeds": [int(x["seed"]) for x in checkpoints],
        "evaluation_script": "scripts/q2_eval_ensemble.py",
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
        "kinds": result["kinds"],
        "seeds": result["seeds"],
        "mask_seed": result["mask_seed"],
        "replicas": result["replicas"],
        "R_MAE": float(np.mean([x["c2_mean_mae"] for x in result["conditions"]])),
        "full_c2": result["full_c2"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
