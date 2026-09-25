#!/usr/bin/env python3
"""Q3 faithful explanation core for the frozen M3-ensemble9 predictor.

The predictor is loaded from the frozen Q2 checkpoints.  Explanations are
computed on the validation split first; attachment 4 is opened only by the
final frozen inference mode.  The implementation uses exact three-player
Shapley values for modality contributions and continuous-window occlusion for
local evidence.  It never substitutes attention maps for either quantity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from q2_eval import POLARITY, choose_device, load_checkpoint
from q2_ensemble_publish import load_models
from q2_train import apply_normalization, load_split, metrics, publish_c2

MODALITIES = ("text", "audio", "vision")
SUBSETS = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
SUBSET_INDEX = {s: i for i, s in enumerate(SUBSETS)}
FULL_MASK = 7


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_numpy(x: Any) -> np.ndarray:
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_q2_data(path: Path, split: str, norm: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        data = load_split(z, split, device)
    apply_normalization(data, norm)
    with np.load(path, allow_pickle=False) as z_text:
        data["raw_text"] = [str(x) for x in z_text[f"{split}_raw_text"].tolist()]
    return data


def load_attachment4(path: Path, norm: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        text_bert = torch.from_numpy(np.asarray(z["text_bert"], dtype=np.int64)).to(device)
        audio = torch.from_numpy(np.asarray(z["audio"], dtype=np.float32)).to(device)
        vision = torch.from_numpy(np.asarray(z["vision"], dtype=np.float32)).to(device)
        ids = [str(x) for x in z["ids"].tolist()]
        filenames = [str(x) for x in z["filenames"].tolist()]
        raw_text = [str(x) for x in z["raw_text"].tolist()]
    text_obs = text_bert[:, 1] > 0
    audio_obs = audio.abs().amax(dim=-1) > 1e-8
    vision_obs = vision.abs().amax(dim=-1) > 1e-8
    obs = torch.stack([text_obs, audio_obs, vision_obs], dim=1)
    text_semantic = text_obs & (text_bert[:, 0] != 101) & (text_bert[:, 0] != 102)
    maskable_obs = torch.stack([text_semantic, audio_obs, vision_obs], dim=1)
    audio = (audio - norm["audio_mean"]) / norm["audio_std"]
    vision = (vision - norm["vision_mean"]) / norm["vision_std"]
    audio = torch.where(audio_obs.unsqueeze(-1), audio, torch.zeros_like(audio))
    vision = torch.where(vision_obs.unsqueeze(-1), vision, torch.zeros_like(vision))
    return {
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "obs": obs,
        "maskable_obs": maskable_obs,
        "ids": ids,
        "filenames": filenames,
        "raw_text": raw_text,
        "labels": None,
        "regression": None,
    }


def select_stratified(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if n >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    raw_alloc = n * counts.astype(float) / counts.sum()
    alloc = np.floor(raw_alloc).astype(int)
    alloc = np.maximum(alloc, 1)
    while alloc.sum() > n:
        candidates = np.where(alloc > 1)[0]
        alloc[candidates[np.argmax(alloc[candidates] - raw_alloc[candidates])]] -= 1
    while alloc.sum() < n:
        candidates = np.where(alloc < counts)[0]
        alloc[candidates[np.argmax(raw_alloc[candidates] - alloc[candidates])]] += 1
    chosen: List[int] = []
    for c, k in zip(classes, alloc):
        pool = np.flatnonzero(labels == c)
        chosen.extend(rng.permutation(pool)[: int(k)].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


@torch.no_grad()
def ensemble_batch_scores(
    models: Sequence[Any],
    data: Dict[str, Any],
    obs: torch.Tensor,
    indices: torch.Tensor,
    target_classes: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class-target log mean probability, mean raw intensity and mean probabilities."""
    b = int(indices.numel())
    prob_sum = torch.zeros((b, 3), device=obs.device, dtype=torch.float32)
    raw_sum = torch.zeros((b,), device=obs.device, dtype=torch.float32)
    for model in models:
        out = model(data["text_bert"][indices, 0], obs, data["audio"][indices], data["vision"][indices])
        prob_sum += F.softmax(out["logits"].float(), dim=-1)
        raw_sum += out["raw_intensity"].float()
    mean_prob = prob_sum / float(len(models))
    mean_raw = raw_sum / float(len(models))
    if target_classes is None:
        target_classes = mean_prob.argmax(dim=-1)
    target_classes = target_classes.to(device=mean_prob.device, dtype=torch.long)
    cls_score = torch.log(mean_prob.gather(1, target_classes[:, None]).clamp_min(1e-9)).squeeze(1)
    return cls_score.detach().cpu().numpy(), mean_raw.detach().cpu().numpy(), mean_prob.detach().cpu().numpy()


def run_ensemble_outputs(models: Sequence[Any], data: Dict[str, Any], batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data["ids"])
    logits_all: List[np.ndarray] = []
    raw_all: List[np.ndarray] = []
    prob_all: List[np.ndarray] = []
    device = next(models[0].parameters()).device
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=device)
        obs = data["obs"][idx]
        cls_score, raw, prob = ensemble_batch_scores(models, data, obs, idx, None)
        # Use log mean probability itself as the published C2 logits.
        logits_all.append(np.log(np.clip(prob, 1e-9, 1.0)).astype(np.float32))
        raw_all.append(raw.astype(np.float32))
        prob_all.append(prob.astype(np.float32))
    return np.concatenate(logits_all), np.concatenate(raw_all), np.concatenate(prob_all)


def shapley3(values: np.ndarray) -> np.ndarray:
    """Exact Shapley for three players; values has shape [n,8]."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    phi = np.zeros((n, 3), dtype=np.float64)
    for i in range(3):
        for mask in range(8):
            if mask & (1 << i):
                continue
            s = mask.bit_count()
            weight = math.factorial(s) * math.factorial(3 - s - 1) / math.factorial(3)
            phi[:, i] += weight * (values[:, mask | (1 << i)] - values[:, mask])
    return phi.astype(np.float32)


def loo_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    full = values[:, FULL_MASK]
    out = np.zeros((values.shape[0], 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = full - values[:, FULL_MASK ^ (1 << i)]
    return out.astype(np.float32)


def modality_values(
    models: Sequence[Any],
    data: Dict[str, Any],
    sample_indices: np.ndarray,
    target_classes: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate all 8 modality subsets for each selected sample."""
    n = len(sample_indices)
    device = next(models[0].parameters()).device
    base_obs = data["obs"][torch.as_tensor(sample_indices, device=device)]
    all_obs: List[torch.Tensor] = []
    all_idx: List[int] = []
    all_target: List[int] = []
    for subset in SUBSETS:
        obs = base_obs.clone()
        for m in range(3):
            if m not in subset:
                obs[:, m, :] = False
        all_obs.append(obs.cpu())
        all_idx.extend(sample_indices.tolist())
        all_target.extend(target_classes.tolist())
    obs_cat = torch.cat(all_obs, dim=0)
    idx_cat = torch.as_tensor(all_idx, dtype=torch.long, device=device)
    target_cat = torch.as_tensor(all_target, dtype=torch.long, device=device)
    cls_scores: List[np.ndarray] = []
    raw_scores: List[np.ndarray] = []
    for start in range(0, len(idx_cat), batch_size):
        end = min(start + batch_size, len(idx_cat))
        cls, raw, _ = ensemble_batch_scores(
            models, data, obs_cat[start:end].to(device), idx_cat[start:end], target_cat[start:end]
        )
        cls_scores.append(cls)
        raw_scores.append(raw)
    cls_flat = np.concatenate(cls_scores).reshape(len(SUBSETS), n).T
    raw_flat = np.concatenate(raw_scores).reshape(len(SUBSETS), n).T
    return cls_flat.astype(np.float32), raw_flat.astype(np.float32)


def window_specs(positions: np.ndarray, window_size: int, stride: int) -> List[Tuple[int, int, np.ndarray]]:
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return []
    specs: List[Tuple[int, int, np.ndarray]] = []
    start = 0
    while start < len(positions):
        end = min(start + window_size, len(positions))
        specs.append((start, end, positions[start:end].copy()))
        if end >= len(positions):
            break
        start += stride
    return specs


def evaluate_perturbations(
    models: Sequence[Any],
    data: Dict[str, Any],
    records: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a list of explicit single-record perturbations."""
    if not records:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    device = next(models[0].parameters()).device
    cls_out: List[np.ndarray] = []
    raw_out: List[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        idx = torch.as_tensor([r["global_index"] for r in chunk], dtype=torch.long, device=device)
        target = torch.as_tensor([r["target_class"] for r in chunk], dtype=torch.long, device=device)
        obs = torch.stack([r["obs"] for r in chunk], dim=0).to(device)
        cls, raw, _ = ensemble_batch_scores(models, data, obs, idx, target)
        cls_out.append(cls)
        raw_out.append(raw)
    return np.concatenate(cls_out).astype(np.float32), np.concatenate(raw_out).astype(np.float32)


def paired_control_stats(continuous: Sequence[float], control: Sequence[float], seed: int, n_boot: int = 5000) -> Dict[str, Any]:
    """Paired comparison for same-budget occlusion controls."""
    x = np.asarray(continuous, dtype=np.float64)
    y = np.asarray(control, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape")
    n = int(len(x))
    if n == 0:
        return {
            "n_pairs": 0, "mean_difference": 0.0, "median_difference": 0.0,
            "bootstrap_95_ci": [0.0, 0.0], "wins": 0, "losses": 0, "ties": 0,
            "win_rate": 0.0, "sign_test_two_sided_p": 1.0, "paired_effect_dz": 0.0,
        }
    diff = x - y
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        boot[b] = float(np.mean(diff[rng.integers(0, n, size=n)]))
    tol = 1e-8
    wins = int(np.sum(diff > tol))
    losses = int(np.sum(diff < -tol))
    ties = int(n - wins - losses)
    sign_n = wins + losses
    if sign_n == 0:
        sign_p = 1.0
    else:
        k = min(wins, losses)
        log_den = sign_n * math.log(2.0)
        log_terms = [
            math.lgamma(sign_n + 1.0) - math.lgamma(i + 1.0) - math.lgamma(sign_n - i + 1.0) - log_den
            for i in range(k + 1)
        ]
        max_log = max(log_terms)
        log_tail = max_log + math.log(sum(math.exp(x - max_log) for x in log_terms))
        sign_p = min(1.0, 2.0 * math.exp(log_tail))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    return {
        "n_pairs": n,
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / n) if n else 0.0,
        "sign_test_two_sided_p": float(sign_p),
        "paired_effect_dz": float(np.mean(diff) / sd) if sd > 0 else 0.0,
        "definition": "difference = continuous_abs_class_delta - control_abs_class_delta; positive favors continuous-window evidence",
    }


def top1_concentration(rows: Sequence[Dict[str, Any]]) -> float:
    vals = [float(x["abs_class_delta"]) for x in rows]
    total = float(sum(vals))
    return float(max(vals) / total) if vals and total > 0 else 0.0


def run_local_occlusion(
    models: Sequence[Any],
    data: Dict[str, Any],
    sample_indices: np.ndarray,
    target_classes: np.ndarray,
    full_class_score: np.ndarray,
    full_raw: np.ndarray,
    window_size: int,
    stride: int,
    seed: int,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Continuous-window occlusion plus same-budget random and scatter controls."""
    records: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    maskable = data["maskable_obs"].detach().cpu().numpy()
    for pos, gidx in enumerate(sample_indices.tolist()):
        for m, modality in enumerate(MODALITIES):
            positions = np.flatnonzero(maskable[int(gidx), m])
            specs = window_specs(positions, window_size, stride)
            for wi, (start, end, actual_positions) in enumerate(specs):
                k = len(actual_positions)
                rng = np.random.default_rng(seed + 1009 * int(gidx) + 97 * m + wi)
                rand_start = int(rng.integers(0, max(1, len(positions) - k + 1)))
                random_positions = positions[rand_start:rand_start + k].copy()
                scatter_idx = rng.choice(len(positions), size=k, replace=False)
                scatter_positions = np.sort(positions[scatter_idx])
                # Keep the scattered control non-contiguous when the budget permits.
                for _ in range(8):
                    if k <= 1 or len(np.unique(np.diff(scatter_positions))) > 1 or len(positions) <= k:
                        break
                    scatter_idx = rng.choice(len(positions), size=k, replace=False)
                    scatter_positions = np.sort(positions[scatter_idx])
                base = data["obs"][int(gidx)].detach().clone()
                for pattern, pat_positions in (
                    ("continuous", actual_positions),
                    ("random_contiguous", random_positions),
                    ("point_scatter", scatter_positions),
                ):
                    obs = base.clone()
                    obs[m, pat_positions] = False
                    rid = len(records)
                    records.append({
                        "obs": obs,
                        "global_index": int(gidx),
                        "target_class": int(target_classes[pos]),
                        "sample_position": pos,
                        "modality": modality,
                        "modality_index": m,
                        "window_index": wi,
                        "pattern": pattern,
                        "window_start_in_valid": int(start),
                        "window_end_in_valid": int(end),
                        "positions": [int(x) for x in pat_positions.tolist()],
                    })
                    meta.append({
                        "record_id": rid,
                        "sample_position": pos,
                        "global_index": int(gidx),
                        "modality": modality,
                        "modality_index": m,
                        "window_index": wi,
                        "pattern": pattern,
                        "positions": [int(x) for x in pat_positions.tolist()],
                        "window_start_in_valid": int(start),
                        "window_end_in_valid": int(end),
                    })
    pert_cls, pert_raw = evaluate_perturbations(models, data, records, batch_size)
    for i, rec in enumerate(records):
        sp = int(rec["sample_position"])
        rec["class_delta"] = float(full_class_score[sp] - pert_cls[i])
        rec["raw_delta"] = float(full_raw[sp] - pert_raw[i])
        rec["abs_class_delta"] = abs(rec["class_delta"])
        rec["abs_raw_delta"] = abs(rec["raw_delta"])

    by_key = {
        (int(r["sample_position"]), int(r["modality_index"]), str(r["pattern"]), int(r["window_index"])): r
        for r in records
    }
    summaries: List[Dict[str, Any]] = []
    for pos, gidx in enumerate(sample_indices.tolist()):
        for m, modality in enumerate(MODALITIES):
            actual_all = [
                r for r in records
                if r["sample_position"] == pos and r["modality_index"] == m and r["pattern"] == "continuous"
            ]
            if not actual_all:
                continue
            actual_all.sort(key=lambda r: r["abs_class_delta"], reverse=True)
            top_actual = actual_all[0]
            control_vals = np.asarray(
                [
                    r["abs_class_delta"] for r in records
                    if r["sample_position"] == pos and r["modality_index"] == m and r["pattern"] in ("random_contiguous", "point_scatter")
                ],
                dtype=np.float64,
            )
            control_beat_rate = (
                float(np.mean(control_vals <= top_actual["abs_class_delta"])) if len(control_vals) else 0.0
            )
            random_rows = [
                by_key.get((pos, m, "random_contiguous", int(r["window_index"])))
                for r in actual_all
            ]
            scatter_rows = [
                by_key.get((pos, m, "point_scatter", int(r["window_index"])))
                for r in actual_all
            ]
            random_rows = [r for r in random_rows if r is not None]
            scatter_rows = [r for r in scatter_rows if r is not None]
            summaries.append({
                "sample_position": pos,
                "global_index": int(gidx),
                "modality": modality,
                "n_windows": len(actual_all),
                "top_window_index": int(top_actual["window_index"]),
                "top_positions": list(top_actual["positions"]),
                "top_class_delta": float(top_actual["class_delta"]),
                "top_abs_class_delta": float(top_actual["abs_class_delta"]),
                "top_raw_delta": float(top_actual["raw_delta"]),
                "top_abs_raw_delta": float(top_actual["abs_raw_delta"]),
                "top1_concentration": top1_concentration(actual_all),
                "mean_abs_class_delta": float(np.mean([r["abs_class_delta"] for r in actual_all])),
                "mean_abs_raw_delta": float(np.mean([r["abs_raw_delta"] for r in actual_all])),
                "control_beat_rate": control_beat_rate,
                "random_top1_concentration": top1_concentration(random_rows) if random_rows else 0.0,
                "scatter_top1_concentration": top1_concentration(scatter_rows) if scatter_rows else 0.0,
                "wins_vs_random_contiguous": int(sum(
                    float(a["abs_class_delta"]) > float(b["abs_class_delta"]) + 1e-8
                    for a, b in zip(actual_all, random_rows)
                )),
                "wins_vs_point_scatter": int(sum(
                    float(a["abs_class_delta"]) > float(b["abs_class_delta"]) + 1e-8
                    for a, b in zip(actual_all, scatter_rows)
                )),
            })

    agg: Dict[str, Any] = {}
    for pattern in ("continuous", "random_contiguous", "point_scatter"):
        vals = [r["abs_class_delta"] for r in records if r["pattern"] == pattern]
        raw_vals = [r["abs_raw_delta"] for r in records if r["pattern"] == pattern]
        agg[pattern] = {
            "n_perturbations": len(vals),
            "mean_abs_class_delta": float(np.mean(vals)) if vals else 0.0,
            "median_abs_class_delta": float(np.median(vals)) if vals else 0.0,
            "mean_abs_raw_delta": float(np.mean(raw_vals)) if raw_vals else 0.0,
        }

    paired_keys = sorted(set(
        (int(r["sample_position"]), int(r["modality_index"]), int(r["window_index"]))
        for r in records if r["pattern"] == "continuous"
    ))
    cont_pairs = [by_key[(p, m, "continuous", w)] for p, m, w in paired_keys if (p, m, "continuous", w) in by_key]
    rand_pairs = [by_key[(p, m, "random_contiguous", w)] for p, m, w in paired_keys if (p, m, "random_contiguous", w) in by_key]
    scat_pairs = [by_key[(p, m, "point_scatter", w)] for p, m, w in paired_keys if (p, m, "point_scatter", w) in by_key]
    valid_r = [(a, b) for a, b in zip(cont_pairs, rand_pairs)]
    valid_s = [(a, b) for a, b in zip(cont_pairs, scat_pairs)]
    stats_r = paired_control_stats(
        [a["abs_class_delta"] for a, _ in valid_r], [b["abs_class_delta"] for _, b in valid_r], seed=seed + 17011
    )
    stats_s = paired_control_stats(
        [a["abs_class_delta"] for a, _ in valid_s], [b["abs_class_delta"] for _, b in valid_s], seed=seed + 27011
    )
    agg["paired_continuous_vs_random_contiguous"] = stats_r
    agg["paired_continuous_vs_point_scatter"] = stats_s
    agg["continuous_win_rate_vs_random_contiguous"] = stats_r["win_rate"]
    agg["continuous_win_rate_vs_point_scatter"] = stats_s["win_rate"]
    agg["mean_top1_concentration_continuous"] = (
        float(np.mean([x["top1_concentration"] for x in summaries])) if summaries else 0.0
    )
    agg["mean_top1_concentration_random_contiguous"] = (
        float(np.mean([x["random_top1_concentration"] for x in summaries])) if summaries else 0.0
    )
    agg["mean_top1_concentration_point_scatter"] = (
        float(np.mean([x["scatter_top1_concentration"] for x in summaries])) if summaries else 0.0
    )

    # Keep every continuous window, not only the maximum-absolute window.  The
    # direction of class_delta matters: positive means the window supports the
    # frozen predicted class, negative means it is counter-evidence.
    window_summaries: List[Dict[str, Any]] = []
    for pos, gidx in enumerate(sample_indices.tolist()):
        for m, modality in enumerate(MODALITIES):
            actual_all = [
                r for r in records
                if r["sample_position"] == pos and r["modality_index"] == m and r["pattern"] == "continuous"
            ]
            if not actual_all:
                continue
            actual_all.sort(key=lambda r: int(r["window_index"]))
            class_ranks = percentile_rank([float(r["class_delta"]) for r in actual_all])
            raw_ranks = percentile_rank([float(r["raw_delta"]) for r in actual_all])
            for row, class_rank, raw_rank in zip(actual_all, class_ranks, raw_ranks):
                controls = [
                    by_key[(pos, m, pattern, int(row["window_index"]))]
                    for pattern in ("random_contiguous", "point_scatter")
                    if (pos, m, pattern, int(row["window_index"])) in by_key
                ]
                beat = (
                    float(np.mean([
                        float(c["abs_class_delta"]) <= float(row["abs_class_delta"]) + 1e-8
                        for c in controls
                    ]))
                    if controls else 0.0
                )
                window_summaries.append({
                    "sample_position": int(pos),
                    "global_index": int(gidx),
                    "modality": modality,
                    "modality_index": int(m),
                    "window_index": int(row["window_index"]),
                    "positions": list(row["positions"]),
                    "window_start_in_valid": int(row["window_start_in_valid"]),
                    "window_end_in_valid": int(row["window_end_in_valid"]),
                    "class_delta": float(row["class_delta"]),
                    "abs_class_delta": float(row["abs_class_delta"]),
                    "raw_delta": float(row["raw_delta"]),
                    "abs_raw_delta": float(row["abs_raw_delta"]),
                    "class_rank": float(class_rank),
                    "raw_rank": float(raw_rank),
                    "evidence_rank_score": float(0.5 * (class_rank + raw_rank)),
                    "control_beat_rate": float(beat),
                })
    return records, summaries, window_summaries, agg


def run_point_scan(
    models: Sequence[Any],
    data: Dict[str, Any],
    sample_indices: np.ndarray,
    target_classes: np.ndarray,
    full_class_score: np.ndarray,
    full_raw: np.ndarray,
    point_n: int,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Single-position occlusion scan on a deterministic prefix of the subset."""
    records: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []
    maskable = data["maskable_obs"].detach().cpu().numpy()
    take = min(int(point_n), len(sample_indices))
    for pos in range(take):
        gidx = int(sample_indices[pos])
        for m, modality in enumerate(MODALITIES):
            positions = np.flatnonzero(maskable[gidx, m])
            for pi, p in enumerate(positions.tolist()):
                obs = data["obs"][gidx].detach().clone()
                obs[m, int(p)] = False
                rid = len(records)
                records.append({
                    "obs": obs,
                    "global_index": gidx,
                    "target_class": int(target_classes[pos]),
                    "sample_position": pos,
                    "modality": modality,
                    "modality_index": m,
                    "position": int(p),
                })
                meta.append({
                    "record_id": rid, "sample_position": pos, "global_index": gidx,
                    "modality": modality, "position": int(p),
                })
    pert_cls, pert_raw = evaluate_perturbations(models, data, records, batch_size)
    for i, rec in enumerate(records):
        sp = int(rec["sample_position"])
        rec["class_delta"] = float(full_class_score[sp] - pert_cls[i])
        rec["raw_delta"] = float(full_raw[sp] - pert_raw[i])
        rec["abs_class_delta"] = abs(rec["class_delta"])
        rec["abs_raw_delta"] = abs(rec["raw_delta"])
    summaries: List[Dict[str, Any]] = []
    for pos in range(take):
        for m, modality in enumerate(MODALITIES):
            rows = [r for r in records if r["sample_position"] == pos and r["modality_index"] == m]
            if not rows:
                continue
            rows.sort(key=lambda r: r["abs_class_delta"], reverse=True)
            top = rows[0]
            summaries.append({
                "sample_position": pos,
                "global_index": int(sample_indices[pos]),
                "modality": modality,
                "top_position": int(top["position"]),
                "top_class_delta": float(top["class_delta"]),
                "top_abs_class_delta": float(top["abs_class_delta"]),
                "top_raw_delta": float(top["raw_delta"]),
                "top_abs_raw_delta": float(top["abs_raw_delta"]),
                "mean_abs_class_delta": float(np.mean([r["abs_class_delta"] for r in rows])),
            })
    agg = {
        "n_samples": take,
        "n_perturbations": len(records),
        "mean_top_abs_class_delta": float(np.mean([x["top_abs_class_delta"] for x in summaries])) if summaries else 0.0,
        "mean_abs_class_delta": float(np.mean([r["abs_class_delta"] for r in records])) if records else 0.0,
        "mean_top_abs_raw_delta": float(np.mean([x["top_abs_raw_delta"] for x in summaries])) if summaries else 0.0,
    }
    return records, summaries, agg


def load_offset_pack(path: Path, split: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        prefix = "attachment4" if split == "attachment4" else split
        return {
            "offsets": np.asarray(z[f"{prefix}_offsets"], dtype=np.int32),
            "valid": np.asarray(z[f"{prefix}_valid"], dtype=bool),
            "special": np.asarray(z[f"{prefix}_special"], dtype=bool),
            "match": np.asarray(z[f"{prefix}_match"], dtype=bool),
        }


def load_audit_map(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    audit = json.loads(path.read_text(encoding="utf-8"))
    return {str(x["sample_id"]): x for x in audit.get("samples", [])}


def map_evidence(
    modality: str,
    positions: Sequence[int],
    global_index: int,
    data: Dict[str, Any],
    offsets: Dict[str, np.ndarray],
    audit_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    positions = [int(x) for x in positions]
    raw_text = str(data["raw_text"][global_index])
    if modality == "text":
        spans: List[Tuple[int, int]] = []
        for p in positions:
            if p < len(offsets["valid"]) and offsets["valid"][global_index, p]:
                a, b = offsets["offsets"][global_index, p]
                if int(b) > int(a):
                    spans.append((int(a), int(b)))
        if spans:
            start = min(a for a, _ in spans)
            end = max(b for _, b in spans)
            return {
                "mapping_status": "exact",
                "mapping_precision": "exact_character_offsets_from_official_tokenizer",
                "text_char_start": start,
                "text_char_end": end,
                "text_snippet": raw_text[start:end],
                "token_positions": positions,
                "warning": "",
            }
        return {
            "mapping_status": "unavailable",
            "mapping_precision": "none",
            "text_char_start": None,
            "text_char_end": None,
            "text_snippet": "",
            "token_positions": positions,
            "warning": "no non-empty token offset in this window",
        }
    sid = str(data["ids"][global_index])
    meta = audit_map.get(sid, {})
    mapping_rows = {int(x["token_position"]): x for x in meta.get("position_to_time", [])}
    if modality == "audio":
        rows = [mapping_rows[p] for p in positions if p in mapping_rows]
        if rows:
            return {
                "mapping_status": "approximate",
                "mapping_precision": "approximate_proportional_token_position_to_measured_video_time",
                "time_start_seconds": float(min(float(x["time_start_est"]) for x in rows)),
                "time_end_seconds": float(max(float(x["time_end_est"]) for x in rows)),
                "token_positions": positions,
                "warning": "No original token-level timestamps were provided; audio times are proportional estimates.",
            }
        return {
            "mapping_status": "relative_only",
            "mapping_precision": "relative_position_no_video_metadata",
            "relative_start": float(min(positions) / 50.0) if positions else None,
            "relative_end": float((max(positions) + 1) / 50.0) if positions else None,
            "token_positions": positions,
            "warning": "No per-sample video metadata available for this validation explanation.",
        }
    rows = [mapping_rows[p] for p in positions if p in mapping_rows]
    if rows:
        frames = [int(x["frame_center_est"]) for x in rows if x.get("frame_center_est") is not None]
        return {
            "mapping_status": "approximate",
            "mapping_precision": "approximate_proportional_token_position_to_measured_video_time_and_frame",
            "time_start_seconds": float(min(float(x["time_start_est"]) for x in rows)),
            "time_end_seconds": float(max(float(x["time_end_est"]) for x in rows)),
            "frame_start_est": int(min(frames)) if frames else None,
            "frame_end_est": int(max(frames)) if frames else None,
            "token_positions": positions,
            "warning": "No original token-level timestamps were provided; visual frames are proportional estimates.",
        }
    return {
        "mapping_status": "relative_only",
        "mapping_precision": "relative_position_no_video_metadata",
        "relative_start": float(min(positions) / 50.0) if positions else None,
        "relative_end": float((max(positions) + 1) / 50.0) if positions else None,
        "token_positions": positions,
        "warning": "No per-sample video metadata available for this validation explanation.",
    }


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    for v in np.unique(x):
        idx = np.flatnonzero(x == v)
        ranks[idx] = float(np.mean(ranks[idx]))
    return (ranks - 1.0) / (n - 1.0) if n > 1 else np.ones((n,), dtype=np.float64)


def build_sample_explanations(
    data: Dict[str, Any],
    sample_indices: np.ndarray,
    target_classes: np.ndarray,
    logits: np.ndarray,
    raw: np.ndarray,
    class_phi: np.ndarray,
    raw_phi: np.ndarray,
    class_loo: np.ndarray,
    class_values: np.ndarray,
    raw_values: np.ndarray,
    local_summaries: List[Dict[str, Any]],
    window_summaries: List[Dict[str, Any]],
    offsets: Dict[str, np.ndarray],
    audit_map: Dict[str, Dict[str, Any]],
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    pred_cls, pred_intensity = publish_c2(logits, raw)
    out: List[Dict[str, Any]] = []
    for pos, gidx in enumerate(sample_indices.tolist()):
        c_abs = np.abs(class_phi[pos])
        r_abs = np.abs(raw_phi[pos])
        c_share = c_abs / c_abs.sum() if c_abs.sum() > 0 else np.zeros(3, dtype=np.float32)
        r_share = r_abs / r_abs.sum() if r_abs.sum() > 0 else np.zeros(3, dtype=np.float32)
        modality_effect = 0.5 * (c_share + r_share)
        strongest_idx = int(np.argmax(c_abs))
        support_idx = int(np.argmax(class_phi[pos]))
        support = MODALITIES[support_idx] if class_phi[pos, support_idx] > 0 else None
        primary_idx = max(range(3), key=lambda i: (float(modality_effect[i]), float(c_share[i]), float(r_share[i]), -i))
        evidence: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODALITIES}
        counter_evidence: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODALITIES}
        valid_support_any = False
        for m, modality in enumerate(MODALITIES):
            rows = [
                x for x in window_summaries
                if int(x["sample_position"]) == pos and x["modality"] == modality
            ]
            support_rows = [x for x in rows if float(x["class_delta"]) > 1e-8]
            support_rows.sort(
                key=lambda x: (
                    float(x["evidence_rank_score"]),
                    float(x["class_delta"]),
                    float(x["raw_delta"]),
                ),
                reverse=True,
            )
            for row in support_rows[: int(top_k)]:
                mapped = map_evidence(modality, row["positions"], int(gidx), data, offsets, audit_map)
                score = float(row["evidence_rank_score"])
                beat = float(row["control_beat_rate"])
                if score >= 0.8 and beat >= 0.75:
                    level = "high"
                elif score >= 0.5 or beat >= 0.5:
                    level = "medium"
                else:
                    level = "low"
                evidence[modality].append({
                    "window_index": int(row["window_index"]),
                    "positions": list(row["positions"]),
                    "class_delta": float(row["class_delta"]),
                    "abs_class_delta": float(row["abs_class_delta"]),
                    "raw_delta": float(row["raw_delta"]),
                    "abs_raw_delta": float(row["abs_raw_delta"]),
                    "evidence_direction": "support",
                    "evidence_rank_score": score,
                    "control_beat_rate": beat,
                    "evidence_level": level,
                    **mapped,
                })
                valid_support_any = True

            counter_rows = [x for x in rows if float(x["class_delta"]) < -1e-8]
            counter_rows.sort(
                key=lambda x: (float(x["abs_class_delta"]), float(x["abs_raw_delta"])),
                reverse=True,
            )
            if counter_rows:
                row = counter_rows[0]
                mapped = map_evidence(modality, row["positions"], int(gidx), data, offsets, audit_map)
                counter_evidence[modality].append({
                    "window_index": int(row["window_index"]),
                    "positions": list(row["positions"]),
                    "class_delta": float(row["class_delta"]),
                    "abs_class_delta": float(row["abs_class_delta"]),
                    "raw_delta": float(row["raw_delta"]),
                    "abs_raw_delta": float(row["abs_raw_delta"]),
                    "evidence_direction": "counter",
                    "evidence_rank_score": float(row["evidence_rank_score"]),
                    "control_beat_rate": float(row["control_beat_rate"]),
                    "evidence_level": "counter",
                    **mapped,
                })
        no_evidence_reason = None if valid_support_any else "no positive continuous-window contribution exceeded the numerical evidence threshold"
        out.append({
            "sample_id": str(data["ids"][gidx]),
            "global_index": int(gidx),
            "polarity": POLARITY[int(pred_cls[gidx])],
            "polarity_index": int(pred_cls[gidx]),
            "intensity": float(pred_intensity[gidx]),
            "raw_intensity": float(raw[gidx]),
            "log_mean_probability": [float(x) for x in logits[gidx].tolist()],
            "target_class_for_explanation": int(target_classes[pos]),
            "strongest_modality": MODALITIES[strongest_idx],
            "support_modality": support,
            "primary_modality": MODALITIES[primary_idx],
            "modality_effect_text": float(modality_effect[0]),
            "modality_effect_audio": float(modality_effect[1]),
            "modality_effect_vision": float(modality_effect[2]),
            "class_contrib_text": float(class_phi[pos, 0]),
            "class_contrib_audio": float(class_phi[pos, 1]),
            "class_contrib_vision": float(class_phi[pos, 2]),
            "class_abs_share_text": float(c_share[0]),
            "class_abs_share_audio": float(c_share[1]),
            "class_abs_share_vision": float(c_share[2]),
            "intensity_contrib_text": float(raw_phi[pos, 0]),
            "intensity_contrib_audio": float(raw_phi[pos, 1]),
            "intensity_contrib_vision": float(raw_phi[pos, 2]),
            "intensity_abs_share_text": float(r_share[0]),
            "intensity_abs_share_audio": float(r_share[1]),
            "intensity_abs_share_vision": float(r_share[2]),
            "loo_text": float(class_loo[pos, 0]),
            "loo_audio": float(class_loo[pos, 1]),
            "loo_vision": float(class_loo[pos, 2]),
            "class_shapley_sum_error": float(abs(class_phi[pos].sum() - (class_values[pos, FULL_MASK] - class_values[pos, 0]))),
            "intensity_shapley_sum_error": float(abs(raw_phi[pos].sum() - (raw_values[pos, FULL_MASK] - raw_values[pos, 0]))),
            "key_evidence": evidence,
            "counter_evidence": counter_evidence,
            "no_evidence_reason": no_evidence_reason,
        })
    return out


def validation_metrics(
    logits: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    regression: np.ndarray,
) -> Dict[str, float]:
    return metrics(logits, raw, labels, regression, "c2")


def summarize_shapley(
    class_phi: np.ndarray,
    raw_phi: np.ndarray,
    class_loo: np.ndarray,
    class_values: np.ndarray,
    raw_values: np.ndarray,
) -> Dict[str, Any]:
    c_sum_err = np.abs(class_phi.sum(1) - (class_values[:, FULL_MASK] - class_values[:, 0]))
    r_sum_err = np.abs(raw_phi.sum(1) - (raw_values[:, FULL_MASK] - raw_values[:, 0]))
    c_main = np.argmax(np.abs(class_phi), axis=1)
    l_main = np.argmax(np.abs(class_loo), axis=1)
    c_abs = np.abs(class_phi)
    r_abs = np.abs(raw_phi)
    return {
        "class_shapley_sum_error_max": float(c_sum_err.max()) if len(c_sum_err) else 0.0,
        "class_shapley_sum_error_mean": float(c_sum_err.mean()) if len(c_sum_err) else 0.0,
        "intensity_shapley_sum_error_max": float(r_sum_err.max()) if len(r_sum_err) else 0.0,
        "intensity_shapley_sum_error_mean": float(r_sum_err.mean()) if len(r_sum_err) else 0.0,
        "main_modality_agreement_shapley_vs_loo": float(np.mean(c_main == l_main)) if len(c_main) else 0.0,
        "mean_abs_signed_difference_shapley_vs_loo": float(np.mean(np.abs(class_phi - class_loo))),
        "mean_class_abs_share_text": float(np.mean(c_abs[:, 0] / np.clip(c_abs.sum(1), 1e-12, None))),
        "mean_class_abs_share_audio": float(np.mean(c_abs[:, 1] / np.clip(c_abs.sum(1), 1e-12, None))),
        "mean_class_abs_share_vision": float(np.mean(c_abs[:, 2] / np.clip(c_abs.sum(1), 1e-12, None))),
        "mean_intensity_abs_share_text": float(np.mean(r_abs[:, 0] / np.clip(r_abs.sum(1), 1e-12, None))),
        "mean_intensity_abs_share_audio": float(np.mean(r_abs[:, 1] / np.clip(r_abs.sum(1), 1e-12, None))),
        "mean_intensity_abs_share_vision": float(np.mean(r_abs[:, 2] / np.clip(r_abs.sum(1), 1e-12, None))),
        "n": int(len(class_phi)),
    }


def verify_freeze_manifest(manifest_path: Path | None, checkpoints: Sequence[Path]) -> Dict[str, Any]:
    if manifest_path is None:
        return {"checked": False, "reason": "no manifest supplied"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [str(x) for x in manifest["selected_model"]["checkpoint_sha256"]]
    actual = [sha256_file(Path(p)) for p in checkpoints]
    ok = expected == actual
    return {
        "checked": True,
        "ok": bool(ok),
        "expected": expected,
        "actual": actual,
        "manifest": str(manifest_path),
    }


def mapping_summary(explanations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Dict[str, int]] = {}
    for sample in explanations:
        for modality, rows in sample["key_evidence"].items():
            out.setdefault(modality, {})
            for row in rows:
                status = str(row.get("mapping_status", "unknown"))
                out[modality][status] = out[modality].get(status, 0) + 1
    return out


def run_validation(args: argparse.Namespace, device: torch.device) -> None:
    started = time.time()
    models, checkpoints, norm, kind, seeds = load_models(args.checkpoints, args.model_dir, device)
    freeze_check = verify_freeze_manifest(args.freeze_manifest, args.checkpoints)
    data = load_q2_data(args.data, args.split, norm, device)
    offsets = load_offset_pack(args.offsets, args.split)
    labels = data["labels"].detach().cpu().numpy()
    regression = data["regression"].detach().cpu().numpy()
    logits, raw, probs = run_ensemble_outputs(models, data, args.batch_size)
    full_metrics = validation_metrics(logits, raw, labels, regression)
    selected = select_stratified(labels, args.n, args.seed)
    target = logits[selected].argmax(axis=1).astype(np.int64)
    full_class = logits[selected, target].astype(np.float32)
    full_raw_selected = raw[selected].astype(np.float32)
    print(json.dumps({"stage": "full_metrics", "n": len(labels), "metrics": full_metrics, "selected_n": len(selected)}, ensure_ascii=False), flush=True)
    class_values, raw_values = modality_values(models, data, selected, target, args.batch_size)
    class_phi = shapley3(class_values)
    raw_phi = shapley3(raw_values)
    class_loo = loo_values(class_values)
    shapley_report = summarize_shapley(class_phi, raw_phi, class_loo, class_values, raw_values)
    print(json.dumps({"stage": "shapley", **shapley_report}, ensure_ascii=False), flush=True)
    _, local_summaries, local_window_summaries, local_agg = run_local_occlusion(
        models, data, selected, target, full_class, full_raw_selected,
        args.window_size, args.stride, args.seed, args.batch_size,
    )
    print(json.dumps({"stage": "local_occlusion", **local_agg}, ensure_ascii=False), flush=True)
    _, point_summaries, point_agg = run_point_scan(
        models, data, selected, target, full_class, full_raw_selected,
        args.point_n, args.batch_size,
    )
    print(json.dumps({"stage": "point_scan", **point_agg}, ensure_ascii=False), flush=True)
    explanations = build_sample_explanations(
        data, selected, target, logits, raw, class_phi, raw_phi, class_loo,
        class_values, raw_values, local_summaries, local_window_summaries, offsets, {}, args.top_k,
    )
    report = {
        "mode": "validation",
        "protocol": {
            "question": 3,
            "split": args.split,
            "selection": "stratified by class, validation only",
            "n_selected": int(len(selected)),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "point_scan_n": int(min(args.point_n, len(selected))),
            "top_k_per_modality": int(args.top_k),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "class_explanation_target": "published predicted class log mean probability",
            "intensity_explanation_target": "mean raw intensity",
            "modality_value": "exact Shapley over all 8 modality subsets",
            "local_evidence": "continuous-window occlusion with same-budget random-contiguous and point-scatter controls",
            "local_pairing": "paired by sample x modality x window index",
            "paired_statistics": "5000-resample paired bootstrap 95% CI and exact two-sided sign test",
            "primary_modality_rule": "argmax of 0.5*(class absolute share + intensity absolute share), ties by class share, intensity share, fixed text/audio/vision order",
            "evidence_ranking_rule": "support: positive continuous-window class contribution ranked by average percentile of signed class and raw-intensity deltas; counter: most negative class contribution; top-k support retained",
            "evidence_level_rule": "high: rank>=0.8 and control_beat_rate>=0.75; medium: rank>=0.5 or control_beat_rate>=0.5; low otherwise",
            "attention_maps_used_as_explanation": False,
        },
        "model": {
            "kind": kind,
            "seeds": seeds,
            "checkpoint_paths": [str(x) for x in args.checkpoints],
            "freeze_manifest_check": freeze_check,
        },
        "full_validation_metrics": full_metrics,
        "selected_sample_ids": [str(data["ids"][int(i)]) for i in selected.tolist()],
        "shapley_vs_loo": shapley_report,
        "local_occlusion": local_agg,
        "point_scan": point_agg,
        "mapping_summary": mapping_summary(explanations),
        "elapsed_seconds": float(time.time() - started),
        "notes": [
            "Attachment 3 and attachment 4 were not read in validation mode.",
            "Validation explanations are used only to freeze Q3 explanation parameters.",
            "Text evidence uses exact tokenizer character offsets; audio/vision validation evidence is relative because no per-sample video metadata is available.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "q3_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "q3_validation_explanations.json").write_text(json.dumps(explanations, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "q3_validation_arrays.npz",
        selected_indices=selected,
        target_classes=target,
        class_values=class_values,
        raw_values=raw_values,
        class_phi=class_phi,
        raw_phi=raw_phi,
        class_loo=class_loo,
        full_logits=logits,
        full_raw=raw,
    )
    print(json.dumps({"done": True, "mode": "validation", "out_dir": str(args.out_dir), "report": report}, ensure_ascii=False), flush=True)


def run_attachment4(args: argparse.Namespace, device: torch.device) -> None:
    started = time.time()
    models, checkpoints, norm, kind, seeds = load_models(args.checkpoints, args.model_dir, device)
    freeze_check = verify_freeze_manifest(args.freeze_manifest, args.checkpoints)
    data = load_attachment4(args.attachment4, norm, device)
    offsets = load_offset_pack(args.offsets, "attachment4")
    audit_map = load_audit_map(args.audit)
    logits, raw, probs = run_ensemble_outputs(models, data, args.batch_size)
    all_indices = np.arange(len(data["ids"]), dtype=np.int64)
    target = logits.argmax(axis=1).astype(np.int64)
    full_class = logits[np.arange(len(target)), target].astype(np.float32)
    full_raw = raw.astype(np.float32)
    class_values, raw_values = modality_values(models, data, all_indices, target, args.batch_size)
    class_phi = shapley3(class_values)
    raw_phi = shapley3(raw_values)
    class_loo = loo_values(class_values)
    _, local_summaries, local_window_summaries, local_agg = run_local_occlusion(
        models, data, all_indices, target, full_class, full_raw,
        args.window_size, args.stride, args.seed, args.batch_size,
    )
    _, point_summaries, point_agg = run_point_scan(
        models, data, all_indices, target, full_class, full_raw,
        len(all_indices), args.batch_size,
    )
    explanations = build_sample_explanations(
        data, all_indices, target, logits, raw, class_phi, raw_phi, class_loo,
        class_values, raw_values, local_summaries, local_window_summaries, offsets, audit_map, args.top_k,
    )
    report = {
        "mode": "attachment4",
        "protocol": {
            "question": 3,
            "split": "attachment4",
            "n": int(len(all_indices)),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "point_scan_n": int(min(args.point_n, len(all_indices))),
            "top_k_per_modality": int(args.top_k),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "frozen_explanation_parameters": True,
            "attachment4_used_for_tuning": False,
            "local_pairing": "paired by sample x modality x window index",
            "paired_statistics": "5000-resample paired bootstrap 95% CI and exact two-sided sign test",
        },
        "model": {
            "kind": kind,
            "seeds": seeds,
            "checkpoint_paths": [str(x) for x in args.checkpoints],
            "freeze_manifest_check": freeze_check,
        },
        "shapley_vs_loo": summarize_shapley(class_phi, raw_phi, class_loo, class_values, raw_values),
        "local_occlusion": local_agg,
        "point_scan": point_agg,
        "mapping_summary": mapping_summary(explanations),
        "elapsed_seconds": float(time.time() - started),
        "notes": [
            "Attachment 4 was used only after freezing the predictor and explanation parameters.",
            "Audio and vision evidence locations are approximate proportional mappings because attachment 4 provides no original token-level timestamps.",
            "Text evidence locations are exact tokenizer character offsets.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "q3_attachment4_raw.json").write_text(json.dumps({"report": report, "samples": explanations}, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "q3_attachment4_arrays.npz",
        target_classes=target,
        class_values=class_values,
        raw_values=raw_values,
        class_phi=class_phi,
        raw_phi=raw_phi,
        class_loo=class_loo,
        full_logits=logits,
        full_raw=raw,
    )
    print(json.dumps({"done": True, "mode": "attachment4", "out_dir": str(args.out_dir), "report": report}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validation", "attachment4"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--attachment4", type=Path, default=None)
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--freeze-manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--point-n", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "validation":
        run_validation(args, device)
    else:
        if args.attachment4 is None:
            raise SystemExit("--attachment4 is required in attachment4 mode")
        run_attachment4(args, device)


if __name__ == "__main__":
    main()


