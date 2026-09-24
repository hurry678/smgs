#!/usr/bin/env python3
"""问题一方法对比实验（独立于正式特征流水线）。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import extract_q1_features as ext

DEFAULT_FPS = (2.5, 5.0, 10.0)
ALIGNMENT_NAMES = ("uniform", "length_weighted", "energy_dp_20", "full_dp_10", "full_dp_20", "full_dp_40")
SIGMA_VALUES = (0.0, 0.5, 0.75, 1.0)
TEXT_KEYS = ("bert", "hash")
AUDIO_SETS = {
    "full_40": list(range(40)),
    "mfcc_delta_23": list(range(9, 32)),
    "spectral_mel_31": list(range(9, 40)),
    "core_prosody_9": list(range(0, 9)),
    "energy_voiced_f0_3": [0, 7, 8],
}
VISION_SETS = {
    "full_90": list(range(90)),
    "no_hog_54": list(range(0, 49)) + list(range(85, 90)),
    "compact_38": list(range(0, 33)) + list(range(85, 90)),
}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 method comparison experiments")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--text-feature-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--probe-repeats", type=int, default=10)
    return parser.parse_args(argv)

def safe_tag(value: float) -> str:
    return str(value).replace(".", "p")

def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.sample_id:
        wanted = set(args.sample_id)
        rows = [row for row in rows if row.get("sample_id") in wanted]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    return rows

def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{ext.safe_sample_key(sample_id)}.npz"

def load_cache(path: Path, fps_values: tuple[float, ...]) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    sample = {
        "sample_id": str(data["sample_id"].item()),
        "video_id": str(data["video_id"].item()),
        "clip_id": str(data["clip_id"].item()),
        "annotation": str(data["annotation"].item()),
        "label": float(data["label"].item()),
        "raw_text": str(data["raw_text"].item()),
        "tokens": json.loads(str(data["tokens_json"].item())),
        "text_bert": np.asarray(data["text_bert"], dtype=np.float32),
        "text_bert_valid": np.asarray(data["text_bert_valid"], dtype=bool),
        "text_hash": np.asarray(data["text_hash"], dtype=np.float32),
        "text_hash_valid": np.asarray(data["text_hash_valid"], dtype=bool),
        "bert_meta": json.loads(str(data["bert_meta_json"].item())),
        "audio_raw": np.asarray(data["audio_raw"], dtype=np.float32),
        "audio_valid": np.asarray(data["audio_valid"], dtype=bool),
        "audio_times": np.asarray(data["audio_times"], dtype=np.float32),
        "voiced_flag": np.asarray(data["voiced_flag"], dtype=bool),
        "audio_meta": json.loads(str(data["audio_meta_json"].item())),
        "vision": {},
    }
    for fps in fps_values:
        tag = safe_tag(float(fps))
        sample["vision"][float(fps)] = {
            "raw": np.asarray(data[f"vision_{tag}_raw"], dtype=np.float32),
            "valid": np.asarray(data[f"vision_{tag}_valid"], dtype=bool),
            "times": np.asarray(data[f"vision_{tag}_times"], dtype=np.float32),
            "meta": json.loads(str(data[f"vision_{tag}_meta_json"].item())),
        }
    return sample

def build_cache(rows: list[dict[str, str]], args: argparse.Namespace, fps_values: tuple[float, ...]) -> list[dict[str, Any]]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        path = cache_path(args.cache_dir, sample_id)
        if path.exists() and not args.overwrite_cache:
            samples.append(load_cache(path, fps_values))
            if args.progress_every and index % args.progress_every == 0:
                print(f"[cache {index}/{len(rows)}] loaded {sample_id}", flush=True)
            continue
        video_path = ext.resolve_video_path(row, args.raw_root)
        tokens = ext.tokenize_with_spans(row.get("text", ""))
        bert_features, bert_valid, bert_meta = ext.load_precomputed_text(args.text_feature_dir, sample_id)
        hash_features = ext.hash_text_features(tokens)
        if bert_features.shape != (len(tokens), ext.TEXT_DIM):
            raise RuntimeError(f"BERT shape mismatch for {sample_id}: {bert_features.shape}")
        audio_raw, audio_valid, audio_times, voiced_flag, audio_meta = ext.extract_audio_features(video_path)
        vision_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for fps in fps_values:
            old_fps = ext.VISION_FPS
            try:
                ext.VISION_FPS = float(fps)
                vision_raw, vision_valid, vision_times, vision_meta = ext.extract_visual_features(video_path)
            finally:
                ext.VISION_FPS = old_fps
            vision_payload[safe_tag(float(fps))] = (vision_raw, vision_valid, vision_times, vision_meta)
        payload: dict[str, np.ndarray] = {
            "sample_id": np.asarray(sample_id),
            "video_id": np.asarray(str(row.get("video_id", ""))),
            "clip_id": np.asarray(str(row.get("clip_id", ""))),
            "annotation": np.asarray(str(row.get("annotation", ""))),
            "label": np.asarray(float(row.get("label", "nan")), dtype=np.float32),
            "raw_text": np.asarray(row.get("text", "")),
            "tokens_json": np.asarray(json.dumps(tokens, ensure_ascii=False)),
            "text_bert": bert_features.astype(np.float32),
            "text_bert_valid": bert_valid.astype(bool),
            "text_hash": hash_features.astype(np.float32),
            "text_hash_valid": np.ones_like(hash_features, dtype=bool),
            "bert_meta_json": np.asarray(json.dumps(bert_meta, ensure_ascii=False)),
            "audio_raw": audio_raw.astype(np.float32),
            "audio_valid": audio_valid.astype(bool),
            "audio_times": audio_times.astype(np.float32),
            "voiced_flag": voiced_flag.astype(bool),
            "audio_meta_json": np.asarray(json.dumps(audio_meta, ensure_ascii=False)),
        }
        for tag, (vision_raw, vision_valid, vision_times, vision_meta) in vision_payload.items():
            payload[f"vision_{tag}_raw"] = vision_raw.astype(np.float32)
            payload[f"vision_{tag}_valid"] = vision_valid.astype(bool)
            payload[f"vision_{tag}_times"] = vision_times.astype(np.float32)
            payload[f"vision_{tag}_meta_json"] = np.asarray(json.dumps(vision_meta, ensure_ascii=False))
        np.savez_compressed(path, **payload)
        samples.append(load_cache(path, fps_values))
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"[cache {index}/{len(rows)}] built {sample_id}", flush=True)
    return samples

def weighted_token_weights(tokens: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [1.0 + 0.52 * len(item["clean"]) + (0.30 if item.get("trailing_punct") else 0.0) for item in tokens],
        dtype=np.float64,
    )

def uniform_boundaries(word_count: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, max(0.0, duration), word_count + 1, dtype=np.float64)
    return edges[:-1].astype(np.float32), edges[1:].astype(np.float32)

def length_weighted_boundaries(tokens: list[dict[str, Any]], duration: float) -> tuple[np.ndarray, np.ndarray]:
    weights = weighted_token_weights(tokens)
    if len(weights) == 0 or float(weights.sum()) <= 0:
        return uniform_boundaries(len(tokens), duration)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)]) / float(weights.sum())
    edges = cumulative * max(0.0, duration)
    return edges[:-1].astype(np.float32), edges[1:].astype(np.float32)

def _aggregate_energy_score(audio_features: np.ndarray, audio_valid: np.ndarray, working_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_valid = np.all(audio_valid, axis=1) if len(audio_valid) else np.zeros((0,), dtype=bool)
    energy = ext._robust_unit_scale(audio_features[:, 0])
    score = energy.astype(np.float32)
    score[~frame_valid] = 0.0
    threshold = max(0.45, float(np.percentile(score[frame_valid], 35.0))) if np.any(frame_valid) else 0.45
    active = score >= threshold
    score_clean = score.copy()
    if np.any(frame_valid):
        indices = np.flatnonzero(frame_valid)
        if len(indices) >= 3:
            score_clean[indices] = ext.gaussian_filter1d(score_clean[indices], sigma=0.75, mode="nearest")
    score_clean[~frame_valid] = 0.0
    group_size = max(1, int(round((1.0 / working_hz) / (ext.AUDIO_HOP / ext.AUDIO_SR))))
    score_work, valid_work = ext._aggregate_groups(score_clean, frame_valid, group_size)
    activity_work, _ = ext._aggregate_groups(active.astype(np.float32), frame_valid, group_size)
    return score_clean, frame_valid, score_work, activity_work

def energy_dp_boundaries(
    tokens: list[dict[str, Any]],
    audio_features: np.ndarray,
    audio_valid: np.ndarray,
    duration: float,
    working_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    word_count = len(tokens)
    if word_count == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    frame_count = len(audio_features)
    if frame_count == 0:
        return uniform_boundaries(word_count, duration)
    _, _, score_work, activity_work = _aggregate_energy_score(audio_features, audio_valid, working_hz)
    working_count = len(score_work)
    if working_count < word_count:
        return uniform_boundaries(word_count, duration)
    weights = weighted_token_weights(tokens)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    target_boundaries = cumulative / cumulative[-1] * working_count
    average_word_frames = working_count / max(word_count, 1)
    radius = max(3, int(round(0.72 * average_word_frames)) + 3)

    candidates: list[np.ndarray] = []
    lead = max(2, int(round(0.12 * working_count)))
    start_candidates = np.arange(0, min(lead, working_count - word_count) + 1)
    candidates.append(np.unique(start_candidates if len(start_candidates) else np.asarray([0])).astype(np.int32))
    transitions = np.flatnonzero(np.diff(activity_work.astype(np.int8)) != 0) + 1
    minima, _ = ext.find_peaks(-score_work, distance=2)
    for boundary_index in range(1, word_count):
        target = float(target_boundaries[boundary_index])
        left = max(boundary_index, int(math.floor(target - radius)))
        right = min(working_count - (word_count - boundary_index), int(math.ceil(target + radius)))
        if right < left:
            left = right = int(np.clip(round(target), boundary_index, working_count - (word_count - boundary_index)))
        values = list(range(left, right + 1))
        values.extend(int(v) for v in transitions[(transitions >= left) & (transitions <= right)])
        values.extend(int(v) for v in minima[(minima >= left) & (minima <= right)])
        candidates.append(np.unique(np.asarray(values, dtype=np.int32)))
    tail = max(2, int(round(0.12 * working_count)))
    end_candidates = np.arange(max(word_count - 1, working_count - tail), working_count + 1)
    end_candidates = end_candidates[end_candidates >= word_count - 1]
    candidates.append(np.unique(end_candidates if len(end_candidates) else np.asarray([working_count])).astype(np.int32))

    negative_inf = -1e18
    dp = np.full(len(candidates[0]), negative_inf, dtype=np.float64)
    back: list[np.ndarray] = []
    for index, boundary in enumerate(candidates[0]):
        dp[index] = 0.55 * (1.0 - float(score_work[boundary])) if boundary < working_count else 0.0
    activity_prefix = np.concatenate([[0.0], np.cumsum(activity_work, dtype=np.float64)])
    for word_index in range(1, word_count + 1):
        current = np.full(len(candidates[word_index]), negative_inf, dtype=np.float64)
        previous_indices = np.full(len(candidates[word_index]), -1, dtype=np.int32)
        expected_length = max(1.0, float(weights[word_index - 1] / weights.sum() * working_count))
        for current_index, right in enumerate(candidates[word_index]):
            best_value = negative_inf
            best_previous = -1
            for previous_index, left in enumerate(candidates[word_index - 1]):
                if left >= right or dp[previous_index] <= negative_inf / 2:
                    continue
                actual_length = int(right - left)
                active_fraction = float((activity_prefix[right] - activity_prefix[left]) / max(actual_length, 1))
                log_ratio = math.log((actual_length + 0.5) / (expected_length + 0.5))
                duration_penalty = -2.0 * log_ratio * log_ratio
                short_penalty = -2.5 if actual_length < 1 else 0.0
                pause_reward = 0.75 * (1.0 - float(score_work[right])) if right < working_count else 0.0
                value = dp[previous_index] + 3.0 * active_fraction + duration_penalty + short_penalty + pause_reward
                if value > best_value:
                    best_value = value
                    best_previous = previous_index
            current[current_index] = best_value
            previous_indices[current_index] = best_previous
        dp = current
        back.append(previous_indices)
    final_index = int(np.argmax(dp))
    boundary_indices = [final_index]
    for reverse_step in range(word_count - 1, -1, -1):
        previous = int(back[reverse_step][boundary_indices[-1]])
        boundary_indices.append(0 if previous < 0 else previous)
    boundary_indices.reverse()
    boundary_work = np.asarray([candidates[i][boundary_indices[i]] for i in range(word_count + 1)], dtype=np.int32)
    group_size = max(1, int(round((1.0 / working_hz) / (ext.AUDIO_HOP / ext.AUDIO_SR))))
    boundaries = np.rint(boundary_work.astype(np.float64) * group_size).astype(np.int32)
    boundaries[0] = max(0, min(int(boundaries[0]), frame_count))
    boundaries[-1] = max(0, min(int(boundaries[-1]), frame_count))
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1] + 1)
    if boundaries[-1] > frame_count:
        boundaries = np.linspace(0, frame_count, word_count + 1).round().astype(np.int32)
    frame_hop = ext.AUDIO_HOP / ext.AUDIO_SR
    starts = boundaries[:-1].astype(np.float64) * frame_hop
    ends = boundaries[1:].astype(np.float64) * frame_hop
    ends = np.minimum(ends, float(duration)).astype(np.float32)
    starts = np.minimum(starts, ends).astype(np.float32)
    return starts, ends

def alignment_for_sample(sample: dict[str, Any], name: str, audio_smoothed: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tokens = sample["tokens"]
    duration = float(sample["audio_meta"].get("duration_sec", 0.0))
    decoded_end = duration
    if len(sample["audio_times"]) > 0:
        decoded_end = max(decoded_end, float(sample["audio_times"][-1]) + ext.AUDIO_HOP / ext.AUDIO_SR / 2.0)
    if name == "uniform":
        starts, ends = uniform_boundaries(len(tokens), duration)
        return starts, ends, {"backend": "uniform_duration", "working_hz": None}
    if name == "length_weighted":
        starts, ends = length_weighted_boundaries(tokens, duration)
        return starts, ends, {"backend": "length_weighted", "working_hz": None}
    if name == "energy_dp_20":
        starts, ends = energy_dp_boundaries(tokens, audio_smoothed, sample["audio_valid"], decoded_end, 20.0)
        return starts, ends, {"backend": "energy_dp_20", "working_hz": 20.0}
    if name.startswith("full_dp_"):
        working_hz = float(name.rsplit("_", 1)[1])
        starts, ends, confidence, fallback, meta = ext.align_words_to_audio(
            tokens, audio_smoothed, sample["audio_valid"], sample["audio_times"], sample["voiced_flag"], duration, working_hz=working_hz
        )
        return starts, ends, meta | {"candidate_confidence": confidence, "candidate_fallback": fallback}
    raise KeyError(name)

def summarize_words(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dimension = int(features.shape[1]) if features.ndim == 2 else 0
    if len(features) == 0:
        return np.zeros(dimension * 2 + 1, dtype=np.float32)
    row_valid = np.all(valid, axis=1) if valid.ndim == 2 else np.asarray(valid, dtype=bool)
    selected = features[row_valid]
    if len(selected) == 0:
        return np.concatenate([np.zeros(dimension, dtype=np.float32), np.zeros(dimension, dtype=np.float32), [0.0]])
    mean = selected.mean(axis=0).astype(np.float32)
    std = selected.std(axis=0).astype(np.float32)
    ratio = float(len(selected) / max(len(features), 1))
    return np.concatenate([mean, std, np.asarray([ratio], dtype=np.float32)])

def frame_reference_quality(sample: dict[str, Any], starts: np.ndarray, ends: np.ndarray, reference_score: np.ndarray) -> dict[str, float]:
    tokens = sample["tokens"]
    frame_count = len(sample["audio_raw"])
    frame_hop = ext.AUDIO_HOP / ext.AUDIO_SR
    if frame_count == 0 or len(starts) == 0:
        return {"active_fraction": 0.0, "pause_quality": 0.0, "duration_fit": 0.0, "interval_valid": 0.0}
    edges = np.arange(frame_count + 1, dtype=np.float64) * frame_hop
    weights = weighted_token_weights(tokens)
    active_values: list[float] = []
    pause_values: list[float] = []
    duration_values: list[float] = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        left = int(np.searchsorted(edges, float(start), side="right") - 1)
        right = int(np.searchsorted(edges, float(end), side="left"))
        left = max(0, min(left, frame_count - 1))
        right = max(left + 1, min(right, frame_count))
        segment = reference_score[left:right]
        active_values.append(float(segment.mean()) if len(segment) else 0.0)
        expected = max(1.0, float(weights[i] / weights.sum() * frame_count))
        actual = max(1, right - left)
        duration_values.append(math.exp(-abs(math.log((actual + 0.5) / (expected + 0.5)))))
        pause_values.append(0.5 * (
            (1.0 - float(reference_score[min(left, frame_count - 1)]))
            + (1.0 - float(reference_score[min(max(right - 1, 0), frame_count - 1)]))
        ))
    interval_valid = float(
        np.all(ends > starts)
        and np.all(np.diff(starts) >= 0)
        and np.all(np.diff(ends) >= 0)
        and float(starts.min()) >= -1e-6
        and float(ends.max()) <= float(sample["audio_meta"].get("duration_sec", 0.0)) + 1e-3
    )
    return {
        "active_fraction": float(np.mean(active_values)),
        "pause_quality": float(np.mean(pause_values)),
        "duration_fit": float(np.mean(duration_values)),
        "interval_valid": interval_valid,
    }

def build_bundle(
    sample: dict[str, Any],
    alignment_name: str,
    sigma: float,
    text_key: str,
    audio_indices: list[int],
    vision_fps: float,
    vision_indices: list[int],
    reference_score: np.ndarray | None = None,
) -> dict[str, Any]:
    audio_smoothed, _ = ext.smooth_contiguous_valid(sample["audio_raw"], sample["audio_valid"], sigma=sigma)
    vision_raw = sample["vision"][float(vision_fps)]["raw"]
    vision_valid = sample["vision"][float(vision_fps)]["valid"]
    vision_times = sample["vision"][float(vision_fps)]["times"]
    vision_smoothed, _ = ext.smooth_contiguous_valid(vision_raw, vision_valid, sigma=sigma)
    starts, ends, alignment_meta = alignment_for_sample(sample, alignment_name, audio_smoothed)
    word_audio, word_audio_valid, _ = ext.aggregate_audio_to_words(audio_smoothed, sample["audio_valid"], starts, ends)
    word_vision, word_vision_valid, _ = ext.aggregate_visual_to_words(vision_smoothed, vision_valid, vision_times, starts, ends)
    if text_key == "bert":
        word_text, text_valid = sample["text_bert"], sample["text_bert_valid"]
    elif text_key == "hash":
        word_text, text_valid = sample["text_hash"], sample["text_hash_valid"]
    else:
        raise KeyError(text_key)
    word_audio = word_audio[:, audio_indices]
    word_vision = word_vision[:, vision_indices]
    if reference_score is None:
        reference_score, _, _, _, _ = ext.estimate_speech_activity(audio_smoothed, sample["audio_valid"], sample["voiced_flag"], working_hz=20.0)
    quality = frame_reference_quality(sample, starts, ends, reference_score)
    x_text = summarize_words(word_text, text_valid)
    x_audio = summarize_words(word_audio, word_audio_valid)
    x_vision = summarize_words(word_vision, word_vision_valid)
    return {
        "x_text": x_text,
        "x_audio": x_audio,
        "x_vision": x_vision,
        "x_fused": np.concatenate([x_text, x_audio, x_vision]),
        "quality": quality,
        "starts": starts,
        "ends": ends,
        "alignment_meta": alignment_meta,
        "word_audio_valid": word_audio_valid,
        "word_vision_valid": word_vision_valid,
    }

def _grouped_fold_indices(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    fold_by_group = {str(group): index % n_splits for index, group in enumerate(unique_groups)}
    fold_ids = np.asarray([fold_by_group[str(group)] for group in groups], dtype=np.int32)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        test_idx = np.flatnonzero(fold_ids == fold)
        train_idx = np.flatnonzero(fold_ids != fold)
        if len(test_idx) and len(train_idx):
            folds.append((train_idx, test_idx))
    return folds

def probe_metrics(
    x: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    groups: np.ndarray,
    repeats: int = 1,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return {
            "probe_accuracy": float("nan"), "probe_accuracy_std": float("nan"),
            "probe_macro_f1": float("nan"), "probe_macro_f1_std": float("nan"),
            "probe_mae": float("nan"), "probe_mae_std": float("nan"),
            "probe_pearson": float("nan"), "probe_pearson_std": float("nan"),
            "probe_repeats": 0, "probe_repeat_macro_f1": [], "probe_repeat_pearson": [],
        }

    repeats = max(1, int(repeats))
    repeat_accuracy: list[float] = []
    repeat_macro_f1: list[float] = []
    repeat_mae: list[float] = []
    repeat_pearson: list[float] = []

    for repeat in range(repeats):
        if repeat == 0:
            folds = list(GroupKFold(n_splits=n_splits).split(x, y_class, groups))
        else:
            folds = _grouped_fold_indices(groups, n_splits, seed=20260924 + repeat)
        class_pred = np.empty(len(y_class), dtype=object)
        reg_pred = np.empty(len(y_reg), dtype=np.float64)
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])
            train_classes = np.unique(y_class[train_idx])
            if len(train_classes) < 2:
                class_pred[test_idx] = np.full(len(test_idx), train_classes[0], dtype=object)
            else:
                clf = LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", random_state=20260924)
                clf.fit(x_train, y_class[train_idx])
                class_pred[test_idx] = clf.predict(x_test)
            reg = Ridge(alpha=10.0, random_state=20260924)
            reg.fit(x_train, y_reg[train_idx])
            reg_pred[test_idx] = reg.predict(x_test)

        valid_pred = np.asarray([item is not None for item in class_pred])
        if np.any(valid_pred):
            accuracy = float(accuracy_score(y_class[valid_pred], class_pred[valid_pred]))
            macro_f1 = float(f1_score(y_class[valid_pred], class_pred[valid_pred], average="macro", zero_division=0))
        else:
            accuracy = macro_f1 = float("nan")
        reg_pred = np.nan_to_num(
            reg_pred,
            nan=float(np.mean(y_reg)),
            posinf=float(np.max(y_reg)),
            neginf=float(np.min(y_reg)),
        )
        mae = float(mean_absolute_error(y_reg, reg_pred))
        if len(y_reg) > 1 and float(np.std(reg_pred)) > 1e-12 and float(np.std(y_reg)) > 1e-12:
            pearson = float(np.corrcoef(y_reg, reg_pred)[0, 1])
        else:
            pearson = float("nan")
        repeat_accuracy.append(accuracy)
        repeat_macro_f1.append(macro_f1)
        repeat_mae.append(mae)
        repeat_pearson.append(pearson)

    def finite_mean(values: list[float]) -> float:
        selected = [value for value in values if np.isfinite(value)]
        return float(np.mean(selected)) if selected else float("nan")

    def finite_std(values: list[float]) -> float:
        selected = [value for value in values if np.isfinite(value)]
        return float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0

    return {
        "probe_accuracy": finite_mean(repeat_accuracy),
        "probe_accuracy_std": finite_std(repeat_accuracy),
        "probe_macro_f1": finite_mean(repeat_macro_f1),
        "probe_macro_f1_std": finite_std(repeat_macro_f1),
        "probe_mae": finite_mean(repeat_mae),
        "probe_mae_std": finite_std(repeat_mae),
        "probe_pearson": finite_mean(repeat_pearson),
        "probe_pearson_std": finite_std(repeat_pearson),
        "probe_repeats": repeats,
        "probe_repeat_macro_f1": [float(value) for value in repeat_macro_f1],
        "probe_repeat_pearson": [float(value) for value in repeat_pearson],
    }

def smoothing_metrics(sample: dict[str, Any], sigma: float, vision_fps: float) -> dict[str, float]:
    audio_raw = sample["audio_raw"]
    audio_valid = np.all(sample["audio_valid"], axis=1)
    vision_raw = sample["vision"][float(vision_fps)]["raw"]
    vision_valid = sample["vision"][float(vision_fps)]["valid"]
    audio_s, _ = ext.smooth_contiguous_valid(audio_raw, sample["audio_valid"], sigma=sigma)
    vision_s, _ = ext.smooth_contiguous_valid(vision_raw, vision_valid, sigma=sigma)

    def roughness(values: np.ndarray, valid: np.ndarray) -> float:
        if len(values) < 3 or not np.any(valid):
            return 0.0
        selected = values[valid]
        if len(selected) < 3:
            return 0.0
        scale = float(np.std(selected, axis=0).mean()) + 1e-8
        return float(np.mean(np.abs(np.diff(selected, n=2, axis=0))) / scale)

    def ratio(smoothed: np.ndarray, raw: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
        if not np.any(valid):
            return 0.0, 1.0
        r = raw[valid]
        s = smoothed[valid]
        scale = float(np.std(r)) + 1e-8
        rmse = float(np.sqrt(np.mean((s - r) ** 2)) / scale)
        raw_peak = float(np.max(np.abs(r), axis=0).mean())
        smooth_peak = float(np.max(np.abs(s), axis=0).mean())
        return rmse, smooth_peak / max(raw_peak, 1e-8)

    a_rmse, a_peak = ratio(audio_s, audio_raw, audio_valid)
    v_rmse, v_peak = ratio(vision_s, vision_raw, vision_valid)
    return {
        "audio_roughness": roughness(audio_s, audio_valid),
        "vision_roughness": roughness(vision_s, vision_valid),
        "audio_rmse_ratio": a_rmse,
        "vision_rmse_ratio": v_rmse,
        "audio_peak_retention": a_peak,
        "vision_peak_retention": v_peak,
    }

def evaluate_candidates(samples: list[dict[str, Any]], fps_values: tuple[float, ...], probe_repeats: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_class = np.asarray([sample["annotation"] for sample in samples], dtype=object)
    y_reg = np.asarray([sample["label"] for sample in samples], dtype=np.float64)
    groups = np.asarray([sample["video_id"] for sample in samples], dtype=object)
    rows: list[dict[str, Any]] = []
    reference_scores: dict[str, np.ndarray] = {}

    def reference(sample: dict[str, Any]) -> np.ndarray:
        key = sample["sample_id"]
        if key not in reference_scores:
            audio_s, _ = ext.smooth_contiguous_valid(sample["audio_raw"], sample["audio_valid"], sigma=0.75)
            score, _, _, _, _ = ext.estimate_speech_activity(audio_s, sample["audio_valid"], sample["voiced_flag"], working_hz=20.0)
            reference_scores[key] = score
        return reference_scores[key]

    def bundles_for(
        alignment_name: str,
        sigma: float,
        text_key: str,
        audio_indices: list[int],
        vision_fps: float,
        vision_indices: list[int],
    ) -> list[dict[str, Any]]:
        built = []
        for sample in samples:
            built.append(build_bundle(
                sample, alignment_name, sigma, text_key, audio_indices, vision_fps, vision_indices, reference(sample)
            ))
        return built

    def record(
        family: str,
        method: str,
        status: str,
        bundles: list[dict[str, Any]] | None = None,
        elapsed: float = 0.0,
        error: str = "",
        extra: dict[str, Any] | None = None,
        probe_component: str = "fused",
    ) -> None:
        row: dict[str, Any] = {
            "family": family,
            "method": method,
            "status": status,
            "n_samples": len(samples),
            "elapsed_sec": round(float(elapsed), 6),
            "error": error,
        }
        if bundles is not None:
            quality_rows = [item["quality"] for item in bundles]
            row["mean_active_fraction"] = float(np.mean([item["active_fraction"] for item in quality_rows]))
            row["mean_pause_quality"] = float(np.mean([item["pause_quality"] for item in quality_rows]))
            row["mean_duration_fit"] = float(np.mean([item["duration_fit"] for item in quality_rows]))
            row["interval_valid_ratio"] = float(np.mean([item["interval_valid"] for item in quality_rows]))
            row["mean_audio_observed_ratio"] = float(np.mean([np.mean(item["word_audio_valid"]) if len(item["word_audio_valid"]) else 0.0 for item in bundles]))
            row["mean_vision_observed_ratio"] = float(np.mean([np.mean(item["word_vision_valid"]) if len(item["word_vision_valid"]) else 0.0 for item in bundles]))
            row["finite_ratio"] = float(np.mean([
                bool(np.all(np.isfinite(item["x_fused"])) and np.all(np.isfinite(item["starts"])) and np.all(np.isfinite(item["ends"])))
                for item in bundles
            ]))
            probe_x = np.asarray([item[f"x_{probe_component}"] for item in bundles])
            row.update(probe_metrics(probe_x, y_class, y_reg, groups, repeats=probe_repeats))
            row["probe_component"] = probe_component
            row["feature_dim_probe"] = int(probe_x.shape[1]) if probe_x.ndim == 2 else 0
            row["feature_dim_fused"] = int(np.asarray(bundles[0]["x_fused"]).shape[0])
        if extra:
            row.update(extra)
        rows.append(row)

    for name in ALIGNMENT_NAMES:
        start = time.perf_counter()
        try:
            bundles = bundles_for(name, 0.75, "bert", AUDIO_SETS["full_40"], 5.0, VISION_SETS["full_90"])
            record("alignment", name, "ok", bundles, time.perf_counter() - start)
        except Exception as exc:
            record("alignment", name, "error", None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")

    fixed_alignments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        audio_s, _ = ext.smooth_contiguous_valid(sample["audio_raw"], sample["audio_valid"], sigma=0.75)
        starts, ends, _ = alignment_for_sample(sample, "full_dp_20", audio_s)
        fixed_alignments[sample["sample_id"]] = (starts, ends)
    for sigma in SIGMA_VALUES:
        start = time.perf_counter()
        try:
            bundles = []
            for sample in samples:
                starts, ends = fixed_alignments[sample["sample_id"]]
                audio_s, _ = ext.smooth_contiguous_valid(sample["audio_raw"], sample["audio_valid"], sigma=sigma)
                vision_raw = sample["vision"][5.0]["raw"]
                vision_valid = sample["vision"][5.0]["valid"]
                vision_times = sample["vision"][5.0]["times"]
                vision_s, _ = ext.smooth_contiguous_valid(vision_raw, vision_valid, sigma=sigma)
                wa, wav, _ = ext.aggregate_audio_to_words(audio_s, sample["audio_valid"], starts, ends)
                wv, wvv, _ = ext.aggregate_visual_to_words(vision_s, vision_valid, vision_times, starts, ends)
                x_text = summarize_words(sample["text_bert"], sample["text_bert_valid"])
                x_audio = summarize_words(wa, wav)
                x_vision = summarize_words(wv, wvv)
                bundles.append({
                    "x_text": x_text,
                    "x_audio": x_audio,
                    "x_vision": x_vision,
                    "x_fused": np.concatenate([x_text, x_audio, x_vision]),
                    "quality": frame_reference_quality(sample, starts, ends, reference(sample)),
                    "starts": starts,
                    "ends": ends,
                    "word_audio_valid": wav,
                    "word_vision_valid": wvv,
                })
            smoothing_extras: dict[str, Any] = {}
            metric_rows = [smoothing_metrics(sample, sigma, 5.0) for sample in samples]
            for key in ("audio_roughness", "vision_roughness", "audio_rmse_ratio", "vision_rmse_ratio", "audio_peak_retention", "vision_peak_retention"):
                smoothing_extras[key] = float(np.mean([item[key] for item in metric_rows]))
            record("smoothing", f"sigma_{safe_tag(sigma)}", "ok", bundles, time.perf_counter() - start, extra=smoothing_extras)
        except Exception as exc:
            record("smoothing", f"sigma_{safe_tag(sigma)}", "error", None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")

    for text_key in TEXT_KEYS:
        start = time.perf_counter()
        try:
            bundles = bundles_for("full_dp_20", 0.75, text_key, AUDIO_SETS["full_40"], 5.0, VISION_SETS["full_90"])
            record("text", text_key, "ok", bundles, time.perf_counter() - start, extra={"feature_dim_text": 768}, probe_component="text")
        except Exception as exc:
            record("text", text_key, "error", None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")

    for name, indices in AUDIO_SETS.items():
        start = time.perf_counter()
        try:
            bundles = bundles_for("full_dp_20", 0.75, "bert", indices, 5.0, VISION_SETS["full_90"])
            record("audio_features", name, "ok", bundles, time.perf_counter() - start, extra={"feature_dim_audio": len(indices)}, probe_component="audio")
        except Exception as exc:
            record("audio_features", name, "error", None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")

    for name, indices in VISION_SETS.items():
        start = time.perf_counter()
        try:
            bundles = bundles_for("full_dp_20", 0.75, "bert", AUDIO_SETS["full_40"], 5.0, indices)
            record("vision_features", name, "ok", bundles, time.perf_counter() - start, extra={"feature_dim_vision": len(indices)}, probe_component="vision")
        except Exception as exc:
            record("vision_features", name, "error", None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")

    for fps in fps_values:
        start = time.perf_counter()
        try:
            bundles = bundles_for("full_dp_20", 0.75, "bert", AUDIO_SETS["full_40"], fps, VISION_SETS["full_90"])
            record("vision_fps", f"fps_{safe_tag(fps)}", "ok", bundles, time.perf_counter() - start, extra={"vision_fps": float(fps)}, probe_component="vision")
        except Exception as exc:
            record("vision_fps", f"fps_{safe_tag(fps)}", "error", None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    recommendations: dict[str, Any] = {}
    for family, candidates in by_family.items():
        valid = [
            row for row in candidates
            if row["status"] == "ok"
            and row.get("interval_valid_ratio", 0.0) >= 1.0 - 1e-12
            and row.get("finite_ratio", 0.0) >= 1.0 - 1e-12
        ]
        if not valid:
            recommendations[family] = {"selected": None, "reason": "all candidates failed"}
            continue
        macro = np.asarray([row.get("probe_macro_f1", np.nan) for row in valid], dtype=float)
        pearson = np.asarray([row.get("probe_pearson", np.nan) for row in valid], dtype=float)
        duration_fit = np.asarray([row.get("mean_duration_fit", 0.0) for row in valid], dtype=float)
        active = np.asarray([row.get("mean_active_fraction", 0.0) for row in valid], dtype=float)
        dims = np.asarray([row.get("feature_dim_probe", row.get("feature_dim_fused", 1)) for row in valid], dtype=float)

        macro = np.clip(np.nan_to_num(macro, nan=0.0), 0.0, 1.0)
        pearson = np.clip(np.nan_to_num(pearson, nan=0.0), 0.0, 1.0)
        duration_fit = np.clip(np.nan_to_num(duration_fit, nan=0.0), 0.0, 1.0)
        active = np.clip(np.nan_to_num(active, nan=0.0), 0.0, 1.0)
        dim_penalty = dims / max(float(np.max(dims)), 1.0)
        composite = 0.45 * macro + 0.25 * pearson + 0.15 * duration_fit + 0.10 * active - 0.05 * dim_penalty
        order = np.argsort(-composite)
        selected = valid[int(order[0])]
        recommendations[family] = {
            "selected": selected["method"],
            "composite_score": float(composite[order[0]]),
            "ranking": [
                {"method": valid[int(i)]["method"], "composite_score": float(composite[int(i)])}
                for i in order
            ],
            "rule": "hard constraints first; 0.45*macro_f1 + 0.25*max(pearson,0) + 0.15*duration_fit + 0.10*active - 0.05*(dim/max_dim); repeated grouped probe is selection-only",
        }
    summary = {
        "n_samples": len(samples),
        "families": {family: candidates for family, candidates in by_family.items()},
        "recommendations": recommendations,
        "notes": [
            "probe_* uses repeated grouped cross-validation by video_id and is not a formal problem-1 model.",
            "No external model or dependency was downloaded or installed.",
            "Final choice also needs reproducibility, storage and manuscript interpretability checks.",
        ],
    }
    return rows, summary

def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# 问题一方法对比实验报告",
        "",
        f"- 样本数：{summary['n_samples']}",
        "- 诊断探针：按 `video_id` 重复分组交叉验证，指标为跨重复均值，仅用于方法选型，不是问题一正式模型。",
        "- 硬约束：区间合法、严格单调、非重叠、有限值、样本覆盖。",
        "",
        "## 推荐结果",
        "",
    ]
    for family, rec in summary["recommendations"].items():
        lines.append(f"- `{family}`：推荐 `{rec.get('selected')}`（综合分 {rec.get('composite_score')}）。")
    lines.extend(["", "## 指标表", "", "| family | method | probe | status | macro-F1 mean±sd | Pearson mean±sd | MAE | active | pause | duration fit | interval valid | dim | elapsed s |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        values = {
            **row,
            "probe_component": row.get("probe_component", "fused"),
            "probe_macro_f1": row.get("probe_macro_f1", float("nan")),
            "probe_macro_f1_std": row.get("probe_macro_f1_std", 0.0),
            "probe_pearson": row.get("probe_pearson", float("nan")),
            "probe_pearson_std": row.get("probe_pearson_std", 0.0),
            "probe_mae": row.get("probe_mae", float("nan")),
            "mean_active_fraction": row.get("mean_active_fraction", 0.0),
            "mean_pause_quality": row.get("mean_pause_quality", 0.0),
            "mean_duration_fit": row.get("mean_duration_fit", 0.0),
            "interval_valid_ratio": row.get("interval_valid_ratio", 0.0),
            "feature_dim_probe": row.get("feature_dim_probe", row.get("feature_dim_fused", 0)),
            "elapsed_sec": row.get("elapsed_sec", 0.0),
        }
        lines.append(
            "| {family} | {method} | {probe_component} | {status} | {probe_macro_f1:.4f}±{probe_macro_f1_std:.4f} | {probe_pearson:.4f}±{probe_pearson_std:.4f} | {probe_mae:.4f} | {mean_active_fraction:.4f} | {mean_pause_quality:.4f} | {mean_duration_fit:.4f} | {interval_valid_ratio:.4f} | {feature_dim_probe} | {elapsed_sec:.3f} |".format(**values)
        )
    lines.extend(["", "## 边界说明", "", "- 未使用附件3、附件4。", "- 未安装新依赖，未执行问题二训练。", "- 无人工词边界真值，不能把无监督代理指标解释为真实词边界误差。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args)
    if not rows:
        raise SystemExit("no rows selected")
    samples = build_cache(rows, args, DEFAULT_FPS)
    if args.cache_only:
        print(json.dumps({"cache_dir": str(args.cache_dir), "n_samples": len(samples), "cache_only": True}, ensure_ascii=False, indent=2))
        return 0
    metrics, summary = evaluate_candidates(samples, DEFAULT_FPS, probe_repeats=args.probe_repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({key for row in metrics for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    json_dump(args.output_dir / "comparison_summary.json", summary)
    write_report(args.output_dir / "comparison_report.md", metrics, summary)
    print(json.dumps({"output_dir": str(args.output_dir), "n_samples": len(samples), "recommendations": summary["recommendations"]}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
