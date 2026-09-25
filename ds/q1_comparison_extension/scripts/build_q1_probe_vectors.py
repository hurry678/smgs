#!/usr/bin/env python3
"""Build leakage-free sample-level probe vectors for Q1 candidates A, B and C."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-q1-dir", type=Path, required=True)
    parser.add_argument("--git-result-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(values: np.ndarray, valid: np.ndarray, item_weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    item_weights = np.asarray(item_weights, dtype=np.float64)
    if valid.ndim == 1:
        valid = np.repeat(valid[:, None], values.shape[1], axis=1)
    if values.shape != valid.shape or values.ndim != 2 or item_weights.shape != (len(values),):
        raise ValueError(f"Invalid value/mask/weight shapes: {values.shape}, {valid.shape}, {item_weights.shape}")
    if not np.isfinite(item_weights).all() or np.any(item_weights <= 0):
        raise ValueError("Summary weights must be finite and positive")
    weights = valid * item_weights[:, None]
    count = weights.sum(axis=0)
    total = (np.where(valid, values, 0.0) * item_weights[:, None]).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros(values.shape[1], dtype=np.float64), where=count > 0)
    centered = np.where(valid, values - mean, 0.0)
    variance = np.divide(
        (centered * centered * item_weights[:, None]).sum(axis=0),
        count,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=count > 0,
    )
    observed = float(weights.sum() / (item_weights.sum() * values.shape[1])) if valid.size else 0.0
    vector = np.concatenate([mean, np.sqrt(variance), np.asarray([observed])]).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite sample summary")
    return vector, observed


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["sample_id"]: row for row in rows}
    if len(result) != 100:
        raise ValueError(f"Expected 100 unique labels, got {len(result)}")
    return result


def main() -> int:
    args = parse_args()
    current = args.current_q1_dir.resolve()
    git_result = args.git_result_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels_csv)
    current_meta = {
        row["sample_id"]: row
        for row in map(json.loads, (current / "outputs/alignment.jsonl").read_text(encoding="utf-8").splitlines())
    }
    git_meta = {
        row["sample_id"]: row
        for row in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((git_result / "metadata").glob("*.json"))
        )
    }
    if not (set(labels) == set(current_meta) == set(git_meta)):
        raise ValueError("Candidate and label ID sets differ")

    records: dict[str, list[dict[str, Any]]] = {
        "A_current_audited": [],
        "B_git_frozen": [],
        "C_hybrid_capacity": [],
    }
    for sample_id in sorted(labels):
        label = labels[sample_id]
        local_meta = current_meta[sample_id]
        remote_meta = git_meta[sample_id]
        remote_text = remote_meta.get("raw_text") or remote_meta.get("text", {}).get("raw_text", "")
        if local_meta["text"] != remote_text:
            raise ValueError(f"Transcript mismatch: {sample_id}")
        if local_meta["fingerprint_components"]["source_sha256"].lower() != remote_meta["source_sha256"].lower():
            raise ValueError(f"Source mismatch: {sample_id}")

        with np.load(current / local_meta["feature_file"], allow_pickle=False) as data:
            local_features = np.asarray(data["features"], dtype=np.float32)
            local_valid = np.asarray(data["valid"], dtype=bool)
        with np.load(git_result / "features" / f"{sample_id.replace('$_$', '__')}.npz", allow_pickle=False) as data:
            git_text = np.asarray(data["text_features"], dtype=np.float32)
            git_text_valid = np.asarray(data["text_valid"], dtype=bool)
            git_audio = np.asarray(data["word_audio_features"], dtype=np.float32)
            git_audio_valid = np.asarray(data["word_audio_valid"], dtype=bool)
            git_vision = np.asarray(data["word_vision_features"], dtype=np.float32)
            git_vision_valid = np.asarray(data["word_vision_valid"], dtype=bool)
            git_item_weights = np.asarray(data["word_end_sec"] - data["word_start_sec"], dtype=np.float64)

        local_time_weights = np.asarray(
            [phrase["end_s"] - phrase["start_s"] for phrase in local_meta["phrases"]], dtype=np.float64
        )
        local_text_weights = np.asarray(
            [len(phrase["token_indices"]) for phrase in local_meta["phrases"]], dtype=np.float64
        )
        local_text_vector, local_text_rate = summarize(
            local_features[:, :128], local_valid[:, :128], local_text_weights
        )
        local_audio_vector, local_audio_rate = summarize(
            local_features[:, 128:144], local_valid[:, 128:144], local_time_weights
        )
        local_vision_vector, local_vision_rate = summarize(
            local_features[:, 144:150], local_valid[:, 144:150], local_time_weights
        )
        git_text_vector, git_text_rate = summarize(git_text, git_text_valid, git_item_weights)
        git_audio_vector, git_audio_rate = summarize(git_audio, git_audio_valid, git_item_weights)
        git_vision_vector, git_vision_rate = summarize(git_vision, git_vision_valid, git_item_weights)

        common = {
            "sample_id": sample_id,
            "video_id": label["video_id"],
            "y_class": label["annotation"],
            "y_reg": float(label["label"]),
            "source_sha256": remote_meta["source_sha256"],
        }
        records["A_current_audited"].append(common | {
            "x_text": local_text_vector, "x_audio": local_audio_vector, "x_vision": local_vision_vector,
            "valid_rates": [local_text_rate, local_audio_rate, local_vision_rate],
        })
        records["B_git_frozen"].append(common | {
            "x_text": git_text_vector, "x_audio": git_audio_vector, "x_vision": git_vision_vector,
            "valid_rates": [git_text_rate, git_audio_rate, git_vision_rate],
        })
        records["C_hybrid_capacity"].append(common | {
            "x_text": git_text_vector, "x_audio": local_audio_vector, "x_vision": local_vision_vector,
            "valid_rates": [git_text_rate, local_audio_rate, local_vision_rate],
        })

    manifest = {"schema": "q1-probe-vectors-1", "candidates": {}}
    for candidate, rows in records.items():
        target = args.output_dir / f"{candidate}.npz"
        np.savez_compressed(
            target,
            sample_id=np.asarray([row["sample_id"] for row in rows]),
            video_id=np.asarray([row["video_id"] for row in rows]),
            y_class=np.asarray([row["y_class"] for row in rows]),
            y_reg=np.asarray([row["y_reg"] for row in rows], dtype=np.float32),
            source_sha256=np.asarray([row["source_sha256"] for row in rows]),
            x_text=np.stack([row["x_text"] for row in rows]),
            x_audio=np.stack([row["x_audio"] for row in rows]),
            x_vision=np.stack([row["x_vision"] for row in rows]),
            valid_rates=np.asarray([row["valid_rates"] for row in rows], dtype=np.float32),
        )
        manifest["candidates"][candidate] = {
            "file": target.name,
            "samples": len(rows),
            "dimensions": {
                "text": len(rows[0]["x_text"]),
                "audio": len(rows[0]["x_audio"]),
                "vision": len(rows[0]["x_vision"]),
            },
        }
    (args.output_dir / "candidate_vector_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
