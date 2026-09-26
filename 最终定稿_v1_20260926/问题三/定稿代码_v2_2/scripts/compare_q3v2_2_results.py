#!/usr/bin/env python3
"""Compare v2 and v2.2 outputs while enforcing predictor invariance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "vision")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = {
        str(sample["sample_id"]): sample for sample in payload.get("samples", [])
    }
    if len(samples) != len(payload.get("samples", [])):
        raise ValueError(f"duplicate sample IDs in {path}")
    return payload["report"], samples


def modality_counts(
    samples: Mapping[str, Mapping[str, Any]], key: str
) -> dict[str, int]:
    counts = Counter(str(sample.get(key)) for sample in samples.values())
    return {modality: int(counts.get(modality, 0)) for modality in MODALITIES}


def evidence_counts(
    samples: Mapping[str, Mapping[str, Any]], key: str
) -> dict[str, int]:
    return {
        modality: int(
            sum(len(sample[key].get(modality, [])) for sample in samples.values())
        )
        for modality in MODALITIES
    }


def max_vector_difference(old: Sequence[float], new: Sequence[float]) -> float:
    old_array = np.asarray(old, dtype=np.float64)
    new_array = np.asarray(new, dtype=np.float64)
    if old_array.shape != new_array.shape:
        return float("inf")
    return float(np.max(np.abs(old_array - new_array))) if old_array.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-raw", type=Path, required=True)
    parser.add_argument("--new-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    old_report, old = load_samples(args.old_raw)
    new_report, new = load_samples(args.new_raw)
    if set(old) != set(new):
        raise SystemExit("v2 and v2.2 sample ID sets differ")

    ids = sorted(new)
    polarity_changes = [
        sample_id
        for sample_id in ids
        if old[sample_id]["polarity"] != new[sample_id]["polarity"]
    ]
    intensity_differences = [
        abs(float(old[sample_id]["intensity"]) - float(new[sample_id]["intensity"]))
        for sample_id in ids
    ]
    logit_differences = [
        max_vector_difference(
            old[sample_id]["log_mean_probability"],
            new[sample_id]["log_mean_probability"],
        )
        for sample_id in ids
    ]
    primary_changes = [
        {
            "sample_id": sample_id,
            "old": old[sample_id]["primary_modality"],
            "new": new[sample_id]["primary_modality"],
        }
        for sample_id in ids
        if old[sample_id]["primary_modality"] != new[sample_id]["primary_modality"]
    ]
    evidence_changed = [
        sample_id
        for sample_id in ids
        if old[sample_id]["key_evidence"] != new[sample_id]["key_evidence"]
        or old[sample_id]["counter_evidence"] != new[sample_id]["counter_evidence"]
    ]
    prediction_ok = (
        not polarity_changes
        and max(intensity_differences, default=0.0) <= args.prediction_tolerance
        and max(logit_differences, default=0.0) <= args.prediction_tolerance
    )
    result = {
        "schema": "q3v2.2-result-comparison-v1",
        "status": "PASS" if prediction_ok else "FAIL",
        "old_raw_sha256": sha256_file(args.old_raw),
        "new_raw_sha256": sha256_file(args.new_raw),
        "sample_count": len(ids),
        "old_window_size": old_report.get("protocol", {}).get("window_size"),
        "new_window_size": new_report.get("protocol", {}).get("window_size"),
        "predictor_invariance": {
            "polarity_change_count": len(polarity_changes),
            "polarity_change_ids": polarity_changes,
            "maximum_intensity_difference": max(intensity_differences, default=0.0),
            "maximum_log_probability_difference": max(logit_differences, default=0.0),
            "tolerance": float(args.prediction_tolerance),
        },
        "explanation_changes": {
            "primary_modality_change_count": len(primary_changes),
            "primary_modality_changes": primary_changes,
            "evidence_change_count": len(evidence_changed),
            "evidence_change_ids": evidence_changed,
            "old_primary_counts": modality_counts(old, "primary_modality"),
            "new_primary_counts": modality_counts(new, "primary_modality"),
            "old_key_evidence_counts": evidence_counts(old, "key_evidence"),
            "new_key_evidence_counts": evidence_counts(new, "key_evidence"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not prediction_ok:
        raise SystemExit("v2.2 unexpectedly changed the frozen predictor outputs")


if __name__ == "__main__":
    main()
