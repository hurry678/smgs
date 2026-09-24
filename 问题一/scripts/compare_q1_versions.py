#!/usr/bin/env python3
"""Compare current MFA phrase boundaries with frozen Git DP boundaries on shared evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-q1-dir", type=Path, required=True)
    parser.add_argument("--git-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "p90": float(np.quantile(array, 0.9)) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
        "fraction_le_0_25": float(np.mean(array <= 0.25)) if array.size else None,
        "fraction_le_0_50": float(np.mean(array <= 0.50)) if array.size else None,
    }


def main() -> int:
    args = parse_args()
    current = args.current_q1_dir.resolve()
    git_result = args.git_result_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_rows = {
        row["sample_id"]: row
        for row in map(json.loads, (current / "outputs/alignment.jsonl").read_text(encoding="utf-8").splitlines())
    }
    audit_data = load(current / "reports/boundary_audit.json")
    audits = {row["sample_id"]: row for row in audit_data["samples"]}
    git_meta = {
        row["sample_id"]: row
        for row in map(load, sorted((git_result / "metadata").glob("*.json")))
    }
    if not (set(current_rows) == set(audits) == set(git_meta)) or len(current_rows) != 100:
        raise ValueError("The two versions and independent audit must cover the same 100 sample IDs")

    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []
    for sample_id in sorted(current_rows):
        current_row = current_rows[sample_id]
        git_row = git_meta[sample_id]
        git_text = git_row.get("raw_text") or git_row.get("text", {}).get("raw_text", "")
        source_equal = current_row["fingerprint_components"]["source_sha256"].lower() == git_row["source_sha256"].lower()
        text_equal = current_row["text"] == git_text
        counts["source_hash_match"] += int(source_equal)
        counts["text_match"] += int(text_equal)
        if not source_equal or not text_equal:
            raise ValueError(f"Input mismatch: {sample_id}")

        git_npz = git_result / "features" / f"{sample_id.replace('$_$', '__')}.npz"
        with np.load(git_npz, allow_pickle=False) as data:
            git_start = np.asarray(data["word_start_sec"], dtype=float)
            git_end = np.asarray(data["word_end_sec"], dtype=float)
            git_fallback = np.asarray(data["word_fallback"], dtype=bool)
            git_audio_valid = np.asarray(data["word_audio_valid"], dtype=bool)
        spans = list(TOKEN.finditer(git_text))
        if len(spans) != len(git_start):
            raise ValueError(f"Git token/time mismatch: {sample_id}")

        local_differences: list[float] = []
        git_differences: list[float] = []
        for phrase, audit_phrase in zip(current_row["phrases"], audits[sample_id]["phrases"]):
            if audit_phrase["asr_s"] is None:
                continue
            indices = [
                index for index, token in enumerate(spans)
                if token.start() >= phrase["char_start"] and token.end() <= phrase["char_end"]
            ]
            if not indices:
                continue
            asr_start, asr_end = map(float, audit_phrase["asr_s"])
            values = [
                ("start", float(phrase["start_s"]), float(git_start[indices[0]]), asr_start),
                ("end", float(phrase["end_s"]), float(git_end[indices[-1]]), asr_end),
            ]
            for endpoint, local_value, git_value, reference in values:
                local_difference = abs(local_value - reference)
                git_difference = abs(git_value - reference)
                local_differences.append(local_difference)
                git_differences.append(git_difference)
                output_rows.append({
                    "sample_id": sample_id,
                    "video_id": current_row["video_id"],
                    "phrase_index": phrase["phrase_index"],
                    "endpoint": endpoint,
                    "text": phrase["text"],
                    "asr_s": reference,
                    "current_s": local_value,
                    "git_s": git_value,
                    "current_abs_difference_s": local_difference,
                    "git_abs_difference_s": git_difference,
                })
        per_sample.append({
            "sample_id": sample_id,
            "shared_endpoints": len(local_differences),
            "current_mean_difference_s": float(np.mean(local_differences)) if local_differences else None,
            "git_mean_difference_s": float(np.mean(git_differences)) if git_differences else None,
            "current_speech_alignment_available": current_row["speech_alignment_available"],
            "current_digitally_silent": current_row["digitally_silent_audio"],
            "git_words": len(git_start),
            "git_fallback_words": int(git_fallback.sum()),
            "git_audio_valid_words": int(git_audio_valid.sum()),
        })

    local_values = [row["current_abs_difference_s"] for row in output_rows]
    git_values = [row["git_abs_difference_s"] for row in output_rows]
    local_array, git_array = np.asarray(local_values), np.asarray(git_values)
    comparable_samples = [row for row in per_sample if row["shared_endpoints"]]
    groups = sorted({row["video_id"] for row in output_rows})
    grouped = [
        [row["git_abs_difference_s"] - row["current_abs_difference_s"]
         for row in output_rows if row["video_id"] == group]
        for group in groups
    ]
    sums = np.asarray([sum(values) for values in grouped])
    sizes = np.asarray([len(values) for values in grouped])
    rng = np.random.default_rng(20260924)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    effects = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    summary = {
        "comparison_schema": "q1-version-comparison-1",
        "source_hash_match_count": counts["source_hash_match"],
        "text_match_count": counts["text_match"],
        "shared_phrase_count": len(output_rows) // 2,
        "shared_endpoint_count": len(output_rows),
        "shared_sample_count": len(comparable_samples),
        "shared_video_group_count": len(groups),
        "current_asr_endpoint_difference_s": distribution(local_values),
        "git_asr_endpoint_difference_s": distribution(git_values),
        "current_endpoint_wins": int(np.sum(local_array < git_array - 1e-6)),
        "ties": int(np.sum(np.abs(local_array - git_array) <= 1e-6)),
        "git_endpoint_wins": int(np.sum(git_array < local_array - 1e-6)),
        "git_minus_current_mean_difference_s": float(np.mean(git_array - local_array)),
        "video_cluster_bootstrap_95pct": np.quantile(effects, [0.025, 0.975]).tolist(),
        "human_ground_truth_used": False,
        "interpretation": "Independent audio-only ASR model agreement on an exact-lexical-match subset; not human boundary accuracy.",
        "special_cases": [row for row in per_sample if row["current_digitally_silent"] or not row["current_speech_alignment_available"]],
    }
    with (args.output_dir / "version_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    with (args.output_dir / "version_sample_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    (args.output_dir / "version_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
