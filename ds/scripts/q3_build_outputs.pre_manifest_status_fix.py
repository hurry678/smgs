#!/usr/bin/env python3
"""Build Q3 submission CSV/JSONL and a reproducible freeze manifest.

The script does not predict or explain.  It only packages an already frozen
Q3 explanation run, validates its schema and mapping fields, and records
hashes for reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MODALITIES = ("text", "audio", "vision")
POLARITIES = {"Negative", "Neutral", "Positive"}
CST = timezone(timedelta(hours=8))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_raw(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "samples" not in obj or "report" not in obj:
        raise ValueError("raw Q3 output must contain report and samples")
    return obj


def validate_sample(sample: Dict[str, Any], raw_text: str, sample_id: str) -> List[str]:
    errors: List[str] = []
    if str(sample.get("sample_id")) != str(sample_id):
        errors.append(f"{sample_id}: sample_id mismatch")
    if sample.get("polarity") not in POLARITIES:
        errors.append(f"{sample_id}: invalid polarity {sample.get('polarity')!r}")
    intensity = float(sample.get("intensity", float("nan")))
    if not math.isfinite(intensity) or not (-3.0 - 1e-6 <= intensity <= 3.0 + 1e-6):
        errors.append(f"{sample_id}: invalid intensity {intensity!r}")
    for key in ("primary_modality", "strongest_modality"):
        if sample.get(key) not in MODALITIES:
            errors.append(f"{sample_id}: invalid {key}={sample.get(key)!r}")
    if sample.get("support_modality") not in (*MODALITIES, None):
        errors.append(f"{sample_id}: invalid support_modality={sample.get('support_modality')!r}")
    for key in ("class_shapley_sum_error", "intensity_shapley_sum_error"):
        value = float(sample.get(key, float("nan")))
        if not math.isfinite(value) or value > 1e-5:
            errors.append(f"{sample_id}: {key} too large ({value!r})")

    key_evidence = sample.get("key_evidence")
    counter_evidence = sample.get("counter_evidence")
    if not isinstance(key_evidence, dict) or not isinstance(counter_evidence, dict):
        errors.append(f"{sample_id}: missing key_evidence/counter_evidence")
        return errors
    for modality in MODALITIES:
        rows = key_evidence.get(modality, [])
        if not isinstance(rows, list):
            errors.append(f"{sample_id}: {modality} key_evidence is not a list")
            continue
        for row in rows:
            if row.get("evidence_direction") != "support":
                errors.append(f"{sample_id}: {modality} key evidence is not support")
            if float(row.get("class_delta", 0.0)) <= 0.0:
                errors.append(f"{sample_id}: {modality} support evidence has non-positive class_delta")
            if modality == "text" and row.get("mapping_status") == "exact":
                a = row.get("text_char_start")
                b = row.get("text_char_end")
                if not isinstance(a, int) or not isinstance(b, int) or not (0 <= a < b <= len(raw_text)):
                    errors.append(f"{sample_id}: invalid text span {(a, b)!r}")
                elif row.get("text_snippet") != raw_text[a:b]:
                    errors.append(f"{sample_id}: text snippet does not match raw text span")
            elif modality in ("audio", "vision") and row.get("mapping_status") == "approximate":
                a = float(row.get("time_start_seconds", -1.0))
                b = float(row.get("time_end_seconds", -1.0))
                if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a <= b):
                    errors.append(f"{sample_id}: invalid approximate time {(a, b)!r}")
                if "approximate" not in str(row.get("warning", "")).lower() and "proportional" not in str(row.get("warning", "")).lower():
                    errors.append(f"{sample_id}: approximate mapping lacks warning")
            elif modality in ("audio", "vision") and row.get("mapping_status") == "relative_only":
                a = float(row.get("relative_start", -1.0))
                b = float(row.get("relative_end", -1.0))
                if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a <= b <= 1.0):
                    errors.append(f"{sample_id}: invalid relative interval {(a, b)!r}")
        counter_rows = counter_evidence.get(modality, [])
        if not isinstance(counter_rows, list):
            errors.append(f"{sample_id}: {modality} counter_evidence is not a list")
            continue
        for row in counter_rows:
            if row.get("evidence_direction") != "counter":
                errors.append(f"{sample_id}: {modality} counter evidence has wrong direction")
            if float(row.get("class_delta", 0.0)) >= 0.0:
                errors.append(f"{sample_id}: {modality} counter evidence has non-negative class_delta")

    has_support = any(key_evidence.get(m) for m in MODALITIES)
    reason = sample.get("no_evidence_reason")
    if has_support and reason is not None:
        errors.append(f"{sample_id}: has support evidence but non-null no_evidence_reason")
    if not has_support and not isinstance(reason, str):
        errors.append(f"{sample_id}: no support evidence but missing no_evidence_reason")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--attachment4", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--explain-script", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--offsets", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--validation-report", type=Path, required=True)
    args = ap.parse_args()

    raw = load_raw(args.raw_json)
    samples = raw["samples"]
    report = raw["report"]
    with np.load(args.attachment4, allow_pickle=False) as z:
        expected_ids = [str(x) for x in z["ids"].tolist()]
        raw_texts = [str(x) for x in z["raw_text"].tolist()]
    if len(samples) != len(expected_ids):
        raise SystemExit(f"sample count mismatch: {len(samples)} vs {len(expected_ids)}")
    if len(set(expected_ids)) != len(expected_ids):
        raise SystemExit("attachment4 contains duplicate sample IDs")
    by_id = {str(s["sample_id"]): s for s in samples}
    if set(by_id) != set(expected_ids):
        raise SystemExit("raw output sample IDs do not exactly cover attachment4")

    errors: List[str] = []
    for sid, text in zip(expected_ids, raw_texts):
        errors.extend(validate_sample(by_id[sid], text, sid))
    if errors:
        raise SystemExit("Q3 output validation failed:\n" + "\n".join(errors[:50]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "q3_predictions_explanations.csv"
    jsonl_path = args.out_dir / "q3_evidence.jsonl"
    validation_path = args.out_dir / "q3_validation_report.json"
    manifest_path = args.out_dir / "q3_freeze_manifest.json"

    csv_fields = [
        "sample_id", "polarity", "intensity", "raw_intensity",
        "primary_modality", "strongest_modality", "support_modality",
        "modality_effect_text", "modality_effect_audio", "modality_effect_vision",
        "class_contrib_text", "class_contrib_audio", "class_contrib_vision",
        "intensity_contrib_text", "intensity_contrib_audio", "intensity_contrib_vision",
        "class_abs_share_text", "class_abs_share_audio", "class_abs_share_vision",
        "intensity_abs_share_text", "intensity_abs_share_audio", "intensity_abs_share_vision",
        "key_evidence_text_json", "key_evidence_audio_json", "key_evidence_vision_json",
        "counter_evidence_text_json", "counter_evidence_audio_json", "counter_evidence_vision_json",
        "no_evidence_reason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for sid in expected_ids:
            s = by_id[sid]
            row: Dict[str, Any] = {k: s.get(k) for k in csv_fields if k in s}
            for modality in MODALITIES:
                row[f"key_evidence_{modality}_json"] = json_compact(s["key_evidence"].get(modality, []))
                row[f"counter_evidence_{modality}_json"] = json_compact(s["counter_evidence"].get(modality, []))
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8", newline="") as f:
        for sid in expected_ids:
            f.write(json.dumps(by_id[sid], ensure_ascii=False, separators=(",", ":")) + "\n")

    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "question": "问题三",
        "status": "frozen_after_validation_before_attachment4",
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "model_manifest": str(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "explanation_script": str(args.explain_script),
        "explanation_script_sha256": sha256_file(args.explain_script),
        "inputs": {
            "data": str(args.data),
            "data_sha256": sha256_file(args.data),
            "offsets": str(args.offsets),
            "offsets_sha256": sha256_file(args.offsets),
            "audit": str(args.audit),
            "audit_sha256": sha256_file(args.audit),
            "attachment4": str(args.attachment4),
            "attachment4_sha256": sha256_file(args.attachment4),
        },
        "frozen_parameters": report.get("protocol", {}),
        "validation_report": str(args.validation_report),
        "validation_report_sha256": sha256_file(args.validation_report),
        "outputs": {
            "raw_json": str(args.raw_json),
            "raw_json_sha256": sha256_file(args.raw_json),
            "csv": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "jsonl": str(jsonl_path),
            "jsonl_sha256": sha256_file(jsonl_path),
            "validation_report_copy": str(validation_path),
            "validation_report_copy_sha256": sha256_file(validation_path),
        },
        "validation_checks": {
            "sample_count": len(samples),
            "unique_sample_ids": len(set(expected_ids)),
            "all_sample_ids_covered": True,
            "polarity_values_valid": True,
            "intensity_range_valid": True,
            "shapley_sum_error_tolerance": 1e-5,
            "text_offsets_exact_and_snippet_checked": True,
            "audio_vision_mapping_warnings_checked": True,
            "support_evidence_positive_class_delta": True,
            "counter_evidence_negative_class_delta": True,
            "no_evidence_reason_present_when_empty": True,
            "errors": [],
        },
        "limitations": [
            "Modality contributions are occlusion-based Shapley attributions, not physical causal effects.",
            "Continuous-window evidence is model-intervention evidence under a fixed occlusion baseline.",
            "Audio and visual locations are approximate proportional mappings because attachment 4 provides no original token-level timestamps.",
            "Validation-set local-window comparisons did not establish a significant universal advantage over random contiguous controls; concentration improved at the frozen window size.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "validation_report": str(validation_path),
        "manifest": str(manifest_path),
        "n": len(samples),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
