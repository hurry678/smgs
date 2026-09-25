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
PAIR_NAMES = ("text_audio", "text_vision", "audio_vision")
POLARITIES = {"Negative", "Neutral", "Positive"}
CST = timezone(timedelta(hours=8))
SCRIPT_PATH = Path(__file__).resolve()
EXTENSION_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    for prefix, root in (
        ("extension://", EXTENSION_ROOT),
        ("repo://", REPO_ROOT),
    ):
        try:
            return prefix + resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return "external://" + resolved.name


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
        errors.append(
            f"{sample_id}: invalid support_modality={sample.get('support_modality')!r}"
        )
    class_phi = np.asarray(
        [
            sample.get(f"class_contrib_{modality}", float("nan"))
            for modality in MODALITIES
        ],
        dtype=np.float64,
    )
    if not np.isfinite(class_phi).all():
        errors.append(f"{sample_id}: class Shapley values are not finite")
    else:
        strongest = MODALITIES[int(np.argmax(np.abs(class_phi)))]
        support_index = int(np.argmax(class_phi))
        if float(class_phi[support_index]) > 0.0:
            primary = MODALITIES[support_index]
            support = primary
            basis = "largest_positive_class_shapley"
        else:
            primary = strongest
            support = None
            basis = "no_positive_class_shapley_fallback_to_absolute"
        if sample.get("strongest_modality") != strongest:
            errors.append(
                f"{sample_id}: strongest_modality violates class-Shapley rule"
            )
        if sample.get("primary_modality") != primary:
            errors.append(f"{sample_id}: primary_modality violates support rule")
        if sample.get("support_modality") != support:
            errors.append(f"{sample_id}: support_modality violates support rule")
        if sample.get("primary_modality_basis") != basis:
            errors.append(f"{sample_id}: primary_modality_basis mismatch")
    for key in ("class_shapley_sum_error", "intensity_shapley_sum_error"):
        value = float(sample.get(key, float("nan")))
        if not math.isfinite(value) or value > 1e-5:
            errors.append(f"{sample_id}: {key} too large ({value!r})")
    for interaction_key in (
        "class_pair_interactions",
        "intensity_pair_interactions",
    ):
        values = sample.get(interaction_key)
        if not isinstance(values, dict) or set(values) != set(PAIR_NAMES):
            errors.append(f"{sample_id}: invalid {interaction_key} keys")
        elif not all(math.isfinite(float(values[name])) for name in PAIR_NAMES):
            errors.append(f"{sample_id}: non-finite {interaction_key}")

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
                errors.append(
                    f"{sample_id}: {modality} support evidence has non-positive class_delta"
                )
            if modality == "text":
                if row.get("mapping_status") != "exact":
                    errors.append(f"{sample_id}: text evidence is not exactly mapped")
                    continue
                a = row.get("text_char_start")
                b = row.get("text_char_end")
                if (
                    not isinstance(a, int)
                    or not isinstance(b, int)
                    or not (0 <= a < b <= len(raw_text))
                ):
                    errors.append(f"{sample_id}: invalid text span {(a, b)!r}")
                elif row.get("text_snippet") != raw_text[a:b]:
                    errors.append(
                        f"{sample_id}: text snippet does not match raw text span"
                    )
            elif row.get("mapping_status") in {
                "forced_alignment",
                "aligned_with_fallback",
                "approximate",
            }:
                a = float(row.get("time_start_seconds", -1.0))
                b = float(row.get("time_end_seconds", -1.0))
                if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a <= b):
                    errors.append(f"{sample_id}: invalid mapped time {(a, b)!r}")
                if (
                    row.get("mapping_status") == "approximate"
                    and "proportional" not in str(row.get("warning", "")).lower()
                ):
                    errors.append(f"{sample_id}: approximate mapping lacks warning")
                if modality == "vision":
                    frame_start = row.get("frame_start_est")
                    frame_end = row.get("frame_end_est")
                    if not (
                        isinstance(frame_start, int)
                        and isinstance(frame_end, int)
                        and 0 <= frame_start <= frame_end
                    ):
                        errors.append(f"{sample_id}: invalid visual frame interval")
            else:
                errors.append(
                    f"{sample_id}: invalid {modality} mapping status "
                    f"{row.get('mapping_status')!r}"
                )
        counter_rows = counter_evidence.get(modality, [])
        if not isinstance(counter_rows, list):
            errors.append(f"{sample_id}: {modality} counter_evidence is not a list")
            continue
        for row in counter_rows:
            if row.get("evidence_direction") != "counter":
                errors.append(
                    f"{sample_id}: {modality} counter evidence has wrong direction"
                )
            if float(row.get("class_delta", 0.0)) >= 0.0:
                errors.append(
                    f"{sample_id}: {modality} counter evidence has non-negative class_delta"
                )
            if modality == "text":
                start = row.get("text_char_start")
                end = row.get("text_char_end")
                if row.get("mapping_status") != "exact" or not (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(raw_text)
                    and row.get("text_snippet") == raw_text[start:end]
                ):
                    errors.append(f"{sample_id}: invalid counter-text mapping")
            else:
                status = row.get("mapping_status")
                start = float(row.get("time_start_seconds", -1.0))
                end = float(row.get("time_end_seconds", -1.0))
                if status not in {
                    "forced_alignment",
                    "aligned_with_fallback",
                    "approximate",
                } or not (
                    math.isfinite(start) and math.isfinite(end) and 0.0 <= start <= end
                ):
                    errors.append(f"{sample_id}: invalid counter-{modality} mapping")
                if (
                    status == "approximate"
                    and "proportional" not in str(row.get("warning", "")).lower()
                ):
                    errors.append(
                        f"{sample_id}: approximate counter mapping lacks warning"
                    )

    has_support = any(key_evidence.get(m) for m in MODALITIES)
    reason = sample.get("no_evidence_reason")
    if has_support and reason is not None:
        errors.append(
            f"{sample_id}: has support evidence but non-null no_evidence_reason"
        )
    if not has_support and not isinstance(reason, str):
        errors.append(
            f"{sample_id}: no support evidence but missing no_evidence_reason"
        )
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
    ap.add_argument("--explanation-parameters", type=Path, required=True)
    ap.add_argument("--alignment-script", type=Path, required=True)
    ap.add_argument("--math-script", type=Path, required=True)
    args = ap.parse_args()

    raw = load_raw(args.raw_json)
    samples = raw["samples"]
    report = raw["report"]
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    frozen_parameters = json.loads(
        args.explanation_parameters.read_text(encoding="utf-8")
    )
    alignment_audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if report.get("mode") != "attachment4":
        raise SystemExit("raw report is not an attachment-4 inference report")
    if validation_report.get("mode") != "validation":
        raise SystemExit("--validation-report is not a validation report")
    if frozen_parameters.get("status") != "frozen":
        raise SystemExit("explanation parameter file is not frozen")
    if (
        alignment_audit.get("schema") != "q3v2-mfa-alignment-audit-v1"
        or alignment_audit.get("status") != "PASS"
    ):
        raise SystemExit("attachment-4 alignment audit did not pass")
    if float(alignment_audit.get("mfa_sample_rate", -1.0)) < float(
        alignment_audit.get("minimum_mfa_sample_rate", 1.0)
    ):
        raise SystemExit("attachment-4 MFA coverage is below the frozen threshold")
    if sha256_file(args.validation_report) != frozen_parameters.get(
        "selected_report_sha256"
    ):
        raise SystemExit("validation report does not match the frozen selected report")
    if sha256_file(args.model_manifest) != frozen_parameters.get("predictor", {}).get(
        "q2_version_manifest_sha256"
    ):
        raise SystemExit("model manifest does not match the frozen predictor")
    protocol = report.get("protocol", {})
    frozen_pairs = {
        "window_size": "selected_window_size",
        "stride": "selected_stride",
        "top_k_per_modality": "top_k_per_modality",
        "seed": "seed",
    }
    for protocol_key, frozen_key in frozen_pairs.items():
        if protocol.get(protocol_key) != frozen_parameters.get(frozen_key):
            raise SystemExit(
                f"attachment-4 protocol {protocol_key} does not match frozen parameters"
            )
    if protocol.get("frozen_explanation_parameters") is not True:
        raise SystemExit("attachment-4 inference is not marked frozen")
    if protocol.get("attachment4_used_for_tuning") is not False:
        raise SystemExit("attachment-4 tuning policy is invalid")
    with np.load(args.attachment4, allow_pickle=False) as z:
        expected_ids = [str(x) for x in z["ids"].tolist()]
        raw_texts = [str(x) for x in z["raw_text"].tolist()]
    if len(samples) != len(expected_ids):
        raise SystemExit(
            f"sample count mismatch: {len(samples)} vs {len(expected_ids)}"
        )
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
    attachment_report_path = args.out_dir / "q3_attachment4_report.json"
    manifest_path = args.out_dir / "q3_freeze_manifest.json"

    csv_fields = [
        "sample_id",
        "polarity",
        "intensity",
        "raw_intensity",
        "primary_modality",
        "primary_modality_basis",
        "strongest_modality",
        "support_modality",
        "modality_effect_text",
        "modality_effect_audio",
        "modality_effect_vision",
        "class_contrib_text",
        "class_contrib_audio",
        "class_contrib_vision",
        "intensity_contrib_text",
        "intensity_contrib_audio",
        "intensity_contrib_vision",
        "class_abs_share_text",
        "class_abs_share_audio",
        "class_abs_share_vision",
        "intensity_abs_share_text",
        "intensity_abs_share_audio",
        "intensity_abs_share_vision",
        "loo_text",
        "loo_audio",
        "loo_vision",
        "class_pair_interactions_json",
        "intensity_pair_interactions_json",
        "key_evidence_text_json",
        "key_evidence_audio_json",
        "key_evidence_vision_json",
        "counter_evidence_text_json",
        "counter_evidence_audio_json",
        "counter_evidence_vision_json",
        "no_evidence_reason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for sid in expected_ids:
            s = by_id[sid]
            row: Dict[str, Any] = {k: s.get(k) for k in csv_fields if k in s}
            row["class_pair_interactions_json"] = json_compact(
                s["class_pair_interactions"]
            )
            row["intensity_pair_interactions_json"] = json_compact(
                s["intensity_pair_interactions"]
            )
            for modality in MODALITIES:
                row[f"key_evidence_{modality}_json"] = json_compact(
                    s["key_evidence"].get(modality, [])
                )
                row[f"counter_evidence_{modality}_json"] = json_compact(
                    s["counter_evidence"].get(modality, [])
                )
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8", newline="") as f:
        for sid in expected_ids:
            f.write(
                json.dumps(by_id[sid], ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    validation_path.write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    attachment_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fallback_count = int(alignment_audit.get("fallback_samples", 0))
    mapping_limitation = (
        "Some audio/visual evidence locations use explicitly marked proportional "
        "fallback because MFA did not align every attachment-4 sample."
        if fallback_count
        else "All published audio/visual evidence locations use MFA-derived word timings."
    )

    manifest: Dict[str, Any] = {
        "manifest_version": 2,
        "question": "问题三",
        "status": "completed_with_frozen_attachment4_inference",
        "created_at": datetime.now(CST).isoformat(timespec="seconds"),
        "model_manifest": portable_path(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "explanation_script": portable_path(args.explain_script),
        "explanation_script_sha256": sha256_file(args.explain_script),
        "alignment_script": portable_path(args.alignment_script),
        "alignment_script_sha256": sha256_file(args.alignment_script),
        "math_script": portable_path(args.math_script),
        "math_script_sha256": sha256_file(args.math_script),
        "packager_script": portable_path(SCRIPT_PATH),
        "packager_script_sha256": sha256_file(SCRIPT_PATH),
        "explanation_parameters": portable_path(args.explanation_parameters),
        "explanation_parameters_sha256": sha256_file(args.explanation_parameters),
        "inputs": {
            "data": portable_path(args.data),
            "data_sha256": sha256_file(args.data),
            "offsets": portable_path(args.offsets),
            "offsets_sha256": sha256_file(args.offsets),
            "audit": portable_path(args.audit),
            "audit_sha256": sha256_file(args.audit),
            "attachment4": portable_path(args.attachment4),
            "attachment4_sha256": sha256_file(args.attachment4),
        },
        "frozen_parameters": frozen_parameters,
        "validation_report": portable_path(args.validation_report),
        "validation_report_sha256": sha256_file(args.validation_report),
        "outputs": {
            "raw_json": portable_path(args.raw_json),
            "raw_json_sha256": sha256_file(args.raw_json),
            "csv": portable_path(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "jsonl": portable_path(jsonl_path),
            "jsonl_sha256": sha256_file(jsonl_path),
            "validation_report_copy": portable_path(validation_path),
            "validation_report_copy_sha256": sha256_file(validation_path),
            "attachment4_report": portable_path(attachment_report_path),
            "attachment4_report_sha256": sha256_file(attachment_report_path),
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
            "primary_and_support_rules_checked": True,
            "pair_interaction_schema_checked": True,
            "frozen_parameter_match_checked": True,
            "alignment_audit_status_checked": True,
            "support_evidence_positive_class_delta": True,
            "counter_evidence_negative_class_delta": True,
            "no_evidence_reason_present_when_empty": True,
            "errors": [],
        },
        "limitations": [
            "Modality contributions are occlusion-based Shapley attributions, not physical causal effects.",
            "Continuous-window evidence is model-intervention evidence under a fixed occlusion baseline.",
            mapping_limitation,
            "Window size is selected only from validation by budget-normalized top-support fidelity with video-group clustered confidence intervals.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "csv": str(csv_path),
                "jsonl": str(jsonl_path),
                "validation_report": str(validation_path),
                "attachment4_report": str(attachment_report_path),
                "manifest": str(manifest_path),
                "n": len(samples),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
