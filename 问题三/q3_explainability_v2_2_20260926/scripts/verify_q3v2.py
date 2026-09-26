#!/usr/bin/env python3
"""Verify Q3 v2.2 source integrity, frozen inference, and final submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)


def verify_source_manifest(extension_root: Path) -> dict[str, Any]:
    manifest_path = extension_root / "MANIFEST_SHA256.txt"
    if not manifest_path.is_file():
        raise ValueError("source MANIFEST_SHA256.txt is missing")
    rows: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split("  ", 1)
        if relative in rows:
            raise ValueError(f"duplicate source-manifest entry: {relative}")
        rows[relative] = digest
    expected_files = {
        path.relative_to(extension_root).as_posix()
        for path in extension_root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST_SHA256.txt"
        and "__pycache__" not in path.parts
        and ".ruff_cache" not in path.parts
    }
    if set(rows) != expected_files:
        raise ValueError(
            "source manifest coverage mismatch: "
            f"missing={sorted(expected_files - set(rows))}, "
            f"extra={sorted(set(rows) - expected_files)}"
        )
    mismatches = [
        relative
        for relative, digest in rows.items()
        if sha256_file(extension_root / relative) != digest
    ]
    if mismatches:
        raise ValueError(f"source manifest digest mismatch: {mismatches}")
    return {"files": len(rows), "manifest_sha256": sha256_file(manifest_path)}


def verify_output_hashes(
    submission_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "csv": submission_dir / "q3_predictions_explanations.csv",
        "jsonl": submission_dir / "q3_evidence.jsonl",
        "validation_report_copy": submission_dir / "q3_validation_report.json",
        "attachment4_report": submission_dir / "q3_attachment4_report.json",
    }
    checked = 0
    for key, path in expected.items():
        if not path.is_file():
            raise ValueError(f"missing final output: {path}")
        recorded = manifest["outputs"].get(f"{key}_sha256")
        actual = sha256_file(path)
        if recorded != actual:
            raise ValueError(f"final output SHA-256 mismatch: {key}")
        checked += 1
    return {"files": checked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    submission_dir = args.run_dir / "submission"
    required = {
        "preflight_validation": args.run_dir / "preflight_validation.json",
        "preflight_attachment4": args.run_dir / "preflight_attachment4.json",
        "freeze_parameters": args.run_dir / "freeze_parameters.json",
        "result_comparison": args.run_dir / "result_comparison.json",
        "alignment_audit": args.run_dir / "alignment_audit.json",
        "raw_attachment4": (
            args.run_dir / "final_attachment4" / "q3_attachment4_raw.json"
        ),
        "manifest": submission_dir / "q3_freeze_manifest.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required Q3 outputs: {missing}")

    preflight_validation = json.loads(
        required["preflight_validation"].read_text(encoding="utf-8")
    )
    preflight_attachment4 = json.loads(
        required["preflight_attachment4"].read_text(encoding="utf-8")
    )
    freeze = json.loads(required["freeze_parameters"].read_text(encoding="utf-8"))
    result_comparison = json.loads(
        required["result_comparison"].read_text(encoding="utf-8")
    )
    alignment = json.loads(required["alignment_audit"].read_text(encoding="utf-8"))
    protocol_path = args.extension_root / "configs" / "protocol.json"
    protocol_config = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(protocol_path)
    raw = json.loads(required["raw_attachment4"].read_text(encoding="utf-8"))
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if (
        preflight_validation.get("status") != "PASS"
        or preflight_validation.get("scope") != "validation"
        or preflight_validation.get("protocol_sha256") != protocol_sha256
    ):
        raise SystemExit("validation preflight status is not PASS")
    if (
        preflight_attachment4.get("status") != "PASS"
        or preflight_attachment4.get("scope") != "attachment4"
        or preflight_attachment4.get("protocol_sha256") != protocol_sha256
    ):
        raise SystemExit("attachment-4 preflight status is not PASS")
    if (
        freeze.get("schema") != "q3v2.2-explanation-freeze-v1"
        or freeze.get("status") != "frozen"
    ):
        raise SystemExit("explanation parameters are not frozen")
    if freeze.get("attachment4_used_for_selection") is not False:
        raise SystemExit("attachment 4 was marked as used for selection")
    if (
        result_comparison.get("schema") != "q3v2.2-result-comparison-v1"
        or result_comparison.get("status") != "PASS"
        or result_comparison.get("predictor_invariance", {}).get(
            "polarity_change_count"
        )
        != 0
        or float(
            result_comparison.get("predictor_invariance", {}).get(
                "maximum_intensity_difference", float("inf")
            )
        )
        > 1e-7
        or float(
            result_comparison.get("predictor_invariance", {}).get(
                "maximum_log_probability_difference", float("inf")
            )
        )
        > 1e-7
    ):
        raise SystemExit("v2-to-v2.2 predictor invariance check failed")
    if alignment.get("status") != "PASS":
        raise SystemExit("alignment status is not PASS")
    if float(alignment.get("mfa_sample_rate", -1.0)) < float(
        alignment.get("minimum_mfa_sample_rate", 1.0)
    ):
        raise SystemExit("MFA sample coverage is below the frozen threshold")
    reusable = protocol_config["attachment4"]["reusable_verified_artifacts"]
    if result_comparison.get("old_raw_sha256") != reusable[
        "v2_attachment4_raw_sha256"
    ] or result_comparison.get("new_raw_sha256") != sha256_file(
        required["raw_attachment4"]
    ):
        raise SystemExit("v2-to-v2.2 comparison input SHA-256 mismatch")
    reusable_paths = {
        "attachment4_aligned_npz_sha256": (
            args.run_dir / "attachment4_data" / "attachment4_aligned.npz"
        ),
        "attachment4_offsets_npz_sha256": args.run_dir / "attachment4_offsets.npz",
        "alignment_audit_sha256": required["alignment_audit"],
        "alignment_mapper_sha256": (
            args.extension_root / "scripts" / "q3_prepare_alignment_v2.py"
        ),
    }
    for key, path in reusable_paths.items():
        if not path.is_file() or sha256_file(path) != reusable[key]:
            raise SystemExit(f"reused attachment-4 artifact SHA-256 mismatch: {key}")
    expected_resources = protocol_config["alignment"]["resource_sha256"]
    observed_resources = alignment.get("mfa", {})
    for observed_key, expected_key in (
        ("dictionary_sha256", "dictionary"),
        ("acoustic_model_sha256", "acoustic_model"),
        ("g2p_model_sha256", "g2p_model"),
    ):
        if observed_resources.get(observed_key) != expected_resources[expected_key]:
            raise SystemExit(f"MFA resource SHA-256 mismatch: {observed_key}")
    expected_q2_freeze = protocol_config["immutable_repo_files"][
        "\u4f18\u5316/q2yhv2_1/server_results_20260925/freeze_manifest.json"
    ]
    expected_q2_version = protocol_config["immutable_repo_files"][
        "ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json"
    ]
    predictor = freeze.get("predictor", {})
    if predictor.get("q2_freeze_manifest_sha256") != expected_q2_freeze:
        raise SystemExit("frozen Q2 decision manifest SHA-256 mismatch")
    if predictor.get("q2_version_manifest_sha256") != expected_q2_version:
        raise SystemExit("frozen Q2 version manifest SHA-256 mismatch")
    if raw.get("report", {}).get("mode") != "attachment4":
        raise SystemExit("raw final output mode is not attachment4")
    samples = raw.get("samples", [])
    if len(samples) != 20 or len({sample["sample_id"] for sample in samples}) != 20:
        raise SystemExit("raw attachment-4 output does not contain 20 unique samples")
    protocol = raw["report"].get("protocol", {})
    if protocol.get("frozen_explanation_parameters") is not True:
        raise SystemExit("raw attachment-4 report is not marked frozen")
    if protocol.get("attachment4_used_for_tuning") is not False:
        raise SystemExit("raw attachment-4 tuning policy is invalid")
    top_fidelity = (
        raw["report"].get("local_occlusion", {}).get("top_support_fidelity", {})
    )
    if (
        top_fidelity.get("comparison_schema") != "q3v2.2-symmetric-top-support-v1"
        or top_fidelity.get("control_selection_symmetric") is not True
    ):
        raise SystemExit("raw output does not use the symmetric v2.2 comparison")
    if (
        manifest.get("version") != "q3_explainability_v2_2_20260926"
        or manifest.get("manifest_version") != 3
        or manifest.get("status") != "completed_with_frozen_attachment4_inference"
    ):
        raise SystemExit("final manifest status is invalid")
    if manifest.get("model_manifest_sha256") != expected_q2_version:
        raise SystemExit("final manifest references the wrong Q2 model")
    if manifest.get("outputs", {}).get("raw_json_sha256") != sha256_file(
        required["raw_attachment4"]
    ):
        raise SystemExit("raw attachment-4 output SHA-256 mismatch")
    if manifest.get("inputs", {}).get("audit_sha256") != sha256_file(
        required["alignment_audit"]
    ):
        raise SystemExit("alignment audit SHA-256 mismatch")

    absolute_values = [
        value
        for value in json_strings(manifest)
        if value.startswith("/") or value.startswith("file:/")
    ]
    if absolute_values:
        raise SystemExit(
            f"final manifest contains non-portable absolute paths: {absolute_values[:5]}"
        )

    source_result = verify_source_manifest(args.extension_root)
    output_result = verify_output_hashes(submission_dir, manifest)
    submission_bytes = sum(
        path.stat().st_size for path in submission_dir.rglob("*") if path.is_file()
    )
    maximum_bytes = 50_000_000
    if submission_bytes > maximum_bytes:
        raise SystemExit(f"submission size {submission_bytes} exceeds {maximum_bytes}")

    baseline_diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ds",
            "\u4f18\u5316/q2yhv2_1",
            "\u95ee\u9898\u4e09/q3_explainability_v2_20260925",
            "\u4f18\u5316/q3_explainability_v2_20260925",
        ],
        cwd=args.repo_root,
    )
    if baseline_diff.returncode != 0:
        raise SystemExit("protected Q2/Q3-v2 baseline paths differ from HEAD")

    result = {
        "schema": "q3v2.2-final-verification-v1",
        "status": "PASS",
        "source_manifest": source_result,
        "output_hashes": output_result,
        "submission_bytes": submission_bytes,
        "maximum_submission_bytes": maximum_bytes,
        "attachment4_samples": len(samples),
        "mfa_aligned_samples": int(alignment["mfa_aligned_samples"]),
        "fallback_samples": int(alignment["fallback_samples"]),
        "selected_window_size": int(freeze["selected_window_size"]),
        "protected_baseline_diff": "clean",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
