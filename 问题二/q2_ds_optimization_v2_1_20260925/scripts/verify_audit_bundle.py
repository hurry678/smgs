#!/usr/bin/env python3
"""Verify every file in a split Q2 high-compute audit bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    args = parser.parse_args()
    export_dir = args.export_dir.resolve()
    index_path = export_dir / "audit_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "complete":
        raise SystemExit("audit index status is not complete")
    required_counts = {
        "cv_results",
        "cv_logs",
        "pool_results",
        "pool_checkpoints",
        "pool_logs",
        "select_caches",
        "confirm_caches",
        "frozen_members",
    }
    if set(index.get("count_checks", {})) != required_counts:
        raise SystemExit("audit index count-check set is incomplete")
    for name, check in index["count_checks"].items():
        if int(check["actual"]) != int(check["expected"]):
            raise SystemExit(f"count check failed: {name}={check}")
    if not index.get("parts") or not index.get("files"):
        raise SystemExit("audit bundle cannot be empty")
    fixed_counts = {
        "cv_results": 36,
        "cv_logs": 36,
        "pool_results": 48,
        "pool_checkpoints": 48,
        "pool_logs": 48,
        "select_caches": 57,
    }
    for name, expected in fixed_counts.items():
        if int(index["count_checks"][name]["expected"]) != expected:
            raise SystemExit(f"unexpected required count: {name}")
    if int(index["count_checks"]["confirm_caches"]["expected"]) < 9:
        raise SystemExit("confirmation cache count is too small")
    frozen_count = int(index["count_checks"]["frozen_members"]["expected"])
    if not 5 <= frozen_count <= 13:
        raise SystemExit("frozen member count is outside the protocol bounds")

    checksum_path = export_dir / "SHA256SUMS.txt"
    checksum_rows = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(None, 1)
        checksum_rows[filename.strip()] = digest
    expected_checksum_names = {part["file"] for part in index["parts"]} | {
        index_path.name
    }
    if set(checksum_rows) != expected_checksum_names:
        raise SystemExit("SHA256SUMS.txt has missing or unexpected entries")
    if checksum_rows[index_path.name] != sha256_file(index_path):
        raise SystemExit("audit index SHA-256 mismatch")

    archives = {}
    for part in index["parts"]:
        path = export_dir / part["file"]
        actual = sha256_file(path)
        if actual != part["sha256"] or actual != checksum_rows[part["file"]]:
            raise SystemExit(f"archive SHA-256 mismatch: {path}")
        archive = zipfile.ZipFile(path, "r")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SystemExit(f"archive CRC failure: {path}:{bad_member}")
        archives[part["file"]] = archive

    try:
        records_by_part = {part["file"]: [] for part in index["parts"]}
        seen_paths = set()
        for record in index["files"]:
            if record["part"] not in records_by_part:
                raise SystemExit(f"unknown archive part: {record['part']}")
            if record["path"] in seen_paths:
                raise SystemExit(f"duplicate indexed payload: {record['path']}")
            seen_paths.add(record["path"])
            records_by_part[record["part"]].append(record)
        required_paths = {
            "current_run/resolved_inputs.json",
            "current_run/baseline_replay/verification.json",
            "current_run/baseline_replay/valid_valid_evaluation.json",
            "current_run/arm_selection.json",
            "current_run/pool_inventory.json",
            "current_run/selection.json",
            "current_run/confirmation.json",
            "current_run/frozen/freeze_manifest.json",
            "current_run/audit_support/audit_targets.npz",
            "current_run/publication/publication_summary.json",
            "current_run/publication/test_evaluation.json",
            "current_run/publication/test_predictions.npz",
            "current_run/publication/attachment3_predictions.csv",
            "current_run/publication/attachment3_predictions.json",
            "shared_encoder/pytorch_model.bin",
            "source/scripts/q2hc_core.py",
            "source/scripts/q2hc_select.py",
            "source/scripts/run_q2hc.py",
            "source/scripts/verify_audit_bundle.py",
            "source/configs/protocol.json",
            "source/ds/scripts/q2_models.py",
            "source/ds/scripts/q2_train.py",
            "source/ds/scripts/q2_eval.py",
            "source/ds/scripts/q2_eval_ensemble.py",
            "source/ds/scripts/q2_eval_robustness.py",
        }
        if not required_paths <= seen_paths:
            raise SystemExit(
                f"audit bundle is missing required paths: {sorted(required_paths - seen_paths)}"
            )
        path_counts = {
            "cv_results": sum(
                path.startswith("current_cv/search/") and path.endswith("/result.json")
                for path in seen_paths
            ),
            "cv_logs": sum(
                path.startswith("current_cv/search/") and path.endswith("/run.log")
                for path in seen_paths
            ),
            "pool_results": sum(
                path.startswith("current_run/pool/") and path.endswith("/result.json")
                for path in seen_paths
            ),
            "pool_checkpoints": sum(
                path.startswith("current_run/pool/") and path.endswith("/checkpoint.pt")
                for path in seen_paths
            ),
            "pool_logs": sum(
                path.startswith("current_run/pool/") and path.endswith("/run.log")
                for path in seen_paths
            ),
            "select_caches": sum(
                path.startswith("current_run/cache/select/") and path.endswith(".npz")
                for path in seen_paths
            ),
            "confirm_caches": sum(
                path.startswith("current_run/cache/confirm/") and path.endswith(".npz")
                for path in seen_paths
            ),
            "frozen_members": sum(
                path.startswith("current_run/frozen/members/") and path.endswith(".pt")
                for path in seen_paths
            ),
        }
        for name, actual in path_counts.items():
            if actual != int(index["count_checks"][name]["actual"]):
                raise SystemExit(f"indexed path count mismatch: {name}")
        auxiliary_cache_counts = {
            "select_metadata": sum(
                path.startswith("current_run/cache/select/")
                and path.endswith(".json")
                and not path.endswith("/inventory.json")
                for path in seen_paths
            ),
            "select_logs": sum(
                path.startswith("current_run/cache/select/") and path.endswith(".log")
                for path in seen_paths
            ),
            "confirm_metadata": sum(
                path.startswith("current_run/cache/confirm/")
                and path.endswith(".json")
                and not path.endswith("/inventory.json")
                for path in seen_paths
            ),
            "confirm_logs": sum(
                path.startswith("current_run/cache/confirm/") and path.endswith(".log")
                for path in seen_paths
            ),
        }
        if auxiliary_cache_counts["select_metadata"] != path_counts["select_caches"]:
            raise SystemExit("selection cache metadata count mismatch")
        if auxiliary_cache_counts["select_logs"] != path_counts["select_caches"]:
            raise SystemExit("selection cache log count mismatch")
        if auxiliary_cache_counts["confirm_metadata"] != path_counts["confirm_caches"]:
            raise SystemExit("confirmation cache metadata count mismatch")
        if auxiliary_cache_counts["confirm_logs"] != path_counts["confirm_caches"]:
            raise SystemExit("confirmation cache log count mismatch")
        for part in index["parts"]:
            indexed = {record["path"] for record in records_by_part[part["file"]]}
            archive_names = archives[part["file"]].namelist()
            if len(archive_names) != len(set(archive_names)):
                raise SystemExit(f"duplicate ZIP member: {part['file']}")
            if set(archive_names) != indexed:
                raise SystemExit(f"ZIP/index member mismatch: {part['file']}")
            if int(part["entries"]) != len(indexed):
                raise SystemExit(f"ZIP entry count mismatch: {part['file']}")
        checked = 0
        for record in index["files"]:
            archive = archives[record["part"]]
            digest = hashlib.sha256()
            with archive.open(record["path"], "r") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise SystemExit(f"payload SHA-256 mismatch: {record['path']}")
            checked += 1
    finally:
        for archive in archives.values():
            archive.close()

    print(
        json.dumps(
            {
                "status": "passed",
                "parts": len(index["parts"]),
                "files": checked,
                "total_source_bytes": index["total_source_bytes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
