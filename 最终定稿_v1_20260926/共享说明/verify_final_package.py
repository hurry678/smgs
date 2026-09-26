#!/usr/bin/env python3
"""Verify the frozen final package with Python standard-library tools only."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"invalid manifest line: {line!r}")
        entries[line[66:]] = line[:64]
    return entries


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    failures: list[str] = []
    manifest = load_manifest()
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }

    missing = sorted(set(manifest) - actual)
    unlisted = sorted(actual - set(manifest))
    failures.extend(f"missing file: {path}" for path in missing)
    failures.extend(f"unlisted file: {path}" for path in unlisted)

    for relative, expected in sorted(manifest.items()):
        path = ROOT / relative
        if path.is_file():
            observed = sha256(path)
            if observed != expected:
                failures.append(f"SHA-256 mismatch: {relative}")

    zip_files = sorted(ROOT.rglob("*.zip"))
    for path in zip_files:
        try:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                failures.append(
                    f"ZIP CRC failure: {path.relative_to(ROOT)}::{bad_member}"
                )
        except zipfile.BadZipFile:
            failures.append(f"invalid ZIP: {path.relative_to(ROOT)}")

    q1_features = list((ROOT / "问题一/正式内容_q1_2_1/outputs/features").glob("*.npz"))
    if len(q1_features) != 100:
        failures.append(f"Q1 feature count is {len(q1_features)}, expected 100")

    q2_freeze = json.loads(
        (ROOT / "问题二/定稿决策审计/freeze_manifest.json").read_text(encoding="utf-8")
    )
    if q2_freeze.get("confirmation_status") != "candidate_rejected_baseline_retained":
        failures.append("Q2 freeze decision is not baseline retention")
    if csv_rows(ROOT / "问题二/正式提交/q2_predictions.csv") != 30:
        failures.append("Q2 attachment-3 submission does not contain 30 rows")

    q2_archives = sorted((ROOT / "问题二/定稿决策审计/audit_export").glob("*.zip"))
    archived_names: set[str] = set()
    for path in q2_archives:
        with zipfile.ZipFile(path) as archive:
            archived_names.update(archive.namelist())
    expected_members = {
        f"current_run/frozen/members/ds_seed{seed}.pt" for seed in range(42, 51)
    }
    if not expected_members.issubset(archived_names):
        failures.append("Q2 audit archives do not contain all nine frozen checkpoints")
    if "shared_encoder/pytorch_model.bin" not in archived_names:
        failures.append("Q2 audit archives do not contain the shared encoder")

    q3_root = ROOT / "问题三/定稿结果_v2_2/server_outputs"
    q3_verification = json.loads(
        (q3_root / "verification.json").read_text(encoding="utf-8")
    )
    if q3_verification.get("status") != "PASS":
        failures.append("Q3 verification status is not PASS")
    if q3_verification.get("selected_window_size") != 3:
        failures.append("Q3 frozen window is not W=3")
    q3_rows = csv_rows(q3_root / "submission/q3_predictions_explanations.csv")
    if q3_rows != 20:
        failures.append(f"Q3 attachment-4 submission contains {q3_rows} rows, expected 20")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "manifest_files": len(manifest),
        "zip_files_checked": len(zip_files),
        "q1_feature_files": len(q1_features),
        "q2_attachment3_rows": csv_rows(ROOT / "问题二/正式提交/q2_predictions.csv"),
        "q2_frozen_checkpoints_archived": len(expected_members & archived_names),
        "q2_shared_encoder_archived": "shared_encoder/pytorch_model.bin"
        in archived_names,
        "q3_attachment4_rows": q3_rows,
        "q3_selected_window_size": q3_verification.get("selected_window_size"),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
