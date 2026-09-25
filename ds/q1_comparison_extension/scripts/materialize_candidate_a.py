#!/usr/bin/env python3
"""Verify and safely materialize the frozen candidate-A comparison bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXTENSION_ROOT = HERE.parent
DEFAULT_METADATA = EXTENSION_ROOT / "candidates/A_current_audited_q1_2_1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_tree(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    required = [
        "outputs/alignment.jsonl",
        "outputs/acceptance.jsonl",
        "reports/boundary_audit.json",
        "reports/validation.json",
        "configs/q1.json",
        "run.py",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Candidate A bundle is incomplete: {missing}")
    features = sorted((root / "outputs/features").glob("*.npz"))
    alignment = [
        json.loads(line)
        for line in (root / "outputs/alignment.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    acceptance = [
        json.loads(line)
        for line in (root / "outputs/acceptance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = load_json(root / "reports/boundary_audit.json")
    validation = load_json(root / "reports/validation.json")
    feature_ids = {path.stem for path in features}
    alignment_ids = {row["sample_id"] for row in alignment}
    acceptance_ids = {row["sample_id"] for row in acceptance}
    audit_ids = {row["sample_id"] for row in audit["samples"]}
    if not (feature_ids == alignment_ids == acceptance_ids == audit_ids) or len(feature_ids) != 100:
        raise ValueError("Candidate A ID coverage differs across features, mapping, acceptance or audit")
    accepted = {row["sample_id"]: row for row in acceptance}
    feature_hash_errors = [
        sample_id for sample_id in sorted(feature_ids)
        if sha256(root / "outputs/features" / f"{sample_id}.npz")
        != accepted[sample_id]["feature_sha256"]
    ]
    if feature_hash_errors:
        raise ValueError(f"Candidate A feature hashes differ from acceptance: {feature_hash_errors[:5]}")
    if validation.get("checked_samples") != 100 or not validation.get("strict_release_passed"):
        raise ValueError("Candidate A strict validation is absent or did not pass")
    expected = {
        "features": metadata["feature_files"],
        "alignment": metadata["alignment_records"],
        "audit": metadata["boundary_audit_samples"],
        "validation": metadata["validation_checked_samples"],
    }
    observed = {
        "features": len(features),
        "alignment": len(alignment),
        "audit": len(audit["samples"]),
        "validation": validation["checked_samples"],
    }
    if observed != expected:
        raise ValueError(f"Candidate A metadata counts differ: observed={observed}, expected={expected}")
    return {
        "candidate_id": metadata["candidate_id"],
        "schema_version": metadata["candidate_schema_version"],
        "acceptance_policy": validation["acceptance_policy"],
        "counts": observed,
        "strict_release_passed": True,
        "feature_hashes_verified": 100,
    }


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = Path(member.filename)
        mode = member.external_attr >> 16
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {member.filename}")
        if (mode & 0o170000) == 0o120000:
            raise ValueError(f"Symlink is not allowed in candidate bundle: {member.filename}")
    return members


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    metadata = load_json(metadata_path)
    archive = metadata_path.parent / metadata["archive"]
    if not archive.is_file():
        raise FileNotFoundError(f"Candidate A archive is missing: {archive}")
    if archive.stat().st_size != metadata["archive_bytes"]:
        raise ValueError("Candidate A archive byte count differs from metadata")
    actual_sha = sha256(archive)
    if actual_sha != metadata["archive_sha256"]:
        raise ValueError(f"Candidate A archive SHA-256 differs: {actual_sha}")
    output = args.output_dir.resolve()
    marker = output / ".candidate_a_materialization.json"
    if output.exists() and marker.is_file() and not args.force:
        previous = load_json(marker)
        if previous.get("archive_sha256") == actual_sha:
            result = validate_tree(output, metadata)
            print(json.dumps({"status": "already_materialized_and_verified", "output_dir": str(output), **result}, ensure_ascii=False))
            return 0
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite existing directory without --force: {output}")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="candidate_a_", dir=output.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            members = safe_members(package)
            if len(members) != metadata["archive_entries"]:
                raise ValueError("Candidate A archive entry count differs from metadata")
            package.extractall(staging)
        result = validate_tree(staging, metadata)
        marker_payload = {
            "archive": str(archive),
            "archive_sha256": actual_sha,
            **result,
        }
        (staging / marker.name).write_text(
            json.dumps(marker_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    print(json.dumps({"status": "materialized_and_verified", "output_dir": str(output), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
