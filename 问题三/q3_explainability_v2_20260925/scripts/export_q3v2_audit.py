#!/usr/bin/env python3
"""Create and self-check a portable Q3 v2 audit archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable


MAX_BYTES = 50_000_000
FIXED_TIME = (2026, 9, 25, 0, 0, 0)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or ".ruff_cache" in path.parts:
            continue
        yield f"source/{path.relative_to(root).as_posix()}", path


def run_files(root: Path, export_dir: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or export_dir in path.parents:
            continue
        relative = path.relative_to(root)
        if (
            len(relative.parts) >= 2
            and relative.parts[0] == "mfa_work"
            and relative.parts[1] in {"audio", "corpus"}
        ):
            continue
        if "__pycache__" in path.parts or ".ruff_cache" in path.parts:
            continue
        yield f"run/{relative.as_posix()}", path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()

    args.export_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.export_dir / "q3_explainability_v2_audit.zip"
    index_path = args.export_dir / "audit_index.json"
    sums_path = args.export_dir / "SHA256SUMS.txt"
    files = list(source_files(args.extension_root)) + list(
        run_files(args.run_dir, args.export_dir)
    )
    names = [name for name, _ in files]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate audit archive member name")
    required = {
        "source/MANIFEST_SHA256.txt",
        "source/configs/protocol.json",
        "source/scripts/q3_explain_core_v2.py",
        "source/scripts/q3_prepare_alignment_v2.py",
        "run/preflight_validation.json",
        "run/preflight_attachment4.json",
        "run/freeze_parameters.json",
        "run/alignment_audit.json",
        "run/final_attachment4/q3_attachment4_raw.json",
        "run/submission/q3_freeze_manifest.json",
        "run/verification.json",
    }
    if not required <= set(names):
        raise SystemExit(f"audit archive is missing: {sorted(required - set(names))}")

    records = []
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, path in files:
            payload = path.read_bytes()
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
            records.append(
                {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
    archive_bytes = archive_path.stat().st_size
    if archive_bytes > MAX_BYTES:
        archive_path.unlink()
        raise SystemExit(f"audit archive exceeds {MAX_BYTES} bytes: {archive_bytes}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SystemExit(f"audit ZIP CRC failure: {bad_member}")
        archive_names = archive.namelist()
        if archive_names != names:
            raise SystemExit("audit ZIP member order or coverage mismatch")
        for record in records:
            if sha256_bytes(archive.read(record["path"])) != record["sha256"]:
                raise SystemExit(f"audit payload mismatch: {record['path']}")

    index = {
        "schema": "q3v2-audit-index-v1",
        "status": "complete",
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_bytes,
        "maximum_archive_bytes": MAX_BYTES,
        "entries": len(records),
        "files": records,
        "excluded": [
            "MFA extracted WAV files",
            "MFA corpus WAV symlinks",
            "frozen Q2 checkpoints and MiniLM already covered by Q2 v2.1 audit archive",
        ],
    }
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sums_path.write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n"
        f"{sha256_file(index_path)}  {index_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(index, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
