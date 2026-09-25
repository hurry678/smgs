#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a deterministic file inventory and SHA-256 manifest for ds."""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "file_inventory.csv"
MANIFEST = ROOT / "MANIFEST_SHA256.txt"
EXCLUDED_PARTS = {"__pycache__"}
EXCLUDED_NAMES = {INVENTORY.name, MANIFEST.name}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_for(rel: str) -> str:
    if rel.startswith("artifacts/q1/") or rel.startswith("q1_comparison_extension/"):
        return "Q1"
    if rel.startswith("artifacts/q2/") or rel.startswith("submission/q2_"):
        return "Q2"
    if rel.startswith("artifacts/q3/") or rel.startswith("logs/q3/") or rel.startswith("submission/q3_"):
        return "Q3"
    if rel.startswith("paper/") or rel in {"README.md", "PAPER_HANDOFF.md", "file_inventory.csv", "MANIFEST_SHA256.txt"}:
        return "PAPER_DELIVERY"
    if rel.startswith("scripts/"):
        return "REPRODUCTION"
    return "GENERAL"


def kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".json", ".jsonl", ".npz", ".svg", ".md", ".txt", ".py", ".tar", ".gz", ".zip"}:
        return suffix.lstrip(".") or "file"
    return suffix.lstrip(".") or "file"


def iter_files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        result.append(path)
    return sorted(result, key=lambda p: p.relative_to(ROOT).as_posix())


def build() -> tuple[int, int]:
    rows = []
    manifest_lines = []
    for path in iter_files():
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        stat = path.stat()
        rows.append({
            "path": rel,
            "size_bytes": stat.st_size,
            "sha256": digest,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "phase": phase_for(rel),
            "kind": kind_for(path),
        })
        manifest_lines.append(f"{digest}  {rel}")
    with INVENTORY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256", "modified_utc", "phase", "kind"])
        writer.writeheader()
        writer.writerows(rows)
    MANIFEST.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return len(rows), sum(int(row["size_bytes"]) for row in rows)


def verify() -> tuple[int, list[str]]:
    if not MANIFEST.exists():
        return 0, ["manifest missing"]
    errors = []
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing: {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"hash mismatch: {rel}")
        checked += 1
    return checked, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify ds inventory and SHA-256 manifest.")
    parser.add_argument("--check", action="store_true", help="Verify existing manifest")
    args = parser.parse_args()
    if args.check:
        checked, errors = verify()
        if errors:
            print("\n".join(errors))
            return 1
        print(f"Verified {checked} files")
        return 0
    count, total = build()
    print(f"Wrote {INVENTORY.name} and {MANIFEST.name}: {count} files, {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
