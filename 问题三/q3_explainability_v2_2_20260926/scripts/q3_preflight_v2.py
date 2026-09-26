#!/usr/bin/env python3
"""Verify immutable Q2 inputs and Q3 runtime prerequisites before execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(
    checks: list[dict[str, Any]],
    label: str,
    path: Path,
    expected_sha256: str | None = None,
) -> None:
    exists = path.is_file()
    actual = sha256_file(path) if exists else None
    passed = exists and (expected_sha256 is None or actual == expected_sha256)
    checks.append(
        {
            "label": label,
            "file": path.name,
            "exists": exists,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual,
            "passed": bool(passed),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["validation", "attachment4"],
        required=True,
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--mfa-executable", type=Path)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--acoustic-model", type=Path)
    parser.add_argument("--g2p-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    check_file(checks, "protocol", args.protocol)
    for relative_path, digest in protocol["immutable_repo_files"].items():
        check_file(
            checks,
            f"repo:{relative_path}",
            args.repo_root / relative_path,
            str(digest),
        )
    for relative_path, digest in protocol["legacy_inputs"].items():
        check_file(
            checks,
            f"legacy:{relative_path}",
            args.legacy_root / relative_path,
            str(digest),
        )
    if args.scope == "attachment4":
        required_options = {
            "--raw-dir": args.raw_dir,
            "--mfa-executable": args.mfa_executable,
            "--dictionary": args.dictionary,
            "--acoustic-model": args.acoustic_model,
            "--g2p-model": args.g2p_model,
        }
        missing_options = [
            name for name, value in required_options.items() if value is None
        ]
        if missing_options:
            raise SystemExit(f"attachment4 scope requires options: {missing_options}")
        resource_hashes = protocol["alignment"]["resource_sha256"]
        resources = (
            ("mfa_executable", args.mfa_executable, None),
            ("mfa_dictionary", args.dictionary, resource_hashes["dictionary"]),
            (
                "mfa_acoustic_model",
                args.acoustic_model,
                resource_hashes["acoustic_model"],
            ),
            ("mfa_g2p_model", args.g2p_model, resource_hashes["g2p_model"]),
        )
        for label, path, expected in resources:
            assert path is not None
            check_file(checks, label, path, expected)

        assert args.raw_dir is not None
        aligned_dir = args.raw_dir / "\u5bf9\u9f50\u7248\u672c"
        unaligned_dir = args.raw_dir / "\u672a\u5bf9\u9f50\u7248\u672c"
        aligned_pickles = sorted(aligned_dir.glob("*.pkl"))
        unaligned_pickles = sorted(unaligned_dir.glob("*.pkl"))
        videos = sorted((aligned_dir / "videos").glob("*.mp4"))
        raw_data_check = {
            "label": "attachment4_raw_inventory",
            "aligned_pickles": len(aligned_pickles),
            "unaligned_pickles": len(unaligned_pickles),
            "videos": len(videos),
            "passed": (
                len(aligned_pickles)
                == len(unaligned_pickles)
                == len(videos)
                == int(protocol["attachment4"]["expected_samples"])
            ),
        }
        checks.append(raw_data_check)

    tool_checks: list[dict[str, Any]] = []
    if args.scope == "attachment4":
        for tool in ("ffmpeg", "ffprobe"):
            resolved = shutil.which(tool)
            tool_checks.append(
                {
                    "tool": tool,
                    "file": Path(resolved).name if resolved else None,
                    "passed": resolved is not None,
                }
            )
        assert args.mfa_executable is not None
        try:
            version = subprocess.check_output(
                [str(args.mfa_executable), "version"], text=True
            ).strip()
        except Exception as exc:
            version = f"ERROR:{type(exc).__name__}:{exc}"
        tool_checks.append(
            {
                "tool": "mfa",
                "file": args.mfa_executable.name,
                "version": version,
                "expected_version": protocol["alignment"]["mfa_version"],
                "passed": version == protocol["alignment"]["mfa_version"],
            }
        )

    import_checks: list[dict[str, Any]] = []
    for package in ("numpy", "torch", "transformers"):
        try:
            module = importlib.import_module(package)
            import_checks.append(
                {
                    "package": package,
                    "version": str(getattr(module, "__version__", "unknown")),
                    "passed": True,
                }
            )
        except Exception as exc:
            import_checks.append(
                {
                    "package": package,
                    "error": f"{type(exc).__name__}: {exc}",
                    "passed": False,
                }
            )

    failures = [
        item for item in [*checks, *tool_checks, *import_checks] if not item["passed"]
    ]
    result = {
        "schema": "q3v2-preflight-v1",
        "scope": args.scope,
        "status": "PASS" if not failures else "FAIL",
        "protocol_sha256": sha256_file(args.protocol),
        "checks": checks,
        "tools": tool_checks,
        "imports": import_checks,
        "failure_count": len(failures),
        "forbidden_selection_inputs": ["official_test", "attachment3", "attachment4"],
        "selection_input": "validation_only",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit("Q3 v2.2 preflight failed")


if __name__ == "__main__":
    main()
