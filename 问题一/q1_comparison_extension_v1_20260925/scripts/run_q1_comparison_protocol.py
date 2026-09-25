#!/usr/bin/env python3
"""Preflight, execute and validate the frozen Q1 comparison protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXTENSION_ROOT = HERE.parent
Q1_ROOT = EXTENSION_ROOT.parent
ORIGINAL_SCRIPTS = Q1_ROOT / "scripts"
DEFAULT_SPEC = EXTENSION_ROOT / "configs/q1_comparison_protocol.json"
FORMAL = Q1_ROOT / "artifacts/q1/server_final_100/final_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-root", type=Path, required=True, help="Directory containing E题数据/")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--stage", choices=("preflight", "plan", "git-screening", "version-comparison",
                                           "representation-probe", "validate", "all"), default="preflight")
    parser.add_argument("--current-q1-dir", type=Path, help="Current audited 第一问 directory for A-vs-B comparison")
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], log_path: Path, resource_csv: Path, candidate: str, stage: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    row = {
        "run_id": resource_csv.parent.name,
        "candidate": candidate,
        "stage": stage,
        "device": "cpu",
        "cold_or_warm": "cache-dependent",
        "wall_s": f"{elapsed:.6f}",
        "cpu_s": f"{after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime:.6f}",
        "peak_rss_mb": f"{after.ru_maxrss / divisor:.3f}",
        "peak_vram_mb": "",
        "output_bytes": "",
        "status": "completed" if process.returncode == 0 else "failed",
    }
    with resource_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writerow(row)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}: {' '.join(command)}")


def resolve_data(spec: dict[str, Any], e_root: Path) -> dict[str, Path]:
    return {
        key: (e_root / value["relative_path"]).resolve()
        for key, value in spec["dataset_contract"].items()
    }


def preflight(spec: dict[str, Any], e_root: Path, current: Path | None) -> dict[str, Any]:
    paths = resolve_data(spec, e_root)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})

    a1 = paths["attachment_1"]
    videos = sorted(a1.glob("*/*.mp4")) if a1.is_dir() else []
    expected_videos = spec["dataset_contract"]["attachment_1"]["expected_videos"]
    check("attachment_1_directory", a1.is_dir(), str(a1), "existing directory")
    check("attachment_1_videos", len(videos) == expected_videos, len(videos), expected_videos)
    label = a1 / spec["dataset_contract"]["attachment_1"]["required_file"]
    check("attachment_1_label", label.is_file(), str(label), "existing file")

    a2 = paths["attachment_2"]
    for name in spec["dataset_contract"]["attachment_2"]["required_files"]:
        check(f"attachment_2_{name}", (a2 / name).is_file(), str(a2 / name), "existing file")

    a3 = paths["attachment_3"]
    aligned = list((a3 / "对齐版本").glob("*.pkl"))
    unaligned = list((a3 / "未对齐版本").glob("*.pkl"))
    check("attachment_3_aligned_count", len(aligned) == 30, len(aligned), 30)
    check("attachment_3_unaligned_count", len(unaligned) == 30, len(unaligned), 30)
    check("attachment_4_directory", paths["attachment_4"].is_dir(), str(paths["attachment_4"]), "existing directory")

    required_formal = {
        "features": len(list((FORMAL / "features").glob("*.npz"))),
        "metadata": len(list((FORMAL / "metadata").glob("*.json"))),
        "alignment": sum(1 for line in (FORMAL / "q1_alignment.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()),
    }
    for name, count in required_formal.items():
        check(f"git_formal_{name}", count == 100, count, 100)

    if current is not None:
        current = current.resolve()
        current_counts = {
            "features": len(list((current / "outputs/features").glob("*.npz"))),
            "alignment": sum(1 for line in (current / "outputs/alignment.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()) if (current / "outputs/alignment.jsonl").is_file() else 0,
            "boundary_audit": int((current / "reports/boundary_audit.json").is_file()),
            "validation": int((current / "reports/validation.json").is_file()),
        }
        check("current_q1_features", current_counts["features"] == 100, current_counts["features"], 100)
        check("current_q1_alignment", current_counts["alignment"] == 100, current_counts["alignment"], 100)
        check("current_q1_boundary_audit", current_counts["boundary_audit"] == 1, current_counts["boundary_audit"], 1)
        check("current_q1_validation", current_counts["validation"] == 1, current_counts["validation"], 1)

    for executable in ("ffmpeg", "ffprobe"):
        check(f"tool_{executable}", shutil.which(executable) is not None, shutil.which(executable), "available")
    for module in ("numpy", "scipy", "sklearn", "cv2"):
        check(f"python_module_{module}", importlib.util.find_spec(module) is not None, bool(importlib.util.find_spec(module)), True)

    return {
        "protocol_version": spec["protocol_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "e_root": str(e_root.resolve()),
        "resolved_data": {key: str(value) for key, value in paths.items()},
        "current_q1_dir": str(current.resolve()) if current else None,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def commands(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, list[str]]:
    profile = spec["profiles"][args.profile]
    a1 = resolve_data(spec, args.e_root)["attachment_1"]
    work = args.work_dir.resolve()
    compare = [
        sys.executable, str(ORIGINAL_SCRIPTS / "compare_q1_methods.py"),
        "--csv", str(Q1_ROOT / "artifacts/q1/data_cleaning/qualified_samples.csv"),
        "--raw-root", str(a1),
        "--text-feature-dir", str(FORMAL),
        "--cache-dir", str(work / "cache"),
        "--output-dir", str(work / "git_screening"),
        "--probe-repeats", str(profile["probe_repeats"]),
    ]
    if profile["sample_limit"]:
        compare.extend(["--limit", str(profile["sample_limit"])])
    if args.overwrite_cache:
        compare.append("--overwrite-cache")
    result = {
        "git_screening": compare,
        "git_paired_statistics": [
            sys.executable, str(ORIGINAL_SCRIPTS / "analyze_q1_comparison.py"),
            "--metrics", str(work / "git_screening/comparison_metrics.csv"),
            "--output-dir", str(work / "git_screening"),
        ],
    }
    if args.current_q1_dir:
        result["version_comparison"] = [
            sys.executable, str(HERE / "compare_q1_versions.py"),
            "--current-q1-dir", str(args.current_q1_dir.resolve()),
            "--git-result-dir", str(FORMAL),
            "--output-dir", str(work / "version_comparison"),
        ]
        result["build_candidate_vectors"] = [
            sys.executable, str(HERE / "build_q1_probe_vectors.py"),
            "--current-q1-dir", str(args.current_q1_dir.resolve()),
            "--git-result-dir", str(FORMAL),
            "--labels-csv", str(Q1_ROOT / "artifacts/q1/data_cleaning/qualified_samples.csv"),
            "--output-dir", str(work / "candidate_vectors"),
        ]
        seeds = spec["random_seeds"][:1] if args.profile == "smoke" else spec["random_seeds"]
        result["nested_representation_probe"] = [
            sys.executable, str(HERE / "evaluate_q1_candidate_vectors.py"),
            "--candidate-dir", str(work / "candidate_vectors"),
            "--output-dir", str(work / "representation_probe"),
            "--seeds", *[str(seed) for seed in seeds],
            "--outer-folds", str(spec["statistical_protocol"]["outer_folds"]),
            "--inner-folds", str(spec["statistical_protocol"]["inner_folds"]),
        ]
    return result


def write_plan(spec: dict[str, Any], args: argparse.Namespace, command_map: dict[str, list[str]]) -> None:
    payload = {
        "protocol_version": spec["protocol_version"],
        "profile": args.profile,
        "commands": command_map,
        "selection_rule": spec["statistical_protocol"]["selection_rule"],
        "required_outputs": spec["required_outputs"],
    }
    dump(args.work_dir / "resolved_execution_plan.json", payload)
    script = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for name, command in command_map.items():
        script.extend([f"# {name}", shlex.join(command), ""])
    target = args.work_dir / "run_resolved_plan.sh"
    target.write_text("\n".join(script), encoding="utf-8")
    target.chmod(0o755)


def validate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    screening = args.work_dir / "git_screening"
    for name in ("comparison_metrics.csv", "comparison_summary.json", "comparison_report.md",
                 "pairwise_statistics.csv", "pairwise_statistics.json", "pairwise_statistics.md"):
        path = screening / name
        add(f"git_screening_{name}", path.is_file() and path.stat().st_size > 0, str(path))
    metrics = screening / "comparison_metrics.csv"
    if metrics.is_file():
        with metrics.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        add("git_screening_candidate_rows", len(rows) >= 2, len(rows))
        required = {"family", "method", "status", "probe_macro_f1", "probe_pearson", "probe_mae"}
        add("git_screening_metric_columns", required.issubset(rows[0]) if rows else False,
            sorted(rows[0]) if rows else [])
        add("git_screening_all_ok", all(row["status"] == "ok" for row in rows), [row["method"] for row in rows if row["status"] != "ok"])

    if args.current_q1_dir:
        version = args.work_dir / "version_comparison/version_comparison.json"
        add("version_comparison_json", version.is_file(), str(version))
        if version.is_file():
            data = json.loads(version.read_text(encoding="utf-8"))
            add("version_source_match", data.get("source_hash_match_count") == 100, data.get("source_hash_match_count"))
            add("version_text_match", data.get("text_match_count") == 100, data.get("text_match_count"))
            add("version_shared_endpoint_count", data.get("shared_endpoint_count", 0) > 0, data.get("shared_endpoint_count"))
        for name in ("probe_predictions.csv", "probe_metrics.csv", "probe_summary.json"):
            path = args.work_dir / "representation_probe" / name
            add(f"representation_probe_{name}", path.is_file() and path.stat().st_size > 0, str(path))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "checks": checks,
        "passed": bool(checks) and all(item["passed"] for item in checks),
    }


def write_manifest(work_dir: Path) -> None:
    files = sorted(path for path in work_dir.rglob("*") if path.is_file() and path.name != "run_manifest_sha256.txt")
    lines = [f"{sha256(path)}  {path.relative_to(work_dir)}" for path in files]
    (work_dir / "run_manifest_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def initialize_run_files(work_dir: Path, report: dict[str, Any]) -> None:
    environment = {
        key: report[key] for key in ("generated_at", "python", "platform", "executable")
    }
    environment["packages"] = {}
    for distribution in ("numpy", "scipy", "scikit-learn", "opencv-python"):
        try:
            environment["packages"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][distribution] = None
    environment["tools"] = {}
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        version = None
        if path:
            completed = subprocess.run([path, "-version"], capture_output=True, text=True, check=False)
            version = (completed.stdout or completed.stderr).splitlines()[0] if completed.returncode == 0 else None
        environment["tools"][executable] = {"path": path, "version": version}
    dump(work_dir / "environment.json", environment)
    resource = work_dir / "resource_usage.csv"
    if not resource.exists():
        resource.write_text(
            "run_id,candidate,stage,device,cold_or_warm,wall_s,cpu_s,peak_rss_mb,peak_vram_mb,output_bytes,status\n",
            encoding="utf-8",
        )
    failures = work_dir / "failures.jsonl"
    failures.touch(exist_ok=True)
    analysis = work_dir / "result_analysis.md"
    if not analysis.exists():
        analysis.write_text(
            "# 问题一模型与优化算法对比结果\n\n"
            "> 待全量实验完成后，按 `templates/问题一_对比实验结果分析模板.md` 填充；"
            "禁止用冒烟结果形成论文结论。\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    args.e_root = args.e_root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report = preflight(spec, args.e_root, args.current_q1_dir)
    dump(args.work_dir / "preflight.json", report)
    initialize_run_files(args.work_dir, report)
    command_map = commands(spec, args)
    write_plan(spec, args, command_map)
    if args.stage in ("preflight", "plan"):
        write_manifest(args.work_dir)
        print(json.dumps({"stage": args.stage, "preflight_passed": report["passed"], "work_dir": str(args.work_dir)}, ensure_ascii=False))
        return 0 if report["passed"] else 2
    if not report["passed"]:
        raise RuntimeError(f"Preflight failed; see {args.work_dir / 'preflight.json'}")
    if args.stage in ("git-screening", "all"):
        run(command_map["git_screening"], args.work_dir / "logs/git_screening.log",
            args.work_dir / "resource_usage.csv", "B_git_frozen", "component_screening")
        run(command_map["git_paired_statistics"], args.work_dir / "logs/git_paired_statistics.log",
            args.work_dir / "resource_usage.csv", "B_git_frozen", "paired_statistics")
    if args.stage in ("version-comparison", "all"):
        if "version_comparison" not in command_map:
            raise ValueError("--current-q1-dir is required for version-comparison")
        run(command_map["version_comparison"], args.work_dir / "logs/version_comparison.log",
            args.work_dir / "resource_usage.csv", "A_vs_B", "version_comparison")
    if args.stage in ("representation-probe", "all"):
        if "build_candidate_vectors" not in command_map:
            raise ValueError("--current-q1-dir is required for representation-probe")
        run(command_map["build_candidate_vectors"], args.work_dir / "logs/build_candidate_vectors.log",
            args.work_dir / "resource_usage.csv", "A_B_C", "build_candidate_vectors")
        run(command_map["nested_representation_probe"], args.work_dir / "logs/nested_representation_probe.log",
            args.work_dir / "resource_usage.csv", "A_B_C", "nested_representation_probe")
    if args.stage in ("validate", "all"):
        validation = validate_outputs(args)
        dump(args.work_dir / "validation.json", validation)
        if not validation["passed"]:
            raise RuntimeError(f"Output validation failed; see {args.work_dir / 'validation.json'}")
    write_manifest(args.work_dir)
    print(json.dumps({"stage": args.stage, "status": "completed", "work_dir": str(args.work_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
