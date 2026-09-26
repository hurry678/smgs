#!/usr/bin/env python3
"""Select and freeze Q3 v2.2 parameters from symmetric validation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CST = timezone(timedelta(hours=8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_report_spec(value: str) -> tuple[int, Path]:
    try:
        window_text, path_text = value.split("=", 1)
        window = int(window_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "report must use WINDOW=/path/to/q3_validation_report.json"
        ) from exc
    if window <= 0:
        raise argparse.ArgumentTypeError("window size must be positive")
    return window, Path(path_text)


def load_candidate(
    window: int,
    path: Path,
    expected_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    if report.get("mode") != "validation":
        raise ValueError(f"{path}: not a validation report")
    if int(protocol.get("window_size", -1)) != window:
        raise ValueError(f"{path}: window-size mismatch")
    if int(protocol.get("stride", -1)) != window:
        raise ValueError(f"{path}: stride must equal window size")
    if int(protocol.get("n_selected", -1)) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} validation samples")
    sample_ids = report.get("selected_sample_ids", [])
    if len(sample_ids) != expected_n or len(set(sample_ids)) != expected_n:
        raise ValueError(f"{path}: validation sample coverage is incomplete")
    freeze_check = report.get("model", {}).get("freeze_manifest_check", {})
    if freeze_check.get("checked") is not True or freeze_check.get("ok") is not True:
        raise ValueError(f"{path}: frozen checkpoint verification did not pass")
    fidelity = report.get("local_occlusion", {}).get("top_support_fidelity", {})
    if (
        fidelity.get("comparison_schema") != "q3v2.2-symmetric-top-support-v1"
        or fidelity.get("control_selection_symmetric") is not True
    ):
        raise ValueError(f"{path}: top-support comparison is not symmetric v2.2")
    random_result = fidelity.get("top_support_vs_random_contiguous", {})
    scatter_result = fidelity.get("top_support_vs_point_scatter", {})
    random_ci = random_result.get("bootstrap_95_ci", [])
    scatter_ci = scatter_result.get("bootstrap_95_ci", [])
    if len(random_ci) != 2 or len(scatter_ci) != 2:
        raise ValueError(f"{path}: clustered bootstrap intervals are missing")
    score = float(fidelity["mean_budget_normalized_top_support"])
    eligible = float(random_ci[0]) > 0.0 and float(scatter_ci[0]) > 0.0
    summary = {
        "window_size": window,
        "stride": window,
        "report": f"window_{window}/{path.name}",
        "report_sha256": sha256_file(path),
        "mean_budget_normalized_top_support": score,
        "mean_budget_normalized_top_random_contiguous": float(
            fidelity["mean_budget_normalized_top_random_contiguous"]
        ),
        "mean_budget_normalized_top_point_scatter": float(
            fidelity["mean_budget_normalized_top_point_scatter"]
        ),
        "top_support_vs_random_contiguous": random_result,
        "top_support_vs_point_scatter": scatter_result,
        "eligible_by_clustered_ci": bool(eligible),
    }
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        type=parse_report_spec,
        required=True,
        help="Repeat as WINDOW=/path/to/q3_validation_report.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=728)
    parser.add_argument("--fallback-window", type=int, default=5)
    parser.add_argument("--point-scan-n", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--q2-freeze-manifest", type=Path, required=True)
    parser.add_argument("--q2-version-manifest", type=Path, required=True)
    args = parser.parse_args()

    if len(args.report) < 2:
        raise SystemExit("at least two window candidates are required")
    windows = [window for window, _ in args.report]
    if len(set(windows)) != len(windows):
        raise SystemExit("window candidates must be unique")
    if args.fallback_window not in windows:
        raise SystemExit("fallback window is not among the candidates")

    summaries: list[dict[str, Any]] = []
    reference_ids: list[str] | None = None
    reference_metrics: dict[str, Any] | None = None
    for window, path in sorted(args.report):
        report, summary = load_candidate(window, path, args.expected_n)
        ids = [str(value) for value in report["selected_sample_ids"]]
        metrics = report["full_validation_metrics"]
        if reference_ids is None:
            reference_ids = ids
            reference_metrics = metrics
        elif ids != reference_ids:
            raise SystemExit(
                "window candidates do not use identical validation samples"
            )
        elif metrics != reference_metrics:
            raise SystemExit("window candidates have inconsistent predictor metrics")
        summaries.append(summary)

    eligible = [item for item in summaries if item["eligible_by_clustered_ci"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                float(item["mean_budget_normalized_top_support"]),
                -int(item["window_size"]),
            ),
        )
        reason = (
            "Both video-group clustered 95% CI lower bounds exceed zero; "
            "selected the largest budget-normalized top-support score."
        )
        selection_mode = "clustered_ci_eligible_best_score"
    else:
        selected = next(
            item
            for item in summaries
            if int(item["window_size"]) == args.fallback_window
        )
        reason = (
            "No candidate beat both same-budget controls with positive clustered "
            "95% CI lower bounds; used the preregistered middle-scale fallback."
        )
        selection_mode = "preregistered_middle_scale_fallback"

    selected_window = int(selected["window_size"])
    freeze = {
        "schema": "q3v2.2-explanation-freeze-v1",
        "version": "q3_explainability_v2_2_20260926",
        "status": "frozen",
        "frozen_at": datetime.now(CST).isoformat(timespec="seconds"),
        "selection_split": "validation_only",
        "attachment4_used_for_selection": False,
        "candidate_windows": sorted(windows),
        "selection_metric": (
            "mean independently selected budget-normalized continuous top support"
        ),
        "selection_gate": (
            "lower bound of video-group clustered 95% CI > 0 for symmetric "
            "independent-top comparison against both control patterns"
        ),
        "selection_mode": selection_mode,
        "selection_reason": reason,
        "selected_window_size": selected_window,
        "selected_stride": selected_window,
        "point_scan_n": int(args.point_scan_n),
        "top_k_per_modality": int(args.top_k),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "expected_validation_n": int(args.expected_n),
        "predictor": {
            "name": "M3-ensemble9 + C2",
            "q2_freeze_manifest": args.q2_freeze_manifest.name,
            "q2_freeze_manifest_sha256": sha256_file(args.q2_freeze_manifest),
            "q2_version_manifest": args.q2_version_manifest.name,
            "q2_version_manifest_sha256": sha256_file(args.q2_version_manifest),
            "full_validation_metrics": reference_metrics,
        },
        "candidates": summaries,
        "selected_report": selected["report"],
        "selected_report_sha256": selected["report_sha256"],
        "rules": {
            "modality": "exact three-player Shapley in integer bit-mask order",
            "primary": "largest positive predicted-class Shapley contribution",
            "primary_no_positive_fallback": (
                "largest absolute predicted-class Shapley contribution with "
                "support_modality=null"
            ),
            "local_rank": (
                "0.7 signed-class percentile + 0.3 class-oriented intensity percentile"
            ),
            "bootstrap_unit": "video_id",
            "top_support_comparison": (
                "each pattern independently maximizes signed class delta per "
                "actual masked position before paired comparison"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "selected_window_size": selected_window,
                "selection_mode": selection_mode,
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
