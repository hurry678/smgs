#!/usr/bin/env python3
"""Run the isolated high-compute optimization built on the frozen DS M0 model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from q2hc_core import (
    DS_SCRIPTS,
    EXTENSION_ROOT,
    REPO_ROOT,
    apply_normalization,
    cache_member_predictions,
    choose_device,
    compact_existing_checkpoint,
    load_any_checkpoint,
    load_split,
    metrics,
    predict_model,
    read_json,
    sha256_file,
    train_job,
    write_json,
)
from q2hc_select import confirm_ensemble, select_ensemble


DEFAULT_CONFIG = EXTENSION_ROOT / "configs" / "protocol.json"


def _resolve_hashed_file(root: Path, relative_path: str, expected_sha256: str) -> Path:
    direct = root / relative_path
    if direct.is_file() and sha256_file(direct) == expected_sha256:
        return direct.resolve()
    candidates = []
    for path in root.rglob(Path(relative_path).name):
        if path.is_file() and sha256_file(path) == expected_sha256:
            candidates.append(path.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one file with sha256={expected_sha256} for {relative_path}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def preflight(
    config: Mapping[str, Any],
    config_path: Path,
    legacy_root: Path,
    work_dir: Path,
) -> Dict[str, Any]:
    previous_path = work_dir / "resolved_inputs.json"
    downstream_exists = any(
        (work_dir / name).exists()
        for name in (
            "baseline_replay",
            "search",
            "pool",
            "selection.json",
            "publication",
        )
    )
    if previous_path.is_file() and downstream_exists:
        previous_sha = read_json(previous_path).get("protocol_sha256")
        if previous_sha != sha256_file(config_path):
            raise RuntimeError(
                "cannot replace the protocol after execution started; use a new work directory"
            )
    failures = []
    script_checks = {}
    for relative, expected in config["integrity"]["ds_script_sha256"].items():
        path = REPO_ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        script_checks[relative] = {
            "path": str(path),
            "expected": expected,
            "actual": actual,
        }
        if actual != expected:
            failures.append(f"DS script hash mismatch: {relative}")

    data_path = legacy_root / "data" / "q2" / "q2_data.npz"
    if not data_path.is_file():
        matches = list(legacy_root.rglob("q2_data.npz"))
        data_path = matches[0] if len(matches) == 1 else data_path
    data_sha = sha256_file(data_path) if data_path.is_file() else None
    if data_sha != config["baseline"]["data_sha256"]:
        failures.append(
            f"q2_data.npz hash mismatch: expected {config['baseline']['data_sha256']}, got {data_sha}"
        )

    attachment_path = legacy_root / "data" / "q2" / "attachment3_aligned.npz"
    if not attachment_path.is_file():
        matches = list(legacy_root.rglob("attachment3_aligned.npz"))
        attachment_path = matches[0] if len(matches) == 1 else attachment_path

    model_dir = legacy_root / "models" / "all-MiniLM-L6-v2"
    model_weight = model_dir / config["integrity"]["model_weight_filename"]
    if not model_weight.is_file():
        matches = list(legacy_root.rglob(config["integrity"]["model_weight_filename"]))
        model_candidates = [
            path.parent for path in matches if "MiniLM" in str(path.parent)
        ]
        if len(model_candidates) == 1:
            model_dir = model_candidates[0]
            model_weight = model_dir / config["integrity"]["model_weight_filename"]
    if not model_weight.is_file():
        failures.append(f"MiniLM weight not found: {model_weight}")

    checkpoints = []
    for item in config["baseline"]["checkpoints"]:
        try:
            path = _resolve_hashed_file(
                legacy_root, item["relative_path"], item["sha256"]
            )
            checkpoints.append(
                {
                    "id": item["id"],
                    "path": str(path),
                    "sha256": item["sha256"],
                }
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    split_summary = {}
    if data_path.is_file() and data_sha == config["baseline"]["data_sha256"]:
        with np.load(data_path, allow_pickle=False) as archive:
            required = []
            for split in ("train", "valid"):
                required.extend(
                    [
                        f"{split}_ids",
                        f"{split}_text_bert",
                        f"{split}_audio",
                        f"{split}_vision",
                        f"{split}_labels",
                        f"{split}_regression",
                    ]
                )
            missing_keys = [key for key in required if key not in archive.files]
            if missing_keys:
                failures.append(f"q2_data.npz missing keys: {missing_keys}")
            else:
                split_summary = {
                    split: {
                        "n": int(len(archive[f"{split}_ids"])),
                        "text_bert_shape": list(archive[f"{split}_text_bert"].shape),
                        "audio_shape": list(archive[f"{split}_audio"].shape),
                        "vision_shape": list(archive[f"{split}_vision"].shape),
                    }
                    for split in ("train", "valid")
                }
    if not attachment_path.is_file():
        failures.append(f"attachment3_aligned.npz not found: {attachment_path}")

    result = {
        "schema": "q2-ds-hc-preflight-v1",
        "status": "passed" if not failures else "failed",
        "repo_root": str(REPO_ROOT),
        "extension_root": str(EXTENSION_ROOT),
        "legacy_root": str(legacy_root.resolve()),
        "protocol": str(config_path.resolve()),
        "protocol_sha256": sha256_file(config_path),
        "data": str(data_path.resolve()) if data_path.is_file() else str(data_path),
        "data_sha256": data_sha,
        "attachment3": str(attachment_path.resolve())
        if attachment_path.is_file()
        else str(attachment_path),
        "attachment3_sha256": sha256_file(attachment_path)
        if attachment_path.is_file()
        else None,
        "model_dir": str(model_dir.resolve()) if model_dir.is_dir() else str(model_dir),
        "model_weight_sha256": sha256_file(model_weight)
        if model_weight.is_file()
        else None,
        "baseline_checkpoints": checkpoints,
        "script_checks": script_checks,
        "splits": split_summary,
        "failures": failures,
    }
    write_json(work_dir / "resolved_inputs.json", result)
    if failures:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(failures))
    return result


def resolved_inputs(work_dir: Path) -> Dict[str, Any]:
    path = work_dir / "resolved_inputs.json"
    if not path.is_file():
        raise RuntimeError(f"run preflight first; missing {path}")
    result = read_json(path)
    if result.get("status") != "passed":
        raise RuntimeError(f"preflight is not passed: {path}")
    return result


def replay_baseline(
    config: Mapping[str, Any], work_dir: Path, device: str, force: bool
) -> Dict[str, Any]:
    resolved = resolved_inputs(work_dir)
    output_dir = work_dir / "baseline_replay"
    evaluation_path = output_dir / "valid_valid_evaluation.json"
    verification_path = output_dir / "verification.json"
    if verification_path.is_file() and not force:
        return read_json(verification_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(DS_SCRIPTS / "q2_eval_ensemble.py"),
        "--data",
        resolved["data"],
        "--checkpoints",
        *[item["path"] for item in resolved["baseline_checkpoints"]],
        "--model-dir",
        resolved["model_dir"],
        "--out-dir",
        str(output_dir),
        "--mask-seed",
        "2026",
        "--replicas",
        "5",
        "--device",
        device,
    ]
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(f"baseline replay failed; see {log_path}")
    evaluation = read_json(evaluation_path)
    observed = {
        "R_MAE": float(
            np.mean([row["c2_mean_mae"] for row in evaluation["conditions"]])
        ),
        "R_F1": float(
            np.mean([row["c2_mean_macro_f1"] for row in evaluation["conditions"]])
        ),
        "worst_condition_MAE": float(
            max(row["c2_mean_mae"] for row in evaluation["conditions"])
        ),
        "clean_accuracy": float(evaluation["full_c2"]["accuracy"]),
        "clean_macro_f1": float(evaluation["full_c2"]["macro_f1"]),
        "clean_MAE": float(evaluation["full_c2"]["mae"]),
    }
    expected = config["baseline"]["expected_valid"]
    differences = {key: observed[key] - float(expected[key]) for key in expected}
    passed = all(abs(value) <= 1e-6 for value in differences.values())
    result = {
        "schema": "q2-ds-hc-baseline-replay-v1",
        "status": "passed" if passed else "failed",
        "expected": expected,
        "observed": observed,
        "differences": differences,
        "tolerance": 1e-6,
        "evaluation": str(evaluation_path.resolve()),
        "command": command,
    }
    write_json(verification_path, result)
    if not passed:
        raise RuntimeError(f"baseline replay mismatch; see {verification_path}")
    return result


def _run_parallel(jobs: List[Dict[str, Any]], devices: Sequence[str]) -> None:
    if not devices:
        raise ValueError("at least one device is required")
    queue = list(jobs)
    active: List[Dict[str, Any]] = []
    while queue or active:
        while queue and len(active) < len(devices):
            job = queue.pop(0)
            marker = Path(job["marker"])
            if marker.is_file() and not job.get("force", False):
                continue
            occupied = {item["device"] for item in active}
            device = next(value for value in devices if value not in occupied)
            command = [value.replace("{device}", device) for value in job["command"]]
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            active.append(
                {
                    "process": process,
                    "handle": handle,
                    "name": job["name"],
                    "log": log_path,
                    "device": device,
                }
            )
        if not active:
            continue
        time.sleep(1.0)
        still_active = []
        for item in active:
            return_code = item["process"].poll()
            if return_code is None:
                still_active.append(item)
                continue
            item["handle"].close()
            if return_code != 0:
                for other in active:
                    if other is item:
                        continue
                    if other["process"].poll() is None:
                        other["process"].terminate()
                    other["handle"].close()
                raise RuntimeError(f"job {item['name']} failed; see {item['log']}")
        active = still_active


def run_search(
    config_path: Path,
    config: Mapping[str, Any],
    work_dir: Path,
    devices: Sequence[str],
    force: bool,
) -> None:
    resolved_inputs(work_dir)
    replay_path = work_dir / "baseline_replay" / "verification.json"
    if not replay_path.is_file() or read_json(replay_path).get("status") != "passed":
        raise RuntimeError("baseline replay must pass before search")
    jobs = []
    for arm in config["search"]["arms"]:
        for fold in range(int(config["search"]["folds"])):
            output = work_dir / "search" / arm["id"] / f"fold{fold}"
            jobs.append(
                {
                    "name": f"{arm['id']}/fold{fold}",
                    "marker": output / "result.json",
                    "log": output / "run.log",
                    "force": force,
                    "command": [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "train",
                        "--config",
                        str(config_path),
                        "--work-dir",
                        str(work_dir),
                        "--arm",
                        arm["id"],
                        "--seed",
                        str(7000 + fold),
                        "--fold",
                        str(fold),
                        "--device",
                        "{device}",
                        "--output-dir",
                        str(output),
                    ],
                }
            )
    _run_parallel(jobs, devices)


def select_arms(config: Mapping[str, Any], work_dir: Path) -> Dict[str, Any]:
    rows = []
    folds = int(config["search"]["folds"])
    for arm in config["search"]["arms"]:
        results = []
        for fold in range(folds):
            path = work_dir / "search" / arm["id"] / f"fold{fold}" / "result.json"
            if not path.is_file():
                raise RuntimeError(f"missing search result: {path}")
            results.append(read_json(path))
        robust = [result["robust"] for result in results]
        mean = {
            "R_MAE": float(np.mean([item["R_MAE"] for item in robust])),
            "R_F1": float(np.mean([item["R_F1"] for item in robust])),
            "worst_condition_MAE": float(
                np.mean([item["worst_condition_MAE"] for item in robust])
            ),
            "clean_MAE": float(np.mean([item["clean"]["mae"] for item in robust])),
            "clean_F1": float(np.mean([item["clean"]["macro_f1"] for item in robust])),
        }
        objective_weights = config["search"]["arm_ranking_objective_weights"]
        score = (
            float(objective_weights["R_MAE"]) * mean["R_MAE"]
            + float(objective_weights["one_minus_R_F1"]) * (1.0 - mean["R_F1"])
            + float(objective_weights["clean_MAE"]) * mean["clean_MAE"]
            + float(objective_weights["one_minus_clean_F1"]) * (1.0 - mean["clean_F1"])
            + float(objective_weights["worst_condition_MAE"])
            * mean["worst_condition_MAE"]
        )
        best_epochs = [int(result["training"]["best_epoch"]) + 1 for result in results]
        fixed_epochs = int(
            np.clip(
                round(float(np.median(best_epochs))),
                8,
                int(config["search"]["max_epochs"]),
            )
        )
        rows.append(
            {
                "arm_id": arm["id"],
                "score": float(score),
                "mean": mean,
                "best_epochs": best_epochs,
                "fixed_epochs": fixed_epochs,
                "fold_results": [
                    str(work_dir / "search" / arm["id"] / f"fold{fold}" / "result.json")
                    for fold in range(folds)
                ],
            }
        )
    rows.sort(key=lambda row: (row["score"], row["arm_id"]))
    selected = rows[: int(config["search"]["top_arms"])]
    if not any(row["arm_id"] == "A00_ds_exact" for row in selected):
        selected[-1] = next(row for row in rows if row["arm_id"] == "A00_ds_exact")
        selected.sort(key=lambda row: (row["score"], row["arm_id"]))
    result = {
        "schema": "q2-ds-hc-arm-selection-v1",
        "status": "selected_without_reading_official_valid_or_test",
        "ranking": rows,
        "selected": selected,
        "objective_weights": config["search"]["arm_ranking_objective_weights"],
        "selection_source": "grouped cross-validation inside original train only",
    }
    write_json(work_dir / "arm_selection.json", result)
    return result


def run_final_pool(
    config_path: Path,
    config: Mapping[str, Any],
    work_dir: Path,
    devices: Sequence[str],
    force: bool,
) -> None:
    arm_selection = read_json(work_dir / "arm_selection.json")
    jobs = []
    for arm in arm_selection["selected"]:
        for seed in config["search"]["final_seeds"]:
            output = work_dir / "pool" / arm["arm_id"] / f"seed{seed}"
            jobs.append(
                {
                    "name": f"{arm['arm_id']}/seed{seed}",
                    "marker": output / "result.json",
                    "log": output / "run.log",
                    "force": force,
                    "command": [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "train",
                        "--config",
                        str(config_path),
                        "--work-dir",
                        str(work_dir),
                        "--arm",
                        arm["arm_id"],
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(arm["fixed_epochs"]),
                        "--device",
                        "{device}",
                        "--output-dir",
                        str(output),
                    ],
                }
            )
    _run_parallel(jobs, devices)
    inventory = []
    for arm in arm_selection["selected"]:
        for seed in config["search"]["final_seeds"]:
            result_path = (
                work_dir / "pool" / arm["arm_id"] / f"seed{seed}" / "result.json"
            )
            result = read_json(result_path)
            inventory.append(
                {
                    "id": f"{arm['arm_id']}_seed{seed}",
                    "arm_id": arm["arm_id"],
                    "seed": int(seed),
                    "checkpoint": result["checkpoint"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "result": str(result_path.resolve()),
                }
            )
    write_json(
        work_dir / "pool_inventory.json",
        {
            "schema": "q2-ds-hc-pool-v1",
            "members": inventory,
        },
    )


def _all_member_records(work_dir: Path) -> List[Dict[str, Any]]:
    resolved = resolved_inputs(work_dir)
    records = [
        {
            "id": item["id"],
            "checkpoint": item["path"],
            "checkpoint_sha256": item["sha256"],
            "source": "ds_baseline",
        }
        for item in resolved["baseline_checkpoints"]
    ]
    pool_path = work_dir / "pool_inventory.json"
    if pool_path.is_file():
        records.extend(
            {
                **item,
                "source": "new_pool",
            }
            for item in read_json(pool_path)["members"]
        )
    return records


def run_cache_bank(
    config_path: Path,
    config: Mapping[str, Any],
    work_dir: Path,
    bank: str,
    devices: Sequence[str],
    force: bool,
) -> None:
    if bank == "select":
        records = _all_member_records(work_dir)
        mask_seed = int(config["selection"]["mask_seed"])
        replicas = int(config["selection"]["replicas"])
    elif bank == "confirm":
        selection = read_json(work_dir / "selection.json")
        required = set(selection["baseline"]["weights"]) | set(
            selection["selected"]["weights"]
        )
        records = [
            item for item in _all_member_records(work_dir) if item["id"] in required
        ]
        mask_seed = int(config["confirmation"]["mask_seed"])
        replicas = int(config["confirmation"]["replicas"])
    else:
        raise ValueError(f"unknown bank: {bank}")
    jobs = []
    for record in records:
        output = work_dir / "cache" / bank / f"{record['id']}.npz"
        metadata = output.with_suffix(".json")
        jobs.append(
            {
                "name": f"cache-{bank}/{record['id']}",
                "marker": metadata,
                "log": work_dir / "cache" / bank / f"{record['id']}.log",
                "force": force,
                "command": [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "cache-one",
                    "--config",
                    str(config_path),
                    "--work-dir",
                    str(work_dir),
                    "--member-id",
                    record["id"],
                    "--checkpoint",
                    record["checkpoint"],
                    "--bank",
                    bank,
                    "--device",
                    "{device}",
                    "--output",
                    str(output),
                ],
            }
        )
    _run_parallel(jobs, devices)
    write_json(
        work_dir / "cache" / bank / "inventory.json",
        {
            "schema": "q2-ds-hc-cache-inventory-v1",
            "bank": bank,
            "mask_seed": mask_seed,
            "replicas": replicas,
            "members": [
                read_json((work_dir / "cache" / bank / f"{record['id']}.json"))
                for record in records
            ],
        },
    )


def run_selection(config: Mapping[str, Any], work_dir: Path) -> Dict[str, Any]:
    inventory = read_json(work_dir / "cache" / "select" / "inventory.json")
    paths = [Path(item["cache"]) for item in inventory["members"]]
    return select_ensemble(
        config,
        paths,
        Path(resolved_inputs(work_dir)["data"]),
        work_dir / "selection.json",
    )


def run_confirmation(config: Mapping[str, Any], work_dir: Path) -> Dict[str, Any]:
    inventory = read_json(work_dir / "cache" / "confirm" / "inventory.json")
    paths = [Path(item["cache"]) for item in inventory["members"]]
    return confirm_ensemble(
        config,
        paths,
        Path(resolved_inputs(work_dir)["data"]),
        work_dir / "selection.json",
        work_dir / "confirmation.json",
    )


def freeze(
    config: Mapping[str, Any], config_path: Path, work_dir: Path, force: bool
) -> Dict[str, Any]:
    manifest_path = work_dir / "frozen" / "freeze_manifest.json"
    if manifest_path.is_file() and not force:
        return read_json(manifest_path)
    selection = read_json(work_dir / "selection.json")
    confirmation = read_json(work_dir / "confirmation.json")
    resolved = resolved_inputs(work_dir)
    candidate_passed = bool(confirmation["passed"])
    if candidate_passed:
        weights = selection["selected"]["weights"]
        candidate_records = selection["selected"]["members"]
        status = "candidate_frozen_without_test"
        version = "Q2-DS-HC-v2"
    else:
        weights = selection["baseline"]["weights"]
        candidate_records = {
            item["id"]: {
                "checkpoint": item["path"],
                "checkpoint_sha256": item["sha256"],
            }
            for item in resolved["baseline_checkpoints"]
        }
        status = "baseline_retained_without_new_test"
        version = "M3-ensemble9-retained"
    frozen_members = []
    for member_id, weight in sorted(weights.items()):
        source = Path(candidate_records[member_id]["checkpoint"])
        expected = candidate_records[member_id]["checkpoint_sha256"]
        if sha256_file(source) != expected:
            raise RuntimeError(f"checkpoint changed before freeze: {source}")
        destination = work_dir / "frozen" / "members" / f"{member_id}.pt"
        compact = compact_existing_checkpoint(source, destination)
        frozen_members.append(
            {
                "id": member_id,
                "weight": float(weight),
                **compact,
            }
        )
    code_paths = [
        config_path,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("q2hc_core.py"),
        Path(__file__).resolve().with_name("q2hc_select.py"),
        REPO_ROOT / "ds" / "scripts" / "q2_models.py",
        REPO_ROOT / "ds" / "scripts" / "q2_train.py",
    ]
    compact_bytes = sum(item["bytes"] for item in frozen_members)
    projected = (
        compact_bytes
        + int(config["publication"]["q1_known_bytes"])
        + int(config["publication"]["q3_reserved_bytes"])
    )
    limit = int(config["publication"]["combined_submission_byte_limit"])
    if projected > limit:
        raise RuntimeError(
            f"projected combined package exceeds limit: {projected} > {limit}"
        )
    result = {
        "schema": "q2-ds-hc-freeze-v1",
        "status": status,
        "model_version": version,
        "selected_from": "train-only CV plus validation selector, then independent validation confirmation",
        "confirmation_status": confirmation["status"],
        "publish_strategy": config["publication"]["strategy"],
        "ensemble_rule": "weighted arithmetic mean of probabilities and raw intensity",
        "members": frozen_members,
        "compact_checkpoint_bytes": compact_bytes,
        "projected_combined_bytes": projected,
        "combined_submission_byte_limit": limit,
        "shared_encoder": {
            "path_at_freeze": str(Path(resolved["model_dir"]).resolve()),
            "weight_filename": config["integrity"]["model_weight_filename"],
            "weight_sha256": resolved["model_weight_sha256"],
            "stored_once": True,
        },
        "inputs": {
            "q2_data": resolved["data"],
            "q2_data_sha256": resolved["data_sha256"],
            "attachment3": resolved["attachment3"],
            "attachment3_sha256": resolved["attachment3_sha256"],
        },
        "evidence": {
            "baseline_replay": str(
                (work_dir / "baseline_replay" / "verification.json").resolve()
            ),
            "arm_selection": str((work_dir / "arm_selection.json").resolve()),
            "ensemble_selection": str((work_dir / "selection.json").resolve()),
            "confirmation": str((work_dir / "confirmation.json").resolve()),
        },
        "code_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in code_paths
        },
        "test_policy": {
            "status_at_freeze": "not_run_for_this_version",
            "role": config["publication"]["test_role"],
            "requires_explicit_allow_flag": True,
            "may_change_frozen_choice": False,
        },
    }
    write_json(manifest_path, result)
    write_report(work_dir)
    return result


def _load_frozen_members(
    manifest: Mapping[str, Any],
    model_dir: Path,
    device: torch.device,
) -> Iterable[Tuple[float, Any, Dict[str, torch.Tensor]]]:
    for member in manifest["members"]:
        path = Path(member["compact"])
        if sha256_file(path) != member["compact_sha256"]:
            raise RuntimeError(f"frozen member hash mismatch: {path}")
        model, _, norm = load_any_checkpoint(path, model_dir, device)
        yield float(member["weight"]), model, norm


def _weighted_predict(
    manifest: Mapping[str, Any],
    model_dir: Path,
    data: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor]]:
    probability_sum = None
    raw_sum = None
    reference_norm = None
    indices = torch.arange(data["obs"].shape[0], device=device)
    for weight, model, norm in _load_frozen_members(manifest, model_dir, device):
        if reference_norm is None:
            reference_norm = norm
        else:
            for key in reference_norm:
                if not torch.allclose(
                    reference_norm[key], norm[key], atol=1e-6, rtol=1e-6
                ):
                    raise RuntimeError(
                        f"normalization mismatch in frozen member: {key}"
                    )
        logits, raw = predict_model(model, data, data["obs"], indices)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if probability_sum is None:
            probability_sum = weight * probabilities.astype(np.float64)
            raw_sum = weight * raw.astype(np.float64)
        else:
            probability_sum += weight * probabilities
            raw_sum += weight * raw
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if probability_sum is None or raw_sum is None or reference_norm is None:
        raise RuntimeError("frozen manifest contains no members")
    return (
        probability_sum.astype(np.float32),
        raw_sum.astype(np.float32),
        reference_norm,
    )


def _load_attachment(
    path: Path, norm: Mapping[str, torch.Tensor], device: torch.device
) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        text_bert = torch.from_numpy(
            np.asarray(archive["text_bert"], dtype=np.int64)
        ).to(device)
        audio = torch.from_numpy(np.asarray(archive["audio"], dtype=np.float32)).to(
            device
        )
        vision = torch.from_numpy(np.asarray(archive["vision"], dtype=np.float32)).to(
            device
        )
        filenames = [str(value) for value in archive["filenames"].tolist()]
    text_obs = text_bert[:, 1] > 0
    audio_obs = audio.abs().amax(dim=-1) > 1e-8
    vision_obs = vision.abs().amax(dim=-1) > 1e-8
    observed = torch.stack([text_obs, audio_obs, vision_obs], dim=1)
    audio = (audio - norm["audio_mean"]) / norm["audio_std"]
    vision = (vision - norm["vision_mean"]) / norm["vision_std"]
    audio = torch.where(audio_obs.unsqueeze(-1), audio, torch.zeros_like(audio))
    vision = torch.where(vision_obs.unsqueeze(-1), vision, torch.zeros_like(vision))
    return {
        "filenames": filenames,
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "obs": observed,
    }


def publish(
    work_dir: Path, device_name: str, allow_post_freeze: bool
) -> Dict[str, Any]:
    if not allow_post_freeze:
        raise RuntimeError(
            "publish requires --allow-post-freeze after reviewing freeze_manifest.json"
        )
    publication_dir = work_dir / "publication"
    summary_path = publication_dir / "publication_summary.json"
    if summary_path.is_file():
        return read_json(summary_path)
    manifest_path = work_dir / "frozen" / "freeze_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["test_policy"]["status_at_freeze"] != "not_run_for_this_version":
        raise RuntimeError("freeze manifest does not prove pre-test freeze")
    resolved = resolved_inputs(work_dir)
    device = choose_device(device_name)
    model_dir = Path(resolved["model_dir"])

    first_member = Path(manifest["members"][0]["compact"])
    reference_model, _, norm = load_any_checkpoint(first_member, model_dir, device)
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_path = publication_dir / "test_evaluation.json"
    if test_path.is_file():
        test_result = read_json(test_path)
    else:
        with np.load(resolved["data"], allow_pickle=False) as archive:
            test_data = load_split(archive, "test", device)
        apply_normalization(test_data, norm)
        probabilities, raw, _ = _weighted_predict(
            manifest, model_dir, test_data, device
        )
        logits = np.log(np.clip(probabilities, 1e-9, 1.0))
        labels = test_data["labels"].detach().cpu().numpy()
        regression = test_data["regression"].detach().cpu().numpy()
        test_result = {
            "schema": "q2-ds-hc-test-v1",
            "role": "post-freeze descriptive audit only; never used for selection",
            "model_version": manifest["model_version"],
            "n": int(len(labels)),
            "metrics_C2": metrics(logits, raw, labels, regression, "c2"),
            "ids": test_data["ids"],
            "probabilities": probabilities.tolist(),
            "raw_intensity": raw.tolist(),
            "freeze_manifest": str(manifest_path.resolve()),
            "freeze_manifest_sha256": sha256_file(manifest_path),
        }
        write_json(test_path, test_result)
        np.savez_compressed(
            publication_dir / "test_predictions.npz",
            ids=np.asarray(test_data["ids"]),
            probabilities=probabilities,
            raw=raw,
        )

    attachment = _load_attachment(Path(resolved["attachment3"]), norm, device)
    attachment_probabilities, attachment_raw, _ = _weighted_predict(
        manifest,
        model_dir,
        attachment,
        device,
    )
    predicted = attachment_probabilities.argmax(axis=1)
    intensity = np.zeros_like(attachment_raw, dtype=np.float32)
    intensity[predicted == 2] = np.maximum(attachment_raw[predicted == 2], 1e-4)
    intensity[predicted == 0] = np.minimum(attachment_raw[predicted == 0], -1e-4)
    polarity = {0: "Negative", 1: "Neutral", 2: "Positive"}
    rows = []
    for index, filename in enumerate(attachment["filenames"]):
        rows.append(
            {
                "sample_id": filename.rsplit(".", 1)[0],
                "原文件名": filename,
                "polarity": polarity[int(predicted[index])],
                "intensity": float(intensity[index]),
                "probabilities": attachment_probabilities[index].tolist(),
                "raw_intensity": float(attachment_raw[index]),
            }
        )
    write_json(
        publication_dir / "attachment3_predictions.json",
        {
            "schema": "q2-ds-hc-attachment3-v1",
            "role": "frozen inference only",
            "model_version": manifest["model_version"],
            "rows": rows,
        },
    )
    csv_path = publication_dir / "attachment3_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "原文件名", "polarity", "intensity"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "原文件名": row["原文件名"],
                    "polarity": row["polarity"],
                    "intensity": f"{row['intensity']:.6f}",
                }
            )
    result = {
        "schema": "q2-ds-hc-publication-v1",
        "status": "completed_after_freeze",
        "test": {
            "metrics_C2": test_result["metrics_C2"],
            "path": str((publication_dir / "test_evaluation.json").resolve()),
        },
        "attachment3": {
            "n": len(rows),
            "polarity_counts": {
                label: sum(row["polarity"] == label for row in rows)
                for label in polarity.values()
            },
            "csv": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
        },
        "selection_unchanged_after_test": True,
    }
    write_json(summary_path, result)
    write_report(work_dir)
    return result


def _metric_line(name: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {float(summary['R_MAE']):.6f} | "
        f"{float(summary['R_F1']):.6f} | "
        f"{float(summary['worst_condition_MAE']):.6f} | "
        f"{float(summary['clean']['mae']):.6f} | "
        f"{float(summary['clean']['macro_f1']):.6f} |"
    )


def write_report(work_dir: Path) -> Path:
    report_path = work_dir / "REPORT.md"
    lines = [
        "# Q2 DS High-Compute v2 Run Report",
        "",
        f"- Work directory: `{work_dir}`",
    ]
    replay_path = work_dir / "baseline_replay" / "verification.json"
    if replay_path.is_file():
        replay = read_json(replay_path)
        lines.append(f"- Baseline replay: `{replay['status']}`")
    selection_path = work_dir / "selection.json"
    confirmation_path = work_dir / "confirmation.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        lines.extend(
            [
                f"- Selected member count: `{len(selection['selected']['weights'])}`",
                f"- Selector gate: `{selection['selected']['passes_selector_gates']}`",
            ]
        )
    if confirmation_path.is_file():
        confirmation = read_json(confirmation_path)
        lines.extend(
            [
                f"- Confirmation: `{confirmation['status']}`",
                f"- Frozen choice: `{confirmation['frozen_choice']}`",
                "",
                "| Confirmation partition | R_MAE | R_F1 | Worst MAE | Clean MAE | Clean F1 |",
                "|---|---:|---:|---:|---:|---:|",
                _metric_line("Baseline", confirmation["baseline"]),
                _metric_line("Candidate", confirmation["candidate"]),
                "",
                "| Full valid, descriptive | R_MAE | R_F1 | Worst MAE | Clean MAE | Clean F1 |",
                "|---|---:|---:|---:|---:|---:|",
                _metric_line(
                    "Baseline",
                    confirmation["full_valid_descriptive_after_gate"]["baseline"],
                ),
                _metric_line(
                    "Candidate",
                    confirmation["full_valid_descriptive_after_gate"]["candidate"],
                ),
            ]
        )
    training_results = sorted(work_dir.glob("search/*/fold*/result.json"))
    training_results += sorted(work_dir.glob("pool/*/seed*/result.json"))
    if training_results:
        records = [read_json(path) for path in training_results]
        total_seconds = sum(float(record["training"]["seconds"]) for record in records)
        peak_gpu = max(int(record["training"]["peak_gpu_bytes"]) for record in records)
        lines.extend(
            [
                "",
                "## Compute",
                "",
                f"- Completed training jobs: `{len(records)}`",
                f"- Sum of per-job training time: `{total_seconds / 3600.0:.3f} GPU-hours`",
                f"- Maximum recorded allocated GPU memory: `{peak_gpu / (1024**3):.3f} GiB`",
            ]
        )
    freeze_path = work_dir / "frozen" / "freeze_manifest.json"
    if freeze_path.is_file():
        frozen = read_json(freeze_path)
        lines.extend(
            [
                "",
                "## Freeze",
                "",
                f"- Status: `{frozen['status']}`",
                f"- Model version: `{frozen['model_version']}`",
                f"- Compact member bytes: `{frozen['compact_checkpoint_bytes']}`",
                f"- Projected combined bytes: `{frozen['projected_combined_bytes']}`",
                f"- Test status at freeze: `{frozen['test_policy']['status_at_freeze']}`",
            ]
        )
    publication_path = work_dir / "publication" / "publication_summary.json"
    if publication_path.is_file():
        publication = read_json(publication_path)
        test = publication["test"]["metrics_C2"]
        lines.extend(
            [
                "",
                "## Post-Freeze Publication",
                "",
                f"- Test Accuracy: `{test['accuracy']:.6f}`",
                f"- Test Macro-F1: `{test['macro_f1']:.6f}`",
                f"- Test MAE: `{test['mae']:.6f}`",
                f"- Test Pearson: `{test['pearson']:.6f}`",
                f"- Attachment 3 CSV SHA-256: `{publication['attachment3']['csv_sha256']}`",
                "- Test and attachment 3 did not change the frozen selection.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_manifest(extension_root: Path) -> Dict[str, Any]:
    excluded = {"MANIFEST_SHA256.txt"}
    files = sorted(
        path
        for path in extension_root.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(extension_root)}" for path in files
    ]
    manifest_path = extension_root / "MANIFEST_SHA256.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"files": len(files), "manifest": str(manifest_path)}


def _stages_all(
    args: argparse.Namespace,
    config_path: Path,
    config: Mapping[str, Any],
    devices: Sequence[str],
) -> None:
    preflight(config, config_path, args.legacy_root, args.work_dir)
    replay_baseline(config, args.work_dir, devices[0], args.force)
    run_search(config_path, config, args.work_dir, devices, args.force)
    select_arms(config, args.work_dir)
    run_final_pool(config_path, config, args.work_dir, devices, args.force)
    run_cache_bank(config_path, config, args.work_dir, "select", devices, args.force)
    run_selection(config, args.work_dir)
    run_cache_bank(config_path, config, args.work_dir, "confirm", devices, args.force)
    run_confirmation(config, args.work_dir)
    freeze(config, config_path, args.work_dir, args.force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "replay-baseline",
            "search",
            "select-arms",
            "train-pool",
            "cache-select",
            "select-ensemble",
            "cache-confirm",
            "confirm",
            "freeze",
            "publish",
            "all",
            "train",
            "cache-one",
            "manifest",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--legacy-root", type=Path, default=Path("/data2/hy/cts/e_problem")
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/data2/hy/q2_runs/q2_ds_hc_v2_20260925")
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-post-freeze", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--member-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bank", choices=["select", "confirm"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    resolved_path = work_dir / "resolved_inputs.json"
    if args.stage != "preflight" and resolved_path.is_file():
        recorded_protocol = read_json(resolved_path).get("protocol_sha256")
        current_protocol = sha256_file(config_path)
        if recorded_protocol != current_protocol:
            raise RuntimeError(
                "protocol changed after preflight; use a new versioned work directory"
            )
    prepublication_stages = {
        "search",
        "select-arms",
        "train-pool",
        "cache-select",
        "select-ensemble",
        "cache-confirm",
        "confirm",
        "freeze",
        "all",
        "train",
        "cache-one",
    }
    if (
        args.stage in prepublication_stages
        and (work_dir / "publication" / "test_evaluation.json").is_file()
    ):
        raise RuntimeError(
            "this work directory already contains post-freeze test output; "
            "use a new versioned work directory instead of changing selection"
        )

    if args.stage == "preflight":
        result = preflight(config, config_path, args.legacy_root.resolve(), work_dir)
    elif args.stage == "replay-baseline":
        result = replay_baseline(config, work_dir, args.device, args.force)
    elif args.stage == "search":
        run_search(config_path, config, work_dir, devices, args.force)
        result = {"status": "completed", "stage": "search"}
    elif args.stage == "select-arms":
        result = select_arms(config, work_dir)
    elif args.stage == "train-pool":
        run_final_pool(config_path, config, work_dir, devices, args.force)
        result = {"status": "completed", "stage": "train-pool"}
    elif args.stage == "cache-select":
        run_cache_bank(config_path, config, work_dir, "select", devices, args.force)
        result = {"status": "completed", "stage": "cache-select"}
    elif args.stage == "select-ensemble":
        result = run_selection(config, work_dir)
    elif args.stage == "cache-confirm":
        run_cache_bank(config_path, config, work_dir, "confirm", devices, args.force)
        result = {"status": "completed", "stage": "cache-confirm"}
    elif args.stage == "confirm":
        result = run_confirmation(config, work_dir)
    elif args.stage == "freeze":
        result = freeze(config, config_path, work_dir, args.force)
    elif args.stage == "publish":
        result = publish(work_dir, args.device, args.allow_post_freeze)
    elif args.stage == "all":
        _stages_all(args, config_path, config, devices)
        result = {
            "status": "frozen",
            "freeze_manifest": str(work_dir / "frozen" / "freeze_manifest.json"),
            "next": "review freeze manifest, then run publish with --allow-post-freeze",
        }
    elif args.stage == "train":
        if args.arm is None or args.seed is None or args.output_dir is None:
            raise SystemExit("train requires --arm, --seed and --output-dir")
        resolved = resolved_inputs(work_dir)
        result = train_job(
            config,
            Path(resolved["data"]),
            Path(resolved["model_dir"]),
            args.arm,
            args.seed,
            args.device,
            args.output_dir,
            fold=args.fold,
            fixed_epochs=args.epochs,
        )
    elif args.stage == "cache-one":
        if (
            args.member_id is None
            or args.checkpoint is None
            or args.bank is None
            or args.output is None
        ):
            raise SystemExit(
                "cache-one requires --member-id, --checkpoint, --bank and --output"
            )
        resolved = resolved_inputs(work_dir)
        bank_cfg = (
            config["selection"] if args.bank == "select" else config["confirmation"]
        )
        result = cache_member_predictions(
            args.checkpoint,
            args.member_id,
            Path(resolved["data"]),
            Path(resolved["model_dir"]),
            args.output,
            int(bank_cfg["mask_seed"]),
            int(bank_cfg["replicas"]),
            choose_device(args.device),
        )
        write_json(args.output.with_suffix(".json"), result)
    else:
        result = generate_manifest(EXTENSION_ROOT)

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
