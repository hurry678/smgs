#!/usr/bin/env python3
"""Core training and inference utilities for the DS-based Q2 extension."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import sklearn
import torch
import torch.nn.functional as F


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(
    os.environ.get("Q2HC_REPO_ROOT", Path(__file__).resolve().parents[3])
).resolve()
DS_SCRIPTS = REPO_ROOT / "ds" / "scripts"
if str(DS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DS_SCRIPTS))

from q2_models import Q2Model  # noqa: E402
from q2_train import (  # noqa: E402
    AdamW,
    RATIOS,
    SUBSETS,
    apply_normalization,
    compute_norm,
    load_split,
    make_missing,
    metrics,
    predict_model,
    subset_name,
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unit(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_index(key: str, size: int) -> int:
    return min(size - 1, int(stable_unit(key) * size))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name != "cpu" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def group_key(sample_id: str, separator: str = "$_$") -> str:
    return sample_id.split(separator, 1)[0]


def fold_indices(
    ids: Sequence[str], folds: int, fold: int, seed: int, separator: str
) -> Tuple[np.ndarray, np.ndarray]:
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError(f"invalid fold {fold}/{folds}")
    assigned = np.asarray(
        [stable_index(f"{seed}|fold|{group_key(sid, separator)}", folds) for sid in ids]
    )
    holdout = np.flatnonzero(assigned == fold)
    train = np.flatnonzero(assigned != fold)
    if len(train) == 0 or len(holdout) == 0:
        raise RuntimeError("grouped fold assignment produced an empty partition")
    train_groups = {group_key(ids[i], separator) for i in train}
    holdout_groups = {group_key(ids[i], separator) for i in holdout}
    if train_groups & holdout_groups:
        raise RuntimeError("group leakage detected in cross-validation split")
    return train, holdout


def selector_indices(
    ids: Sequence[str], seed: int, fraction: float, separator: str
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("selector fraction must lie in (0, 1)")
    selector = np.asarray(
        [
            stable_unit(f"{seed}|selector|{group_key(sid, separator)}") < fraction
            for sid in ids
        ]
    )
    left = np.flatnonzero(selector)
    right = np.flatnonzero(~selector)
    if len(left) == 0 or len(right) == 0:
        raise RuntimeError("selector/confirmation split produced an empty partition")
    left_groups = {group_key(ids[i], separator) for i in left}
    right_groups = {group_key(ids[i], separator) for i in right}
    if left_groups & right_groups:
        raise RuntimeError("group leakage detected between selector and confirmation")
    return left, right


def slice_data(data: Mapping[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    device = data["labels"].device
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "ids":
            result[key] = [value[int(i)] for i in indices]
        elif torch.is_tensor(value):
            result[key] = value[idx].clone()
        else:
            result[key] = value
    return result


def merged_arm(config: Mapping[str, Any], arm_id: str) -> Dict[str, Any]:
    search = config["search"]
    matches = [arm for arm in search["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise KeyError(f"unknown arm: {arm_id}")
    merged = dict(search["defaults"])
    merged.update(matches[0])
    return merged


def runtime_fingerprint(
    config: Mapping[str, Any], data_path: Path, model_dir: Path
) -> Dict[str, Any]:
    source_paths = [
        Path(__file__).resolve(),
        EXTENSION_ROOT / "scripts" / "q2hc_select.py",
        EXTENSION_ROOT / "scripts" / "run_q2hc.py",
    ]
    source_paths.extend(
        REPO_ROOT / relative for relative in config["integrity"]["ds_script_sha256"]
    )
    return {
        "protocol_sha256": config.get("_runtime_protocol_sha256"),
        "data_sha256": sha256_file(data_path),
        "model_weight_sha256": sha256_file(
            model_dir / config["integrity"]["model_weight_filename"]
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_paths
        },
    }


def _class_weights(labels: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels, minlength=3).float().clamp_min(1.0)
    if mode == "inverse":
        return counts.sum() / (3.0 * counts)
    if mode == "sqrt_inverse":
        weights = counts.rsqrt()
        return weights / weights.mean()
    raise ValueError(f"unknown class weighting: {mode}")


def _weighted_subset(
    sample_id: str, epoch: int, seed: int, policy: str
) -> Tuple[int, ...]:
    if policy == "uniform":
        return SUBSETS[
            stable_index(f"{sample_id}|uniform|{epoch}|{seed}", len(SUBSETS))
        ]
    if policy == "text_focus":
        with_text = [(0,), (0, 1), (0, 2), (0, 1, 2)]
        without_text = [(1,), (2,), (1, 2)]
        u = stable_unit(f"{sample_id}|text-focus|{epoch}|{seed}")
        if u < 0.70:
            return with_text[
                stable_index(f"{sample_id}|text-choice|{epoch}|{seed}", len(with_text))
            ]
        return without_text[
            stable_index(f"{sample_id}|other-choice|{epoch}|{seed}", len(without_text))
        ]
    raise ValueError(f"unknown non-DS mask policy: {policy}")


def training_missing(
    obs: torch.Tensor,
    maskable_obs: torch.Tensor,
    ids: Sequence[str],
    epoch: int,
    seed: int,
    policy: str,
) -> torch.Tensor:
    if policy == "ds":
        return make_missing(obs, ids, maskable_obs, epoch=epoch, seed=seed)
    visible = obs.clone()
    for i, sample_id in enumerate(ids):
        subset = _weighted_subset(sample_id, epoch, seed, policy)
        ratio = RATIOS[stable_index(f"{sample_id}|ratio|{epoch}|{seed}", len(RATIOS))]
        visible[i : i + 1] = make_missing(
            obs[i : i + 1],
            [sample_id],
            maskable_obs[i : i + 1],
            subset=subset,
            ratio=ratio,
            epoch=epoch,
            seed=seed,
        )
    return visible


def _supervised_loss(
    logits: torch.Tensor,
    raw: torch.Tensor,
    labels: torch.Tensor,
    regression: torch.Tensor,
    class_weight: torch.Tensor | None,
    arm: Mapping[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(logits, labels, weight=class_weight)
    huber = F.smooth_l1_loss(raw, regression, beta=float(arm["huber_beta"]))
    loss = (
        float(arm["classification_weight"]) * ce
        + float(arm["regression_weight"]) * huber
    )
    return loss, {"ce": float(ce.detach().cpu()), "huber": float(huber.detach().cpu())}


def _trainable_state(model: Q2Model) -> Dict[str, torch.Tensor]:
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key in trainable
    }


def _load_trainable_state(model: Q2Model, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in state.items():
            parameters[key].copy_(value.to(parameters[key].device))


def _swap_trainable_state(
    model: Q2Model, state: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    old = _trainable_state(model)
    _load_trainable_state(model, state)
    return old


def _learning_rate(
    base_lr: float, step: int, total_steps: int, scheduler: str, warmup_fraction: float
) -> float:
    if scheduler == "constant":
        return base_lr
    if scheduler != "cosine":
        raise ValueError(f"unknown scheduler: {scheduler}")
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return base_lr * float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _condition_predictions(
    model: Q2Model,
    data: Dict[str, Any],
    subset: Tuple[int, ...],
    ratio: float,
    replica: int,
    mask_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if subset:
        visible = make_missing(
            data["obs"],
            data["ids"],
            data["maskable_obs"],
            subset=subset,
            ratio=ratio,
            replica=replica,
            seed=mask_seed,
        )
    else:
        visible = data["obs"]
    return predict_model(model, data, visible)


def evaluate_model(
    model: Q2Model,
    data: Dict[str, Any],
    mask_seed: int,
    replicas: int,
) -> Dict[str, Any]:
    labels = data["labels"].detach().cpu().numpy()
    regression = data["regression"].detach().cpu().numpy()
    clean_logits, clean_raw = _condition_predictions(model, data, (), 0.0, 0, mask_seed)
    clean = metrics(clean_logits, clean_raw, labels, regression, "c2")
    rows: List[Dict[str, Any]] = []
    for subset in SUBSETS:
        for ratio in RATIOS:
            replica_metrics = []
            for replica in range(replicas):
                logits, raw = _condition_predictions(
                    model, data, subset, ratio, replica, mask_seed
                )
                replica_metrics.append(metrics(logits, raw, labels, regression, "c2"))
            rows.append(
                {
                    "subset": subset_name(subset),
                    "ratio": float(ratio),
                    "replicas": replica_metrics,
                    "mean_mae": float(np.mean([row["mae"] for row in replica_metrics])),
                    "mean_macro_f1": float(
                        np.mean([row["macro_f1"] for row in replica_metrics])
                    ),
                }
            )
    return {
        "clean": clean,
        "conditions": rows,
        "R_MAE": float(np.mean([row["mean_mae"] for row in rows])),
        "R_F1": float(np.mean([row["mean_macro_f1"] for row in rows])),
        "worst_condition_MAE": float(max(row["mean_mae"] for row in rows)),
        "worst_condition_F1": float(min(row["mean_macro_f1"] for row in rows)),
        "mask_seed": int(mask_seed),
        "replicas": int(replicas),
    }


def screen_model(
    model: Q2Model, data: Dict[str, Any], mask_seed: int
) -> Dict[str, float]:
    labels = data["labels"].detach().cpu().numpy()
    regression = data["regression"].detach().cpu().numpy()
    conditions = [
        ((), 0.0),
        ((0,), 0.3),
        ((0,), 0.5),
        ((1, 2), 0.5),
        ((0, 1, 2), 0.3),
    ]
    maes, f1s = [], []
    for subset, ratio in conditions:
        logits, raw = _condition_predictions(model, data, subset, ratio, 0, mask_seed)
        block = metrics(logits, raw, labels, regression, "c2")
        maes.append(block["mae"])
        f1s.append(block["macro_f1"])
    mean_mae = float(np.mean(maes))
    mean_f1 = float(np.mean(f1s))
    return {
        "score": mean_mae + 0.1 * (1.0 - mean_f1),
        "screen_MAE": mean_mae,
        "screen_F1": mean_f1,
    }


def train_m0(
    train: Dict[str, Any],
    validation: Dict[str, Any] | None,
    model_dir: Path,
    arm: Mapping[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    min_delta: float,
    screen_mask_seed: int,
    device: torch.device,
) -> Tuple[Q2Model, Dict[str, Any]]:
    set_seed(seed)
    model = Q2Model(
        "M0", model_dir, hidden=int(arm["hidden"]), dropout=float(arm["dropout"])
    ).to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = AdamW(
        trainable,
        lr=float(arm["lr"]),
        weight_decay=float(arm["weight_decay"]),
    )
    class_weight = _class_weights(train["labels"], str(arm["class_weighting"]))
    if class_weight is not None:
        class_weight = class_weight.to(device)
    batch_size = int(arm["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train["ids"]) / batch_size))
    total_steps = max(1, epochs * steps_per_epoch)
    ema_decay = float(arm.get("ema_decay", 0.0))
    ema = _trainable_state(model) if ema_decay > 0 else None
    best_score = float("inf")
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    stale = 0
    global_step = 0
    history: List[Dict[str, Any]] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed * 100003 + epoch)
        order = torch.randperm(len(train["ids"]), generator=generator, device=device)
        losses: List[float] = []
        ce_values: List[float] = []
        huber_values: List[float] = []
        consistency_values: List[float] = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_ids = [train["ids"][int(i)] for i in idx.detach().cpu().tolist()]
            full_obs = train["obs"][idx]
            missing_obs = training_missing(
                full_obs,
                train["maskable_obs"][idx],
                batch_ids,
                epoch,
                seed,
                str(arm["mask_policy"]),
            )
            out_full = model(
                train["text_bert"][idx, 0],
                full_obs,
                train["audio"][idx],
                train["vision"][idx],
            )
            out_missing = model(
                train["text_bert"][idx, 0],
                missing_obs,
                train["audio"][idx],
                train["vision"][idx],
            )
            full_loss, full_parts = _supervised_loss(
                out_full["logits"],
                out_full["raw_intensity"],
                train["labels"][idx],
                train["regression"][idx],
                class_weight,
                arm,
            )
            missing_loss, missing_parts = _supervised_loss(
                out_missing["logits"],
                out_missing["raw_intensity"],
                train["labels"][idx],
                train["regression"][idx],
                class_weight,
                arm,
            )
            loss = full_loss + float(arm["lambda_missing"]) * missing_loss
            consistency_weight = float(arm.get("consistency_weight", 0.0))
            consistency = torch.zeros((), device=device)
            if consistency_weight > 0:
                teacher_prob = F.softmax(out_full["logits"].detach(), dim=-1)
                cls_consistency = F.kl_div(
                    F.log_softmax(out_missing["logits"], dim=-1),
                    teacher_prob,
                    reduction="batchmean",
                )
                reg_consistency = F.smooth_l1_loss(
                    out_missing["raw_intensity"],
                    out_full["raw_intensity"].detach(),
                    beta=float(arm["huber_beta"]),
                )
                consistency = cls_consistency + reg_consistency
                loss = loss + consistency_weight * consistency
            optimizer.lr = _learning_rate(
                float(arm["lr"]),
                global_step,
                total_steps,
                str(arm["scheduler"]),
                float(arm["warmup_fraction"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(arm["gradient_clip"]))
            optimizer.step()
            global_step += 1
            if ema is not None:
                state = model.state_dict()
                with torch.no_grad():
                    for name in trainable_names:
                        ema[name].mul_(ema_decay).add_(
                            state[name].detach().cpu(),
                            alpha=1.0 - ema_decay,
                        )
            losses.append(float(loss.detach().cpu()))
            ce_values.append(0.5 * (full_parts["ce"] + missing_parts["ce"]))
            huber_values.append(0.5 * (full_parts["huber"] + missing_parts["huber"]))
            consistency_values.append(float(consistency.detach().cpu()))

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "ce": float(np.mean(ce_values)),
            "huber": float(np.mean(huber_values)),
            "consistency": float(np.mean(consistency_values)),
            "lr": float(optimizer.lr),
            "elapsed_seconds": time.time() - started,
        }
        if validation is not None:
            raw_state = _swap_trainable_state(model, ema) if ema is not None else None
            screen = screen_model(model, validation, screen_mask_seed)
            row.update(screen)
            candidate_state = _trainable_state(model)
            if raw_state is not None:
                _load_trainable_state(model, raw_state)
            if screen["score"] < best_score - min_delta:
                best_score = screen["score"]
                best_epoch = epoch
                best_state = candidate_state
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(
            json.dumps({"arm": arm["id"], "seed": seed, **row}, ensure_ascii=False),
            flush=True,
        )
        if validation is not None and stale >= patience:
            break

    if validation is not None:
        if best_state is None:
            raise RuntimeError("cross-validation training did not produce a best state")
        _load_trainable_state(model, best_state)
    elif ema is not None:
        _load_trainable_state(model, ema)
        best_epoch = epochs - 1
        best_score = float("nan")
    else:
        best_epoch = epochs - 1
        best_score = float("nan")

    info = {
        "arm": dict(arm),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_run": len(history),
        "best_epoch": int(best_epoch),
        "best_screen_score": best_score,
        "history": history,
        "params_total": int(sum(parameter.numel() for parameter in model.parameters())),
        "params_trainable": int(sum(parameter.numel() for parameter in trainable)),
        "seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
    }
    return model, info


def save_compact_checkpoint(
    path: Path,
    model: Q2Model,
    norm: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> str:
    prefix = "text.bert."
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    }
    payload = {
        "format": "q2-ds-hc-compact-v1",
        "kind": "M0",
        "seed": int(metadata["seed"]),
        "state_dict": state,
        "norm": {key: value.detach().cpu() for key, value in norm.items()},
        "model_config": {
            "hidden": int(metadata["arm"]["hidden"]),
            "dropout": float(metadata["arm"]["dropout"]),
        },
        "metadata": dict(metadata),
        "excluded_prefix": prefix,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_any_checkpoint(
    path: Path,
    model_dir: Path,
    device: torch.device,
) -> Tuple[Q2Model, Dict[str, Any], Dict[str, torch.Tensor]]:
    payload = _torch_load(path)
    kind = str(payload.get("kind", "M0"))
    if kind != "M0":
        raise RuntimeError(f"only M0 checkpoints are admitted, got {kind} from {path}")
    model_config = dict(payload.get("model_config", {}))
    model = Q2Model(
        kind,
        model_dir,
        hidden=int(model_config.get("hidden", 128)),
        dropout=float(model_config.get("dropout", 0.15)),
    ).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    illegal_missing = [key for key in missing if not key.startswith("text.bert.")]
    if illegal_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch for {path}: missing={illegal_missing}, unexpected={unexpected}"
        )
    model.eval()
    norm = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in payload["norm"].items()
    }
    return model, payload, norm


def condition_specs(mask_seed: int, replicas: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "id": "clean",
            "subset": (),
            "ratio": 0.0,
            "replica": 0,
            "mask_seed": int(mask_seed),
        }
    ]
    for subset in SUBSETS:
        for ratio in RATIOS:
            for replica in range(replicas):
                specs.append(
                    {
                        "id": f"{subset_name(subset)}|{ratio:.1f}|r{replica}",
                        "subset": tuple(subset),
                        "ratio": float(ratio),
                        "replica": int(replica),
                        "mask_seed": int(mask_seed),
                    }
                )
    return specs


def cache_member_predictions(
    checkpoint: Path,
    member_id: str,
    data_path: Path,
    model_dir: Path,
    output: Path,
    mask_seed: int,
    replicas: int,
    device: torch.device,
) -> Dict[str, Any]:
    model, payload, norm = load_any_checkpoint(checkpoint, model_dir, device)
    with np.load(data_path, allow_pickle=False) as archive:
        data = load_split(archive, "valid", device)
    apply_normalization(data, norm)
    specs = condition_specs(mask_seed, replicas)
    probabilities: List[np.ndarray] = []
    raws: List[np.ndarray] = []
    for spec in specs:
        logits, raw = _condition_predictions(
            model,
            data,
            tuple(spec["subset"]),
            float(spec["ratio"]),
            int(spec["replica"]),
            mask_seed,
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        probabilities.append(probability.astype(np.float32))
        raws.append(raw.astype(np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        member_id=np.asarray(member_id),
        checkpoint=np.asarray(str(checkpoint.resolve())),
        checkpoint_sha256=np.asarray(sha256_file(checkpoint)),
        ids=np.asarray(data["ids"]),
        condition_ids=np.asarray([spec["id"] for spec in specs]),
        probabilities=np.stack(probabilities),
        raw=np.stack(raws),
    )
    return {
        "member_id": member_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cache": str(output.resolve()),
        "cache_sha256": sha256_file(output),
        "conditions": len(specs),
        "samples": len(data["ids"]),
        "kind": str(payload.get("kind", "M0")),
    }


def compact_existing_checkpoint(source: Path, destination: Path) -> Dict[str, Any]:
    payload = _torch_load(source)
    prefix = "text.bert."
    compact = dict(payload)
    compact["format"] = "q2-ds-hc-compact-v1"
    compact["excluded_prefix"] = prefix
    compact["state_dict"] = {
        key: value
        for key, value in payload["state_dict"].items()
        if not key.startswith(prefix)
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, destination)
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "compact": str(destination.resolve()),
        "compact_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def train_job(
    config: Mapping[str, Any],
    data_path: Path,
    model_dir: Path,
    arm_id: str,
    seed: int,
    device_name: str,
    output_dir: Path,
    fold: int | None = None,
    fixed_epochs: int | None = None,
) -> Dict[str, Any]:
    device = choose_device(device_name)
    arm = merged_arm(config, arm_id)
    search = config["search"]
    with np.load(data_path, allow_pickle=False) as archive:
        complete_train = load_split(archive, "train", device)
    if fold is None:
        train = complete_train
        validation = None
        epochs = int(fixed_epochs or search["max_epochs"])
        split_meta = {"mode": "full_train_fixed_epoch", "fold": None}
    else:
        train_idx, holdout_idx = fold_indices(
            complete_train["ids"],
            int(search["folds"]),
            int(fold),
            int(search["fold_seed"]),
            str(search["group_separator"]),
        )
        train = slice_data(complete_train, train_idx)
        validation = slice_data(complete_train, holdout_idx)
        epochs = int(search["max_epochs"])
        split_meta = {
            "mode": "grouped_cross_validation",
            "fold": int(fold),
            "train_n": len(train_idx),
            "holdout_n": len(holdout_idx),
        }
        del complete_train
    norm = compute_norm(train)
    apply_normalization(train, norm)
    if validation is not None:
        apply_normalization(validation, norm)
    model, training = train_m0(
        train,
        validation,
        model_dir,
        arm,
        seed,
        epochs,
        int(search["patience"]),
        float(search["min_delta"]),
        int(search["screen_mask_seed"]),
        device,
    )
    robust = None
    checkpoint = None
    checkpoint_sha256 = None
    if validation is not None:
        robust = evaluate_model(
            model,
            validation,
            int(search["robust_mask_seed"]),
            int(search["robust_replicas"]),
        )
    else:
        checkpoint = output_dir / "checkpoint.pt"
        checkpoint_sha256 = save_compact_checkpoint(checkpoint, model, norm, training)
    result = {
        "schema": "q2-ds-hc-train-job-v1",
        "protocol_sha256": config.get("_runtime_protocol_sha256"),
        "execution_fingerprint": runtime_fingerprint(config, data_path, model_dir),
        "arm_id": arm_id,
        "arm": arm,
        "seed": int(seed),
        "device": str(device),
        "data": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "model_dir": str(model_dir.resolve()),
        "split": split_meta,
        "training": training,
        "robust": robust,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
    }
    write_json(output_dir / "result.json", result)
    return result
