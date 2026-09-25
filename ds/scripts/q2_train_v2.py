#!/usr/bin/env python3
"""Train and evaluate the Q2 model family on the server.

Only train/valid are used for fitting and selection.  The official test split
is never read by the training loop and is only used after a model has been
frozen by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from q2_models import Q2Model

MODALITIES = ("text", "audio", "vision")
SUBSETS = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
RATIOS = (0.1, 0.3, 0.5)
MODES = ("start", "middle", "end")


class AdamW:
    """Minimal AdamW implementation.

    The server's PyTorch build imports torch.optim through a code path that
    requires an unavailable sympy package.  This local implementation keeps
    the standard AdamW equations and lets the experiment run without adding
    dependencies.
    """
    def __init__(self, params: Sequence[torch.nn.Parameter], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-4):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state = [{"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)} for p in self.params]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            grad = p.grad
            p.mul_(1.0 - self.lr * self.weight_decay)
            st["exp_avg"].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            st["exp_avg_sq"].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = st["exp_avg_sq"].sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
            p.addcdiv_(st["exp_avg"], denom, value=-self.lr / bias_correction1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_unit(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def load_split(z: np.lib.npyio.NpzFile, split: str, device: torch.device) -> Dict[str, Any]:
    text_bert = torch.from_numpy(np.asarray(z[f"{split}_text_bert"], dtype=np.int64)).to(device)
    audio = torch.from_numpy(np.asarray(z[f"{split}_audio"], dtype=np.float32)).to(device)
    vision = torch.from_numpy(np.asarray(z[f"{split}_vision"], dtype=np.float32)).to(device)
    text_feature = torch.from_numpy(np.asarray(z[f"{split}_text_feature"], dtype=np.float32)).to(device)
    labels = torch.from_numpy(np.asarray(z[f"{split}_labels"], dtype=np.int64)).to(device)
    regression = torch.from_numpy(np.asarray(z[f"{split}_regression"], dtype=np.float32)).to(device)
    ids = [str(x) for x in z[f"{split}_ids"].tolist()]
    text_obs = text_bert[:, 1] > 0
    audio_obs = audio.abs().amax(dim=-1) > 1e-8
    vision_obs = vision.abs().amax(dim=-1) > 1e-8
    obs = torch.stack([text_obs, audio_obs, vision_obs], dim=1)
    # [CLS]/[SEP] are valid for BERT encoding but are not semantic
    # evidence positions and must never be counted as maskable text tokens.
    text_semantic = text_obs & (text_bert[:, 0] != 101) & (text_bert[:, 0] != 102)
    maskable_obs = torch.stack([text_semantic, audio_obs, vision_obs], dim=1)
    return {
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "text_feature": text_feature,
        "labels": labels,
        "regression": regression,
        "ids": ids,
        "obs": obs,
        "maskable_obs": maskable_obs,
    }


def compute_norm(train: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, key in (("audio", "audio"), ("vision", "vision")):
        x = train[key].float()
        m = train["obs"][:, 1 if name == "audio" else 2].unsqueeze(-1)
        vals = x[m.expand_as(x)].reshape(-1, x.shape[-1])
        mean = vals.mean(dim=0)
        std = vals.std(dim=0).clamp_min(1e-5)
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = std
    return out


def normalize_modal(x: torch.Tensor, obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    z = (x - mean) / std
    return torch.where(obs.unsqueeze(-1), z, torch.zeros_like(z))


def apply_normalization(data: Dict[str, Any], norm: Dict[str, torch.Tensor]) -> None:
    data["audio"] = normalize_modal(data["audio"].float(), data["obs"][:, 1], norm["audio_mean"], norm["audio_std"])
    data["vision"] = normalize_modal(data["vision"].float(), data["obs"][:, 2], norm["vision_mean"], norm["vision_std"])


def choose_subset(sid: str, epoch: int, seed: int) -> Tuple[int, ...]:
    """Sample the seven non-empty modality subsets uniformly.

    The evaluation weights each non-empty subset equally.  The original
    curriculum sampled singles 70% of the time, which under-trained the
    double/triple missing cases.  This version removes that mismatch.
    """
    u = stable_unit(f"{sid}|balanced_subset|{epoch}|{seed}")
    return SUBSETS[min(len(SUBSETS) - 1, int(u * len(SUBSETS)))]


def choose_ratio(sid: str, epoch: int, seed: int) -> float:
    u = stable_unit(f"{sid}|ratio|{epoch}|{seed}")
    return RATIOS[min(2, int(u * 3))]


def make_missing(
    obs: torch.Tensor,
    ids: Sequence[str],
    maskable_obs: torch.Tensor | None = None,
    subset: Tuple[int, ...] | None = None,
    ratio: float | None = None,
    replica: int = 0,
    mode: str = "random",
    epoch: int = 0,
    seed: int = 42,
) -> torch.Tensor:
    """Return a visibility mask; only original-observed positions can remain."""
    vis = obs.clone()
    base = obs if maskable_obs is None else maskable_obs
    for b, sid in enumerate(ids):
        use_subset = subset if subset is not None else choose_subset(sid, epoch, seed)
        use_ratio = ratio if ratio is not None else choose_ratio(sid, epoch, seed)
        u_start = stable_unit(f"{sid}|start|{replica}|{mode}|{epoch}|{seed}")
        for m in use_subset:
            # Positions eligible for synthetic masking are based on semantic
            # evidence, while visibility always starts from the true observed
            # mask.  This keeps [CLS]/[SEP] and padding out of the denominator.
            positions = torch.nonzero(base[b, m], as_tuple=False).flatten()
            length = int(positions.numel())
            if length < 2:
                continue
            k = min(length - 1, max(1, int(round(use_ratio * length))))
            if mode == "start":
                start = 0
            elif mode == "middle":
                start = (length - k) // 2
            elif mode == "end":
                start = length - k
            else:
                start = min(length - k, int(u_start * (length - k + 1)))
            vis[b, m, positions[start:start + k]] = False
    return vis


def loss_fn(logits: torch.Tensor, raw: torch.Tensor, labels: torch.Tensor, regression: torch.Tensor, class_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, labels, weight=class_weight)
    huber = F.smooth_l1_loss(raw, regression, beta=1.0)
    return ce + huber, ce, huber


def forward_model(model: Q2Model, data: Dict[str, Any], obs: torch.Tensor, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return model(data["text_bert"][idx, 0], obs, data["audio"][idx], data["vision"][idx])


@torch.no_grad()
def predict_model(model: Q2Model, data: Dict[str, Any], obs: torch.Tensor, indices: torch.Tensor | None = None, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    if indices is None:
        indices = torch.arange(data["labels"].numel(), device=data["labels"].device)
    logits_out: List[np.ndarray] = []
    raw_out: List[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        out = forward_model(model, data, obs[idx], idx)
        logits_out.append(out["logits"].detach().float().cpu().numpy())
        raw_out.append(out["raw_intensity"].detach().float().cpu().numpy())
    return np.concatenate(logits_out, axis=0), np.concatenate(raw_out, axis=0)


def publish_c2(logits: np.ndarray, raw: np.ndarray, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    cls = logits.argmax(axis=1)
    intensity = np.zeros_like(raw, dtype=np.float32)
    pos = cls == 2
    neg = cls == 0
    intensity[pos] = np.maximum(raw[pos], eps)
    intensity[neg] = np.minimum(raw[neg], -eps)
    return cls, intensity


def search_c3(raw: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    best = (-1.0, -1.0, -1e9)
    for dn in np.linspace(-1.5, -0.05, 30):
        for dp in np.linspace(0.05, 1.5, 30):
            pred = np.where(raw < dn, 0, np.where(raw > dp, 2, 1))
            score = f1_score(labels, pred, average="macro", zero_division=0) + 0.25 * accuracy_score(labels, pred)
            if score > best[2]:
                best = (float(dn), float(dp), float(score))
    return best


def publish_c3(raw: np.ndarray, dn: float, dp: float) -> Tuple[np.ndarray, np.ndarray]:
    cls = np.where(raw < dn, 0, np.where(raw > dp, 2, 1)).astype(np.int64)
    intensity = np.zeros_like(raw, dtype=np.float32)
    pos = cls == 2
    neg = cls == 0
    intensity[pos] = np.maximum(raw[pos], 1e-4)
    intensity[neg] = np.minimum(raw[neg], -1e-4)
    return cls, intensity


def metrics(logits: np.ndarray, raw: np.ndarray, labels: np.ndarray, regression: np.ndarray, strategy: str = "c2", thresholds: Tuple[float, float] | None = None) -> Dict[str, float]:
    if strategy == "c3" and thresholds is not None:
        pred, intensity = publish_c3(raw, thresholds[0], thresholds[1])
    else:
        pred, intensity = publish_c2(logits, raw)
    if len(np.unique(labels)) > 1:
        pearson = float(np.corrcoef(regression, intensity)[0, 1])
    else:
        pearson = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(regression, intensity)),
        "pearson": pearson,
        "n": int(len(labels)),
    }


def subset_name(subset: Tuple[int, ...]) -> str:
    return "+".join(MODALITIES[i] for i in subset)


def fit_b1(train: Dict[str, Any]) -> Dict[str, Any]:
    # Masked temporal mean of the supplied 768-d text features.
    x = train["text_feature"].float()
    m = train["obs"][:, 0].unsqueeze(-1).float()
    feat = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    x_np = feat.detach().cpu().numpy()
    y_np = train["labels"].detach().cpu().numpy()
    r_np = train["regression"].detach().cpu().numpy()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(x_np, y_np)
    reg = Ridge(alpha=1.0)
    reg.fit(x_np, r_np)
    return {"clf": clf, "reg": reg, "median": float(np.median(r_np))}


def predict_b1(model: Dict[str, Any], data: Dict[str, Any], obs: torch.Tensor, indices: torch.Tensor | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if indices is None:
        indices = torch.arange(data["labels"].numel(), device=data["labels"].device)
    x = data["text_feature"][indices].float()
    m = obs[indices, 0].unsqueeze(-1).float()
    feat = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    x_np = feat.detach().cpu().numpy()
    logits = model["clf"].predict_proba(x_np)
    raw = model["reg"].predict(x_np).astype(np.float32)
    return logits, raw


def evaluate_condition(
    model: Q2Model | Dict[str, Any] | None,
    kind: str,
    data: Dict[str, Any],
    subset: Tuple[int, ...],
    ratio: float,
    replica: int,
    mode: str = "random",
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    if kind == "B0":
        labels = data["labels"].detach().cpu().numpy()
        prior = np.bincount(labels, minlength=3).astype(np.float32)
        prior = prior / prior.sum()
        logits = np.tile(np.log(np.clip(prior, 1e-9, None)), (len(labels), 1))
        raw = np.full(len(labels), float(np.median(data["regression"].detach().cpu().numpy())), dtype=np.float32)
    else:
        obs = make_missing(data["obs"], data["ids"], data["maskable_obs"], subset=subset, ratio=ratio, replica=replica, mode=mode, seed=seed)
        if kind == "B1":
            logits, raw = predict_b1(model, data, obs)  # type: ignore[arg-type]
        else:
            logits, raw = predict_model(model, data, obs)  # type: ignore[arg-type]
    y = data["labels"].detach().cpu().numpy()
    r = data["regression"].detach().cpu().numpy()
    return metrics(logits, raw, y, r), logits, raw


def screen_score(model: Q2Model, data: Dict[str, Any], n: int = 256) -> Tuple[float, Dict[str, float]]:
    idx = torch.arange(min(n, data["labels"].numel()), device=data["labels"].device)
    conditions = [((), 0.0, 0, "random")]
    for m in range(3):
        for rho in (0.1, 0.3, 0.5):
            conditions.append(((m,), rho, 0, "random"))
    conditions.append(((0, 1, 2), 0.3, 0, "random"))
    maes = []
    f1s = []
    for subset, ratio, replica, mode in conditions:
        if not subset:
            obs = data["obs"][idx]
            logits, raw = predict_model(model, data, obs, idx)
        else:
            sub_ids = [data["ids"][int(i)] for i in idx.detach().cpu().tolist()]
            obs_sub = make_missing(data["obs"][idx], sub_ids, data["maskable_obs"][idx], subset=subset, ratio=ratio, replica=replica, mode=mode, seed=42)
            logits, raw = predict_model(model, data, obs_sub, idx)
        y = data["labels"][idx].detach().cpu().numpy()
        r = data["regression"][idx].detach().cpu().numpy()
        mm = metrics(logits, raw, y, r)
        maes.append(mm["mae"])
        f1s.append(mm["macro_f1"])
    return float(np.mean(maes) + 0.1 * (1.0 - np.mean(f1s))), {"screen_mae": float(np.mean(maes)), "screen_macro_f1": float(np.mean(f1s))}


def train_one(kind: str, seed: int, train: Dict[str, Any], valid: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Tuple[Q2Model, Dict[str, Any]]:
    set_seed(seed)
    model_kind = "M0" if kind == "M1" else kind
    model = Q2Model(model_kind, args.model_dir, hidden=args.hidden, dropout=args.dropout).to(device)
    # The pretrained BERT is frozen by construction.  Only the task/fusion
    # modules are optimized, which is the efficient low-data setting.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    counts = torch.bincount(train["labels"], minlength=3).float()
    class_weight = (counts.sum() / (counts.clamp_min(1) * 3)).to(device)
    best = None
    best_state = None
    history = []
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(train["labels"].numel(), device=device)
        epoch_losses = []
        t0 = time.time()
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            full_obs = train["obs"][idx]
            batch_ids = [train["ids"][int(i)] for i in idx.detach().cpu().tolist()]
            miss_obs = make_missing(full_obs, batch_ids, train["maskable_obs"][idx], epoch=epoch, seed=seed)
            out_full = forward_model(model, train, full_obs, idx)
            out_miss = forward_model(model, train, miss_obs, idx)
            lf, cef, huf = loss_fn(out_full["logits"], out_full["raw_intensity"], train["labels"][idx], train["regression"][idx], class_weight)
            lm, cem, hum = loss_fn(out_miss["logits"], out_miss["raw_intensity"], train["labels"][idx], train["regression"][idx], class_weight)
            loss = lf + args.lambda_miss * lm
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        score, screen = screen_score(model, valid, n=args.screen_n)
        row = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "score": score, **screen, "seconds": time.time() - t0}
        history.append(row)
        print(json.dumps({"kind": kind, "seed": seed, **row}, ensure_ascii=False), flush=True)
        if best is None or score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no state")
    model.load_state_dict(best_state)
    return model, {"kind": kind, "seed": seed, "best_score": best, "history": history, "params_total": sum(p.numel() for p in model.parameters()), "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="B2,B3,B4,B5,M0")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-miss", type=float, default=1.0)
    ap.add_argument("--screen-n", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-eval-models", type=int, default=0)
    ap.add_argument("--save-valid-raw", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    z = np.load(args.data, allow_pickle=False)
    print("loading splits", flush=True)
    train = load_split(z, "train", device)
    valid = load_split(z, "valid", device)
    norm = compute_norm(train)
    apply_normalization(train, norm)
    apply_normalization(valid, norm)
    print(json.dumps({"train_n": len(train["ids"]), "valid_n": len(valid["ids"]), "device": str(device)}, ensure_ascii=False), flush=True)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    all_results: Dict[str, Any] = {"args": vars(args) | {"data": str(args.data), "model_dir": str(args.model_dir), "out_dir": str(args.out_dir)}, "runs": {}}
    (args.out_dir / "config.json").write_text(json.dumps(all_results["args"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for kind in models:
        for seed in seeds:
            key = f"{kind}_seed{seed}"
            if kind == "B0":
                model = None
                info = {"kind": kind, "seed": seed, "params_total": 0, "params_trainable": 0}
            elif kind == "B1":
                model = fit_b1(train)
                info = {"kind": kind, "seed": seed, "params_total": "sklearn", "params_trainable": "sklearn"}
            else:
                print(f"training {key}", flush=True)
                model, info = train_one(kind, seed, train, valid, args, device)
                ckpt = args.out_dir / f"{key}.pt"
                torch.save({
                    "kind": "M0" if kind == "M1" else kind,
                    "variant": kind,
                    "training_protocol": "balanced_subset_v1",
                    "seed": seed,
                    "state_dict": model.state_dict(),
                    "info": info,
                    "norm": {k: v.detach().cpu() for k, v in norm.items()},
                    "model_config": {"hidden": args.hidden, "dropout": args.dropout},
                }, ckpt)
                info["checkpoint"] = str(ckpt)
            all_results["runs"][key] = info
            # Full validation matrix and position analysis are saved for every
            # trained candidate; this is the selection evidence.
            condition_rows = []
            for subset in SUBSETS:
                for ratio in RATIOS:
                    reps = []
                    for replica in range(3):
                        mm, _, _ = evaluate_condition(model, kind, valid, subset, ratio, replica, "random", seed)
                        reps.append(mm)
                    condition_rows.append({"subset": subset_name(subset), "ratio": ratio, "replicas": reps, "mean_mae": float(np.mean([x["mae"] for x in reps])), "mean_macro_f1": float(np.mean([x["macro_f1"] for x in reps]))})
            full_metrics, full_logits, full_raw = evaluate_condition(model, kind, valid, (), 0.0, 0, "random", seed)
            all_results["runs"][key]["full"] = full_metrics
            all_results["runs"][key]["conditions"] = condition_rows
            all_results["runs"][key]["R_MAE"] = float(np.mean([x["mean_mae"] for x in condition_rows]))
            all_results["runs"][key]["mean_macro_f1"] = float(np.mean([x["mean_macro_f1"] for x in condition_rows]))
            all_results["runs"][key]["worst_condition_mae"] = float(max(x["mean_mae"] for x in condition_rows))
            # Position analysis: 3 single modalities x 3 ratios x 3 positions.
            pos_rows = []
            for m in range(3):
                for ratio in RATIOS:
                    for mode in MODES:
                        mm, _, _ = evaluate_condition(model, kind, valid, (m,), ratio, 0, mode, seed)
                        pos_rows.append({"subset": MODALITIES[m], "ratio": ratio, "position": mode, **mm})
            all_results["runs"][key]["positions"] = pos_rows
            print(json.dumps({"key": key, "R_MAE": all_results["runs"][key]["R_MAE"], "mean_macro_f1": all_results["runs"][key]["mean_macro_f1"], "full": full_metrics}, ensure_ascii=False), flush=True)
            # Save raw validation predictions for the final candidate only.
            if kind == "M0" or args.save_valid_raw:
                np.savez_compressed(args.out_dir / f"{key}_valid_full.npz", logits=full_logits, raw=full_raw)
            (args.out_dir / "validation_results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.max_eval_models and len(all_results["runs"]) >= args.max_eval_models:
                break
    (args.out_dir / "validation_results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "runs": list(all_results["runs"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
