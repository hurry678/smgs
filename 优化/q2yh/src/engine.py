"""Training / evaluation engine for q2-plan-1.0 (server tasks T02-T03)."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D
from losses import compute_loss, class_weights_sqrt_inverse
from metrics import metric_block, softmax
from models import build_model, non_encoder_parameter_count, trainable_parameter_count
from optimizers import build_optimizer, pcgrad_project, shared_parameters
from protocol_core import publish_c2, stable_index


# --------------------------------------------------------------------------
# run data
# --------------------------------------------------------------------------

class RunData:
    """Train/valid tensors with the frozen normalizer already applied."""

    def __init__(self, run_dir, device="cuda:0"):
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        # the normalizer must exist before _split applies it to audio/vision
        self.normalizer = D.load_normalizer(self.run_dir / "data" / "normalizer.npz")
        self.train = self._split("train", "train.npz")
        self.valid = self._split("valid", "valid.npz")

    def _split(self, name, filename):
        raw = D.load_npz(self.run_dir / "data" / filename)
        out = {
            "name": name,
            "id": [str(x) for x in raw["id"]],
            "video_id": [str(x) for x in raw["video_id"]],
            "domain": raw["domain"].astype(bool),
            "observed": raw["observed"].astype(bool),
            "text_bert": raw["text_bert"].astype(np.int64),
        }
        for field in ("audio", "vision"):
            mean = self.normalizer[field]["mean"]
            std = self.normalizer[field]["std"]
            out[field] = D.apply_normalizer(raw[field], raw["observed"][:, {"audio": 1, "vision": 2}[field]],
                                            mean, std)
        if "classification_labels" in raw:
            out["classification_labels"] = raw["classification_labels"].astype(np.int64)
            out["regression_labels"] = raw["regression_labels"].astype(np.float32)
        return out

    def tensors(self, split, index=None):
        """Move one split (or a slice) onto the device as torch tensors."""
        s = self.train if split == "train" else self.valid
        if index is None:
            idx = np.arange(len(s["id"]))
        else:
            idx = np.asarray(index)
        return {
            "text_bert": torch.from_numpy(s["text_bert"][idx]).to(self.device),
            "audio": torch.from_numpy(s["audio"][idx]).to(self.device),
            "vision": torch.from_numpy(s["vision"][idx]).to(self.device),
            "domain": torch.from_numpy(s["domain"][idx]).to(self.device),
            "observed": torch.from_numpy(s["observed"][idx]).to(self.device),
            "classification_labels": torch.from_numpy(s["classification_labels"][idx]).to(self.device),
            "regression_labels": torch.from_numpy(s["regression_labels"][idx]).to(self.device),
            "id": [s["id"][i] for i in idx],
            "video_id": [s["video_id"][i] for i in idx],
        }


# --------------------------------------------------------------------------
# training-time augmentation (never reads a validation bank)
# --------------------------------------------------------------------------

def augmentation_assignment(ids, aug_seed, epoch):
    """Deterministic per-sample (subset, ratio) for one epoch, independent of order."""
    subs, ratios = [], []
    for sid in ids:
        s = stable_index((aug_seed, epoch, sid, "subset"), len(D.SUBSETS))
        r = stable_index((aug_seed, epoch, sid, "ratio"), len(D.RATIOS))
        subs.append(D.SUBSETS[s])
        ratios.append(D.RATIOS[r])
    return np.asarray(subs, dtype=object), np.asarray(ratios, dtype=float)


def build_epoch_missing(domain, observed, ids, aug_seed, epoch):
    """One synthetic missing view per train sample; never restores native missing."""
    from protocol_core import make_missing
    subs, ratios = augmentation_assignment(ids, aug_seed, epoch)
    out = observed.copy()
    groups = {}
    for i, (sub, ratio) in enumerate(zip(subs, ratios)):
        groups.setdefault((tuple(int(x) for x in sub), float(ratio)), []).append(i)
    for (sub, ratio), members in groups.items():
        members = np.asarray(members)
        vis, _ = make_missing(domain[members], observed[members],
                              [ids[i] for i in members], sub, ratio,
                              seed=aug_seed, split="train", replica=epoch,
                              placement="random", coupling="independent")
        out[members] = vis
    if np.any(out & ~observed):
        raise ValueError("Augmentation invented observations")
    return out


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------

def prior_statistics(train_split):
    cls = np.asarray(train_split["classification_labels"])
    reg = np.asarray(train_split["regression_labels"])
    counts = np.asarray([(cls == k).sum() for k in range(3)], dtype=np.float64)
    means = np.asarray([reg[cls == k].mean() if (cls == k).any() else 0.0 for k in range(3)],
                       dtype=np.float64)
    return counts, means


def make_model(architecture, run_data, **kwargs):
    if architecture == "training_prior":
        counts, means = prior_statistics(run_data.train)
        return build_model(architecture, class_counts=counts, class_intensity_mean=means)
    return build_model(architecture, **kwargs)


def set_text_cache(model, slot):
    if hasattr(model, "_cache_slot"):
        if slot != model._cache_slot:
            model.clear_text_cache()
        model._cache_slot = slot


# --------------------------------------------------------------------------
# one forward pass over a mask condition
# --------------------------------------------------------------------------

def forward_condition(model, tensors, visible, batch_size=256):
    n = tensors["domain"].shape[0]
    logits, raw = [], []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        out = model(tensors["text_bert"][start:stop], tensors["audio"][start:stop],
                    tensors["vision"][start:stop], tensors["domain"][start:stop],
                    visible[start:stop])
        logits.append(out["logits"].detach().float().cpu())
        raw.append(out["raw_intensity"].detach().float().cpu())
    return torch.cat(logits).numpy(), torch.cat(raw).numpy()


# --------------------------------------------------------------------------
# checkpoint helpers (frozen encoder is never duplicated into a candidate file)
# --------------------------------------------------------------------------

def checkpoint_payload(model, meta):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if not k.startswith("text_encoder.")}
    return {"state_dict": state, "meta": meta,
            "encoder": "resources/bert_shared (shared, stored once)"}


def save_checkpoint(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, meta), path)
    return D.sha256_file(path)


def load_checkpoint(path, model):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    missing = [k for k in missing if not k.startswith("text_encoder.")]
    unexpected = [k for k in unexpected if not k.startswith("text_encoder.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return payload.get("meta", {})


# --------------------------------------------------------------------------
# evaluation over a mask bank
# --------------------------------------------------------------------------

def evaluate_bank(model, run_data, bank_data, split="valid", batch_size=256, epsilon=1e-4,
                  limit=None):
    """Per-condition predictions and metrics for one bank (smoke may pass a small limit)."""
    tensors = run_data.tensors(split, None if limit is None else np.arange(int(limit)))
    source = run_data.train if split == "train" else run_data.valid
    n = tensors["domain"].shape[0]
    labels_class = tensors["classification_labels"].cpu().numpy()
    labels_reg = tensors["regression_labels"].cpu().numpy()
    observed_np = source["observed"][:n]
    domain_np = source["domain"][:n]
    model.eval()
    per_condition, predictions = {}, {}
    with torch.no_grad():
        for ci, cid in enumerate(bank_data["condition_ids"]):
            vis = D.visibility_of(bank_data, ci)[:n]
            text_unmasked = np.array_equal(vis[:, 0], observed_np[:, 0])
            set_text_cache(model, "full" if text_unmasked else None)
            logits, raw = forward_condition(model, tensors, torch.from_numpy(vis).to(run_data.device),
                                            batch_size=batch_size)
            pub_class, pub_intensity = publish_c2(softmax(logits), raw, epsilon=epsilon)
            block = metric_block(labels_class, labels_reg, logits, raw, pub_class, pub_intensity)
            length = domain_np.sum(1)
            visible_count = vis.sum(-1)
            block["actual_missing_ratio"] = float(np.mean(1 - visible_count.sum(1) / (3 * length)))
            block["samples_with_new_masking"] = int((vis.sum(-1) < observed_np.sum(-1)).any(1).sum())
            per_condition[cid] = block
            predictions[cid] = {
                "logits": logits, "raw": raw, "published_class": pub_class,
                "published_intensity": pub_intensity,
                "mask_sha256": [D.sample_mask_sha(vis[j]) for j in range(len(vis))],
            }
    if hasattr(model, "clear_text_cache"):
        model.clear_text_cache()
    set_text_cache(model, None)
    return per_condition, predictions


def summarise_bank(per_condition, specs):
    """R_* aggregates: mean over replicas per (subset,ratio) cell, then equal-weight cells."""
    from metrics import aggregate_condition_matrix
    out = {}
    for metric in ("accuracy", "macro_f1", "mae", "pearson", "raw_mae", "raw_pearson"):
        values = {cid: b[metric] for cid, b in per_condition.items()
                  if b.get(metric) is not None}
        if not values:
            out[metric] = None
            continue
        mean, per_cell = aggregate_condition_matrix(values, specs)
        out[metric] = mean
        if metric in ("mae", "macro_f1"):
            out[f"worst_condition_{metric}"] = float(max(per_cell.values()))
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _task_gradients(model, batch, visible, shared_params, huber_beta, class_weights):
    """CE / regression gradients w.r.t. shared params for one view (PCGrad + diagnostics)."""
    out = model(batch["text_bert"], batch["audio"], batch["vision"], batch["domain"], visible)
    ce = F.cross_entropy(out["logits"], batch["classification_labels"], weight=class_weights)
    reg = F.smooth_l1_loss(out["raw_intensity"], batch["regression_labels"], beta=huber_beta)
    g_ce = torch.autograd.grad(ce, shared_params, retain_graph=True, allow_unused=True)
    g_reg = torch.autograd.grad(reg, shared_params, retain_graph=True, allow_unused=True)
    return float(ce.detach()), float(reg.detach()), g_ce, g_reg


def grad_cosine(g_ce, g_reg):
    """Cosine between the classification and regression gradients of the shared trunk."""
    dot = sum(float((a * b).sum()) for a, b in zip(g_ce, g_reg) if a is not None and b is not None)
    na = math.sqrt(sum(float((a * a).sum()) for a in g_ce if a is not None))
    nb = math.sqrt(sum(float((b * b).sum()) for b in g_reg if b is not None))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return dot / (na * nb)


def train_one(run_dir, architecture, config, run_data, select_bank, device="cuda:0",
              log_path=None, seed=42, max_epochs=None, patience=None, train_limit=None,
              valid_limit=None, progress=None, state_path=None, resume_from=None,
              cos_every=100):
    """Train one candidate; early stopping uses only the select bank on valid.

    ``valid_limit`` is the smoke-test numeric check only (never used for ranking);
    production runs evaluate the full 728-sample valid split every epoch.
    """
    cfg = dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = make_model(architecture, run_data, **cfg.get("model_kwargs", {})).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    shared = shared_parameters(model)
    shared_names = [n for n, _ in shared]
    shared_params = [p for _, p in shared]
    n_train = len(run_data.train["id"]) if train_limit is None else int(train_limit)
    epochs = int(max_epochs or cfg.get("max_epochs", 25))
    patience = int(patience if patience is not None else cfg.get("patience", 5))
    batch_size = int(cfg.get("batch_size", 64))
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_steps = steps_per_epoch * epochs
    opt = build_optimizer(model, cfg.get("optimizer", "O1"), lr=cfg.get("lr"),
                          weight_decay=cfg.get("weight_decay", 1e-4),
                          rho=cfg.get("rho", 0.05), total_steps=total_steps)
    optimizer, scheduler = opt["optimizer"], opt["scheduler"]
    kind = opt["kind"]
    loss_id = cfg.get("loss", "L1")
    reg_w = float(cfg.get("regression_weight", 1.0))
    huber_beta = float(cfg.get("huber_beta", 0.5))
    aug_seed = 10000 + int(seed)
    class_weights = None
    if loss_id == "L2":
        counts = [int((run_data.train["classification_labels"] == k).sum()) for k in range(3)]
        class_weights = class_weights_sqrt_inverse(counts).to(device)
    ema = None
    if cfg.get("ema"):
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()
               if v.dtype.is_floating_point}
    clip = float(cfg.get("gradient_clip", 1.0))
    use_missing_view = bool(cfg.get("missing_view", True))
    history = []
    best = {"score": float("inf"), "epoch": -1, "state": None, "ema_state": None}
    start_epoch = 0
    indices = np.arange(n_train)
    started = time.time()
    if resume_from is not None and Path(resume_from).exists():
        state = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        (optimizer.base_optimizer if kind == "O3" else optimizer).load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
        history = list(state.get("history", []))
        best = state["best"]
        start_epoch = int(state["epoch"]) + 1
        print(json.dumps({"resume": str(resume_from), "start_epoch": start_epoch}), flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        if hasattr(model, "_cache_slot"):
            model.clear_text_cache()
            model._cache_slot = None
        epoch_rng = np.random.default_rng([seed, epoch])
        order = epoch_rng.permutation(indices)
        missing_all = build_epoch_missing(run_data.train["domain"], run_data.train["observed"],
                                          run_data.train["id"], aug_seed, epoch) \
            if use_missing_view else None
        running, parts_sum, n_batches, grad_norm_sum, n_grad = 0.0, {}, 0, 0.0, 0
        cos_samples = []
        for step, start in enumerate(range(0, n_train, batch_size)):
            batch_idx = order[start:start + batch_size]
            batch = run_data.tensors("train", batch_idx)
            vis_full = batch["observed"]
            vis_missing = None
            if missing_all is not None:
                vis_missing = torch.from_numpy(missing_all[batch_idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "O3":
                rng_before = torch.get_rng_state()
                out_full = model(batch["text_bert"], batch["audio"], batch["vision"],
                                 batch["domain"], vis_full)
                loss_a, parts_a = compute_loss(loss_id, out_full, batch, regression_weight=reg_w,
                                               huber_beta=huber_beta, class_weights=class_weights)
                if vis_missing is not None:
                    out_miss = model(batch["text_bert"], batch["audio"], batch["vision"],
                                     batch["domain"], vis_missing)
                    loss_b, parts_b = compute_loss(loss_id, out_miss, batch,
                                                   regression_weight=reg_w, huber_beta=huber_beta,
                                                   class_weights=class_weights)
                    loss = 0.5 * loss_a + 0.5 * loss_b
                else:
                    loss, parts_b = loss_a, {}
                loss.backward()
                optimizer.first_step()
                torch.set_rng_state(rng_before)
                optimizer.zero_grad(set_to_none=True)
                out_full2 = model(batch["text_bert"], batch["audio"], batch["vision"],
                                  batch["domain"], vis_full)
                loss_a2, _ = compute_loss(loss_id, out_full2, batch, regression_weight=reg_w,
                                          huber_beta=huber_beta, class_weights=class_weights)
                if vis_missing is not None:
                    out_miss2 = model(batch["text_bert"], batch["audio"], batch["vision"],
                                      batch["domain"], vis_missing)
                    loss_b2, _ = compute_loss(loss_id, out_miss2, batch, regression_weight=reg_w,
                                              huber_beta=huber_beta, class_weights=class_weights)
                    loss2 = 0.5 * loss_a2 + 0.5 * loss_b2
                else:
                    loss2 = loss_a2
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                optimizer.second_step()
                scheduler.step()
                parts = {f"full_{k}": v for k, v in parts_a.items()}
                parts.update({f"missing_{k}": v for k, v in parts_b.items()})
                parts["total"] = float(loss.detach())
            elif kind == "O4":
                views = [(vis_full, 0.5)]
                if vis_missing is not None:
                    views.append((vis_missing, 0.5))
                grad_ce = {n: torch.zeros_like(p) for n, p in shared}
                grad_reg = {n: torch.zeros_like(p) for n, p in shared}
                loss_total = None
                for view_vis, weight in views:
                    out = model(batch["text_bert"], batch["audio"], batch["vision"],
                                batch["domain"], view_vis)
                    ce = F.cross_entropy(out["logits"], batch["classification_labels"],
                                         weight=class_weights)
                    reg = F.smooth_l1_loss(out["raw_intensity"], batch["regression_labels"],
                                           beta=huber_beta)
                    loss_total = weight * (ce + reg_w * reg) if loss_total is None \
                        else loss_total + weight * (ce + reg_w * reg)
                    g_ce = torch.autograd.grad(ce, shared_params, retain_graph=True,
                                               allow_unused=True)
                    g_reg = torch.autograd.grad(reg, shared_params, retain_graph=True,
                                                allow_unused=True)
                    for name, gc, gr in zip(shared_names, g_ce, g_reg):
                        if gc is not None:
                            grad_ce[name] += weight * gc
                        if gr is not None:
                            grad_reg[name] += weight * gr
                loss_total.backward()
                ga, gb = pcgrad_project([grad_ce[n] for n in shared_names],
                                        [grad_reg[n] for n in shared_names])
                with torch.no_grad():
                    for i, (name, param) in enumerate(shared):
                        param.grad = ga[i] + reg_w * gb[i]
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                optimizer.step()
                scheduler.step()
                loss = loss_total.detach()
                parts = {"total": float(loss)}
            else:
                out_full = model(batch["text_bert"], batch["audio"], batch["vision"],
                                 batch["domain"], vis_full)
                loss, parts = compute_loss(loss_id, out_full, batch, regression_weight=reg_w,
                                           huber_beta=huber_beta, class_weights=class_weights)
                if vis_missing is not None:
                    out_miss = model(batch["text_bert"], batch["audio"], batch["vision"],
                                     batch["domain"], vis_missing)
                    loss_m, parts_m = compute_loss(loss_id, out_miss, batch,
                                                   regression_weight=reg_w, huber_beta=huber_beta,
                                                   class_weights=class_weights)
                    loss = 0.5 * loss + 0.5 * loss_m
                    parts = {f"full_{k}": v for k, v in parts.items()}
                    parts.update({f"missing_{k}": v for k, v in parts_m.items()})
                    parts["total"] = float(loss.detach())
                else:
                    parts = {f"full_{k}": v for k, v in parts.items()}
                if cos_every and (step % int(cos_every) == 0) and shared_params:
                    # diagnostic only: _task_gradients runs its own forward pass, so it must
                    # happen before the main backward frees the training graph
                    with torch.enable_grad():
                        _, _, g_ce, g_reg = _task_gradients(model, batch, vis_full, shared_params,
                                                            huber_beta, class_weights)
                        cos = grad_cosine(g_ce, g_reg)
                    if cos is not None:
                        cos_samples.append(cos)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, clip)
                optimizer.step()
                scheduler.step()
                grad_norm_sum += float(grad_norm)
                n_grad += 1
            if ema is not None:
                with torch.no_grad():
                    state = model.state_dict()
                    for key, value in ema.items():
                        value.mul_(cfg.get("ema_decay", 0.99)).add_(
                            state[key].detach(), alpha=1 - cfg.get("ema_decay", 0.99))
            running += float(loss.detach())
            n_batches += 1
            for key, value in parts.items():
                parts_sum[key] = parts_sum.get(key, 0.0) + value
        train_seconds = time.time() - started
        model.eval()
        set_text_cache(model, None)
        # EMA arm: the stabilised weights are the ones that are validated and, if they win,
        # deployed. Training then resumes from the raw weights.
        raw_state = None
        if ema is not None:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
        per_condition, _ = evaluate_bank(model, run_data, select_bank, split="valid",
                                         limit=valid_limit)
        summary = summarise_bank(per_condition, {c: s for c, s in zip(select_bank["condition_ids"],
                                                                      select_bank["specs"])})
        score = summary["mae"]
        if raw_state is not None:
            model.load_state_dict(raw_state)
        record = {
            "epoch": epoch, "train_loss": running / max(1, n_batches),
            "loss_parts": {k: v / max(1, n_batches) for k, v in parts_sum.items()},
            "grad_norm_mean": (grad_norm_sum / n_grad) if n_grad else None,
            "ce_reg_grad_cosine": cos_samples,
            "select_R_MAE": summary["mae"], "select_R_F1": summary["macro_f1"],
            "select_worst_condition_MAE": summary.get("worst_condition_mae"),
            "elapsed_seconds": train_seconds, "lr": scheduler.base_lrs[0] * scheduler.ratio(),
            "early_stop_subset": None if valid_limit is None else int(valid_limit),
        }
        history.append(record)
        if log_path:
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if progress:
            progress(record)
        if score < best["score"] - cfg.get("min_delta", 1e-4):
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "ema_state": (None if ema is None else
                                  {k: v.detach().cpu().clone() for k, v in ema.items()})}
        if state_path:
            payload = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": (optimizer.base_optimizer if kind == "O3" else optimizer).state_dict(),
                "scheduler": scheduler.state_dict(), "best": best, "history": history,
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                "seed": seed, "architecture": architecture, "config": cfg,
            }
            torch.save(payload, state_path)
        if best["epoch"] >= 0 and epoch - best["epoch"] >= patience:
            break
    if best["ema_state"] is not None:
        # EMA arm: deploy the stabilised weights that were selected on
        model.load_state_dict(best["ema_state"])
    elif best["state"] is not None:
        model.load_state_dict(best["state"])
    result = {
        "architecture": architecture, "seed": seed, "config": cfg,
        "epochs_run": len(history), "best_epoch": best["epoch"],
        "best_select_R_MAE": best["score"], "history": history,
        "trainable_params": trainable_parameter_count(model),
        "non_encoder_params": non_encoder_parameter_count(model),
        "total_seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "amp": False,
        "resumable_state": str(state_path) if state_path else None,
    }
    if best["ema_state"] is not None:
        result["ema_state"] = best["ema_state"]
    return model, result
