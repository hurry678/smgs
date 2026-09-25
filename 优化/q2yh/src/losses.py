"""Task losses L0-L4 (server task T02)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOSSES = ("L0", "L1", "L2", "L3", "L4")


def class_weights_sqrt_inverse(counts):
    """L2: sqrt-inverse train frequency, normalised to mean 1."""
    c = torch.as_tensor(counts, dtype=torch.float64).clamp(min=1.0)
    w = c.rsqrt()
    w = w / w.mean()
    return w.to(torch.float32)


def sign_penalty(raw, target_class, margin=0.05):
    """L3: discourage intensity signs that contradict the true polarity."""
    neg = torch.relu(raw + margin)
    pos = torch.relu(margin - raw)
    neu = raw.abs()
    return torch.where(target_class == 0, neg, torch.where(target_class == 2, pos, neu)).mean()


def compute_loss(loss_id, outputs, batch, *, regression_weight=1.0, huber_beta=0.5,
                 class_weights=None, consistency_weight=0.05, sign_margin=0.05):
    """Return (total, parts). ``batch`` holds classification/regression targets."""
    logits, raw = outputs["logits"], outputs["raw_intensity"]
    y, yr = batch["classification_labels"], batch["regression_labels"]
    ce = F.cross_entropy(logits, y, weight=class_weights)
    parts = {"ce": float(ce.detach())}
    if loss_id == "L0":
        reg = F.mse_loss(raw, yr)
        parts["regression"] = float(reg.detach())
        total = ce + regression_weight * reg
    elif loss_id in ("L1", "L3"):
        reg = F.smooth_l1_loss(raw, yr, beta=huber_beta)
        parts["regression"] = float(reg.detach())
        total = ce + regression_weight * reg
        if loss_id == "L3":
            pen = sign_penalty(raw, y, sign_margin)
            parts["sign_penalty"] = float(pen.detach())
            total = total + consistency_weight * pen
    elif loss_id == "L2":
        reg = F.smooth_l1_loss(raw, yr, beta=huber_beta)
        parts["regression"] = float(reg.detach())
        total = ce + regression_weight * reg
    elif loss_id == "L4":
        raise NotImplementedError("L4 is a P2 class-conditional head, not a loss swap")
    else:
        raise ValueError(f"Unknown loss: {loss_id}")
    parts["total"] = float(total.detach())
    return total, parts


def dual_view_loss(loss_id, full_outputs, missing_outputs, batch, **kwargs):
    """Each epoch uses one full view and one synthetic-missing view, weighted 0.5/0.5."""
    full, parts_full = compute_loss(loss_id, full_outputs, batch, **kwargs)
    miss, parts_miss = compute_loss(loss_id, missing_outputs, batch, **kwargs)
    total = 0.5 * full + 0.5 * miss
    parts = {f"full_{k}": v for k, v in parts_full.items()}
    parts.update({f"missing_{k}": v for k, v in parts_miss.items()})
    parts["total"] = float(total.detach())
    return total, parts