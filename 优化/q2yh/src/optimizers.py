"""Optimisers O1-O5 with the mechanism checks the protocol requires."""
from __future__ import annotations

import math

import torch


class WarmupCosine:
    """Linear warmup for the first ``warmup_fraction`` of steps, then cosine decay."""

    def __init__(self, optimizer, total_steps, warmup_fraction=0.1, min_ratio=0.0):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(round(warmup_fraction * self.total_steps)))
        self.min_ratio = min_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

    def ratio(self):
        if self.step_count < self.warmup_steps:
            return self.step_count / self.warmup_steps
        progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def step(self):
        self.step_count += 1
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * self.ratio()

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs,
                "total_steps": self.total_steps, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]


class SAM:
    """Sharpness-aware perturbation with the two passes sharing batch/mask/RNG."""

    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = [p for p in params if p.requires_grad]
        self.base_optimizer = base_optimizer
        self.rho = rho
        self._e_w = None

    def first_step(self):
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):
            raise RuntimeError("SAM requires gradients on every parameter before the ascent step")
        norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        scale = self.rho / (norm + 1e-12)
        self._e_w = [g.detach() * scale for g in grads]
        with torch.no_grad():
            for p, e in zip(self.params, self._e_w):
                p.add_(e)

    def second_step(self):
        with torch.no_grad():
            for p, e in zip(self.params, self._e_w):
                p.sub_(e)
        self._e_w = None
        self.base_optimizer.step()

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


def pcgrad_project(grads_a, grads_b):
    """PCGrad: remove the conflicting component of each task from the other."""
    def dot(x, y):
        return sum((a * b).sum() for a, b in zip(x, y))

    if dot(grads_a, grads_b) < 0:
        # the coefficients are scalars over the whole parameter vector: mixing a single
        # tensor with a list here would silently truncate inside zip()
        denom_b = dot(grads_b, grads_b).clamp(min=1e-12)
        coef_a = dot(grads_a, grads_b) / denom_b
        denom_a = dot(grads_a, grads_a).clamp(min=1e-12)
        coef_b = dot(grads_b, grads_a) / denom_a
        ga = [x - coef_a * y for x, y in zip(grads_a, grads_b)]
        gb = [y - coef_b * x for x, y in zip(grads_a, grads_b)]
    else:
        ga = [g.clone() for g in grads_a]
        gb = [g.clone() for g in grads_b]
    return ga, gb


def shared_parameters(model):
    """PCGrad scope: parameters used by both heads, i.e. the shared trunk."""
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("text_encoder.") \
                and ("class_head" not in name) and ("intensity_head" not in name):
            names.append((name, param))
    return names


def build_optimizer(model, optimizer_id, lr=None, weight_decay=1e-4, momentum=0.9,
                    rho=0.05, total_steps=1):
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_id == "O1":
        base = torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=weight_decay)
        return {"optimizer": base, "scheduler": WarmupCosine(base, total_steps), "kind": "O1"}
    if optimizer_id == "O2":
        base = torch.optim.SGD(params, lr=lr or 0.03, momentum=momentum,
                              nesterov=True, weight_decay=weight_decay)
        return {"optimizer": base, "scheduler": WarmupCosine(base, total_steps), "kind": "O2"}
    if optimizer_id == "O3":
        base = torch.optim.AdamW(params, lr=lr or 5e-4, weight_decay=weight_decay)
        sam = SAM(params, base, rho=rho)
        return {"optimizer": sam, "scheduler": WarmupCosine(base, total_steps), "kind": "O3"}
    if optimizer_id == "O4":
        base = torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=weight_decay)
        return {"optimizer": base, "scheduler": WarmupCosine(base, total_steps), "kind": "O4"}
    if optimizer_id == "O5":
        base = torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=weight_decay)
        return {"optimizer": base, "scheduler": WarmupCosine(base, total_steps), "kind": "O5"}
    raise ValueError(f"Unknown optimizer: {optimizer_id}")