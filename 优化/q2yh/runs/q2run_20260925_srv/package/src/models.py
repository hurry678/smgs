"""Mask-aware fusion models (server task T02).

All models share the fixed signature

    forward(text_bert, audio, vision, domain, visible) -> dict

with keys ``logits`` [B,3], ``raw_intensity`` [B] and ``diagnostics``.  Padding and
special positions are support only: they never enter the pooled evidence.
"""
from __future__ import annotations

import inspect

import hashlib
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from text_encoder import FrozenTextEncoder

CONTENT_DIM = 128 + 74 + 35
STRUCTURE_DIM = 26


# --------------------------------------------------------------------------
# padding-invariant structure features (torch mirror of protocol_core.gap_structure)
# --------------------------------------------------------------------------

def gap_structure_torch(domain, visible):
    """[B,T,26]; identical to the NumPy reference for the same boolean inputs."""
    d = domain.bool()
    m = visible.bool()
    if m.ndim != 3 or d.shape != m.shape[:1] + m.shape[2:]:
        raise ValueError("Expected domain [B,T] and visible [B,3,T]")
    b, _, t = m.shape
    device, dtype = m.device, torch.float32
    pos = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
    length = d.sum(-1, keepdim=True).clamp(min=0).to(dtype)          # [B,1]
    safe_len = length.clamp(min=1.0)
    out = torch.zeros((b, t, STRUCTURE_DIM), device=device, dtype=dtype)

    obs_idx = torch.where(m, pos.expand(b, 3, t), torch.full_like(pos, -1.0).expand(b, 3, t))
    last_obs = torch.cummax(obs_idx, dim=-1).values
    rev = torch.where(m, pos.expand(b, 3, t), torch.full_like(pos, t).expand(b, 3, t))
    next_obs = torch.flip(torch.cummin(torch.flip(rev, dims=[-1]), dim=-1).values, dims=[-1])

    miss = d.unsqueeze(1) & ~m
    # length of the maximal contiguous missing run containing each position (0 if observed);
    # a sequential scan over T keeps the torch mirror bit-identical to the NumPy reference
    left_run = torch.zeros((b, 3, t), device=device, dtype=dtype)
    acc = torch.zeros((b, 3), device=device, dtype=dtype)
    for step in range(t):
        acc = torch.where(miss[:, :, step], acc + 1.0, torch.zeros_like(acc))
        left_run[:, :, step] = acc
    right_run = torch.zeros_like(left_run)
    acc = torch.zeros_like(acc)
    for step in range(t - 1, -1, -1):
        acc = torch.where(miss[:, :, step], acc + 1.0, torch.zeros_like(acc))
        right_run[:, :, step] = acc
    run = torch.where(miss, left_run + right_run - 1.0, torch.zeros_like(left_run))

    local = torch.zeros((b, 3, t), device=device, dtype=dtype)
    for shift in (-2, -1, 0, 1, 2):
        rolled = torch.roll(m.to(dtype), shifts=shift, dims=-1)
        if shift > 0:
            rolled[..., :shift] = 0
        elif shift < 0:
            rolled[..., shift:] = 0
        local = local + rolled * d.unsqueeze(1).to(dtype)
    window = torch.zeros((b, 1, t), device=device, dtype=dtype)
    for shift in (-2, -1, 0, 1, 2):
        rolled = torch.roll(d.to(dtype).unsqueeze(1), shifts=shift, dims=-1)
        if shift > 0:
            rolled[..., :shift] = 0
        elif shift < 0:
            rolled[..., shift:] = 0
        window = window + rolled
    local = local / window.clamp(min=1.0)

    # the "no observation at all" sentinel is 1.0 *after* normalisation, exactly as in
    # protocol_core.gap_structure; using 1.0 here would be divided down to 1/length
    ones = safe_len.unsqueeze(1).expand_as(last_obs)
    dl = torch.where(last_obs < 0, ones, pos - last_obs) / safe_len.unsqueeze(1)
    dr = torch.where(next_obs > t - 1, ones, next_obs - pos) / safe_len.unsqueeze(1)
    coverage = m.to(dtype).sum(-1, keepdim=True) / safe_len.unsqueeze(1)

    for mod in range(3):
        base = 6 * mod
        out[..., base + 0] = m[:, mod].to(dtype)
        out[..., base + 1] = local[:, mod]
        out[..., base + 2] = run[:, mod] / safe_len
        out[..., base + 3] = dl[:, mod]
        out[..., base + 4] = dr[:, mod]
        out[..., base + 5] = coverage[:, mod, 0].unsqueeze(-1).expand(b, t)

    state = m[:, 0].long() + 2 * m[:, 1].long() + 4 * m[:, 2].long()
    onehot = F.one_hot(state, num_classes=8).to(dtype)
    out[..., 18:26] = onehot
    return out * d.unsqueeze(-1).to(dtype)


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

class TimePool(nn.Module):
    """Masked mean+max over usable timesteps with an explicit no-evidence branch."""

    def __init__(self, dim):
        super().__init__()
        self.no_evidence = nn.Parameter(torch.zeros(2 * dim))

    def forward(self, hidden, usable):
        mask = usable.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(1).clamp(min=1.0)
        mean = (hidden * mask).sum(1) / denom
        filled = hidden.masked_fill(~usable.unsqueeze(-1), float("-inf"))
        maximum = filled.max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        pooled = torch.cat((mean, maximum), -1)
        empty = ~usable.any(1)
        pooled = torch.where(empty.unsqueeze(-1), self.no_evidence.unsqueeze(0), pooled)
        return pooled, empty


def _text_fingerprint(text_bert, domain, visible_text):
    """Digest of the adapter request; the cache key can never alias two views."""
    h = hashlib.sha256()
    for arr in (text_bert, domain, visible_text):
        h.update(np.ascontiguousarray(arr.detach().to("cpu").numpy()).tobytes())
        h.update(b"|")
    return h.hexdigest()


def masked_modality_mean(feature, mask):
    m = mask.unsqueeze(-1).to(feature.dtype)
    return (feature * m).sum(1) / m.sum(1).clamp(min=1.0)


# --------------------------------------------------------------------------
# dual heads
# --------------------------------------------------------------------------

class DualHead(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout))
        self.class_head = nn.Linear(64, 3)
        self.intensity_head = nn.Linear(64, 1)

    def forward(self, pooled):
        h = self.trunk(pooled)
        return {"logits": self.class_head(h), "raw_intensity": self.intensity_head(h).squeeze(-1)}


class LocalTimeAttention(nn.Module):
    """Single-layer multi-head attention restricted to a +/-radius window inside D."""

    def __init__(self, dim, heads=4, radius=5, dropout=0.1):
        super().__init__()
        self.heads, self.radius = heads, radius
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden, domain):
        b, t, d = hidden.shape
        h = self.heads
        q = self.q(hidden).view(b, t, h, d // h).transpose(1, 2)
        k = self.k(hidden).view(b, t, h, d // h).transpose(1, 2)
        v = self.v(hidden).view(b, t, h, d // h).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d // h)
        idx = torch.arange(t, device=hidden.device)
        band = (idx[:, None] - idx[None, :]).abs() <= self.radius
        allowed = band[None, None] & domain[:, None, None, :]
        allowed = allowed | torch.eye(t, dtype=torch.bool, device=hidden.device)[None, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = self.drop(torch.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return self.norm(hidden + self.out(attended))


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------

class TrainingPriorModel(nn.Module):
    """B0: non-trainable training prior (class prior + class-conditional mean intensity)."""

    def __init__(self, class_counts, class_intensity_mean):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float64).clamp(min=1e-9)
        prior = counts / counts.sum()
        self.register_buffer("log_prior", torch.log(prior).to(torch.float32))
        self.register_buffer("class_intensity",
                             torch.as_tensor(class_intensity_mean, dtype=torch.float32))

    def forward(self, text_bert, audio, vision, domain, visible):
        b = domain.shape[0]
        logits = self.log_prior.unsqueeze(0).expand(b, 3).clone()
        raw = (torch.softmax(logits, -1) * self.class_intensity.unsqueeze(0)).sum(-1)
        return {"logits": logits, "raw_intensity": raw,
                "diagnostics": {"trainable": torch.zeros((), device=domain.device)}}


class MaskedMeanMLP(nn.Module):
    """B2: per-modality masked mean, concatenated, dual-head MLP. No mask side channel."""

    def __init__(self, dropout=0.15):
        super().__init__()
        self.text_encoder = FrozenTextEncoder()
        self.head = DualHead(CONTENT_DIM, dropout)
        self._cache_slot = None
        self._text_cache = {}

    def clear_text_cache(self):
        self._text_cache.clear()

    def encode_text(self, text_bert, domain, visible):
        """Cache only an identical (tokens, domain, text visibility) request.

        A masked view can never reuse the full-text hidden state: the fingerprint
        covers the text channel, so a different visibility forces a re-encode.
        """
        slot = self._cache_slot
        if slot is None:
            return self.text_encoder(text_bert, domain, visible)
        key = (slot, _text_fingerprint(text_bert, domain, visible))
        hit = self._text_cache.get(key)
        if hit is not None:
            return hit
        hidden = self.text_encoder(text_bert, domain, visible)
        self._text_cache[key] = hidden
        return hidden

    def forward(self, text_bert, audio, vision, domain, visible):
        hidden = self.encode_text(text_bert, domain, visible[:, 0])
        pooled = torch.cat((
            masked_modality_mean(hidden, visible[:, 0]),
            masked_modality_mean(audio, visible[:, 1]),
            masked_modality_mean(vision, visible[:, 2]),
        ), -1)
        out = self.head(pooled)
        coverage = visible.float().sum(-1) / domain.float().sum(-1, keepdim=True).clamp(min=1.0)
        out["diagnostics"] = {"coverage": coverage.detach(),
                              "no_evidence": ~visible.any(1).any(1)}
        return out


class TextMeanLinear(nn.Module):
    """B1: frozen tiny-BERT masked mean with a linear dual head (text only)."""

    def __init__(self, dropout=0.15):
        super().__init__()
        self.text_encoder = FrozenTextEncoder()
        self.drop = nn.Dropout(dropout)
        self.trunk = nn.Linear(128, 64)
        self.class_head = nn.Linear(64, 3)
        self.intensity_head = nn.Linear(64, 1)
        self._cache_slot = None
        self._text_cache = {}

    def clear_text_cache(self):
        self._text_cache.clear()

    def encode_text(self, text_bert, domain, visible):
        """Cache only an identical (tokens, domain, text visibility) request.

        A masked view can never reuse the full-text hidden state: the fingerprint
        covers the text channel, so a different visibility forces a re-encode.
        """
        slot = self._cache_slot
        if slot is None:
            return self.text_encoder(text_bert, domain, visible)
        key = (slot, _text_fingerprint(text_bert, domain, visible))
        hit = self._text_cache.get(key)
        if hit is not None:
            return hit
        hidden = self.text_encoder(text_bert, domain, visible)
        self._text_cache[key] = hidden
        return hidden

    def forward(self, text_bert, audio, vision, domain, visible):
        hidden = self.encode_text(text_bert, domain, visible[:, 0])
        pooled = masked_modality_mean(hidden, visible[:, 0])
        h = self.drop(torch.relu(self.trunk(pooled)))
        coverage = visible.float().sum(-1) / domain.float().sum(-1, keepdim=True).clamp(min=1.0)
        return {"logits": self.class_head(h), "raw_intensity": self.intensity_head(h).squeeze(-1),
                "diagnostics": {"coverage": coverage.detach(),
                                "no_evidence": ~visible.any(1).any(1)}}


class BigruFusion(nn.Module):
    """B3/M1/M2/M3 backbone: padding-aware BiGRU + residual-evidence pooling.

    ``mask_side_channel`` selects the per-timestep auxiliary channels:
    0 -> none (B3), 3 -> binary visibility (M1), 26 -> gap structure (M2/M3).
    """

    def __init__(self, mask_side_channel=0, hidden=64, dropout=0.15, attention=False,
                 heads=4, radius=5, layers=1):
        super().__init__()
        if mask_side_channel not in (0, 3, 26):
            raise ValueError("mask_side_channel must be 0, 3 or 26")
        self.mask_side_channel = mask_side_channel
        self.attention = attention
        self.text_encoder = FrozenTextEncoder()
        in_dim = CONTENT_DIM + mask_side_channel
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.0 if layers == 1 else dropout)
        self.attn = LocalTimeAttention(2 * hidden, heads=heads, radius=radius) if attention else None
        self.pool = TimePool(2 * hidden)
        self.head = DualHead(4 * hidden, dropout)
        self._cache_slot = None
        self._text_cache = {}

    def clear_text_cache(self):
        self._text_cache.clear()

    def encode_text(self, text_bert, domain, visible):
        """Cache only an identical (tokens, domain, text visibility) request.

        A masked view can never reuse the full-text hidden state: the fingerprint
        covers the text channel, so a different visibility forces a re-encode.
        """
        slot = self._cache_slot
        if slot is None:
            return self.text_encoder(text_bert, domain, visible)
        key = (slot, _text_fingerprint(text_bert, domain, visible))
        hit = self._text_cache.get(key)
        if hit is not None:
            return hit
        hidden = self.text_encoder(text_bert, domain, visible)
        self._text_cache[key] = hidden
        return hidden

    def forward(self, text_bert, audio, vision, domain, visible):
        hidden = self.encode_text(text_bert, domain, visible[:, 0])
        content = torch.cat((
            hidden * visible[:, 0].unsqueeze(-1).to(hidden.dtype),
            audio * visible[:, 1].unsqueeze(-1).to(audio.dtype),
            vision * visible[:, 2].unsqueeze(-1).to(vision.dtype),
        ), -1)
        if self.mask_side_channel == 3:
            content = torch.cat((content, visible.to(content.dtype).permute(0, 2, 1)), -1)
        elif self.mask_side_channel == 26:
            content = torch.cat((content, gap_structure_torch(domain, visible)), -1)
        # hard visibility constraint: unobserved positions can never leak into the GRU,
        # even if a caller passes a non-zero placeholder there
        evidence = (domain & visible.any(1)).unsqueeze(-1).to(content.dtype)
        content = content * evidence
        encoded, _ = self.gru(content)
        encoded = encoded * domain.unsqueeze(-1).to(encoded.dtype)
        if self.attn is not None:
            encoded = self.attn(encoded, domain)
        usable = domain & visible.any(1)
        pooled, empty = self.pool(encoded, usable)
        out = self.head(pooled)
        coverage = visible.float().sum(-1) / domain.float().sum(-1, keepdim=True).clamp(min=1.0)
        out["diagnostics"] = {
            "coverage": coverage.detach(),
            "no_evidence": empty.detach(),
            "pooled_norm": pooled.norm(dim=-1).detach(),
        }
        return out


ARCHITECTURES = {
    "training_prior": TrainingPriorModel,
    "tinybert_masked_mean_linear_dual_head": TextMeanLinear,
    "three_modality_masked_mean_concat_mlp": MaskedMeanMLP,
    "bigru_content_gate_time_pool": lambda **kw: BigruFusion(mask_side_channel=0, **kw),
    "bigru_binary_mask_gate_time_pool": lambda **kw: BigruFusion(mask_side_channel=3, **kw),
    "bigru_domain_safe_gap_structure_gate": lambda **kw: BigruFusion(mask_side_channel=26, **kw),
    "M2_plus_local_cross_time_attention": lambda **kw: BigruFusion(
        mask_side_channel=26, attention=True, **kw),
}


def trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def non_encoder_parameter_count(model):
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith("text_encoder."))


def build_model(architecture, **kwargs):
    """Build a candidate, passing only the kwargs its constructor actually accepts.

    The shared training config carries hidden/dropout for every candidate, but the
    mean-pooling baselines fix their own trunk width; dropping unsupported keys keeps
    one config dict valid for all architectures.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    factory = ARCHITECTURES[architecture]
    accepted = inspect.signature(factory).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return factory(**kwargs)