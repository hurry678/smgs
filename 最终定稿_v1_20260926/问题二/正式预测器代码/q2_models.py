#!/usr/bin/env python3
"""Q2 model implementations for the E-problem.

The server training environment has PyTorch but not a usable transformers
import.  The BERT-compatible module below loads the provided MiniLM BERT
weights directly with torch.nn and is used only as a frozen text encoder.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertLayer(nn.Module):
    def __init__(self, hidden: int = 384, heads: int = 12, intermediate: int = 1536, eps: float = 1e-12):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.attn_out = nn.Linear(hidden, hidden)
        self.attn_ln = nn.LayerNorm(hidden, eps=eps)
        self.intermediate = nn.Linear(hidden, intermediate)
        self.output = nn.Linear(intermediate, hidden)
        self.output_ln = nn.LayerNorm(hidden, eps=eps)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.query(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # mask is [B,T], 1=valid.  Broadcast over heads and query positions.
        scores = scores.masked_fill(mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, t, self.hidden)
        x = self.attn_ln(x + self.dropout(self.attn_out(context)))
        x = self.output_ln(x + self.dropout(self.output(F.gelu(self.intermediate(x)))))
        return x


class BertEncoder(nn.Module):
    def __init__(self, hidden: int = 384, layers: int = 6, heads: int = 12, intermediate: int = 1536, vocab: int = 30522):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, hidden)
        self.position_embeddings = nn.Embedding(512, hidden)
        self.token_type_embeddings = nn.Embedding(2, hidden)
        self.embedding_ln = nn.LayerNorm(hidden, eps=1e-12)
        self.embedding_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([BertLayer(hidden, heads, intermediate) for _ in range(layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        positions = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions) + self.token_type_embeddings(torch.zeros_like(input_ids))
        x = self.embedding_dropout(self.embedding_ln(x))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x

    def load_hf_state(self, path: Path) -> None:
        state = torch.load(str(path), map_location="cpu")
        # HF key names -> compact local names.
        with torch.no_grad():
            self.word_embeddings.weight.copy_(state["embeddings.word_embeddings.weight"])
            self.position_embeddings.weight.copy_(state["embeddings.position_embeddings.weight"])
            self.token_type_embeddings.weight.copy_(state["embeddings.token_type_embeddings.weight"])
            self.embedding_ln.weight.copy_(state["embeddings.LayerNorm.weight"])
            self.embedding_ln.bias.copy_(state["embeddings.LayerNorm.bias"])
            for i, layer in enumerate(self.layers):
                p = f"encoder.layer.{i}."
                layer.query.weight.copy_(state[p + "attention.self.query.weight"])
                layer.query.bias.copy_(state[p + "attention.self.query.bias"])
                layer.key.weight.copy_(state[p + "attention.self.key.weight"])
                layer.key.bias.copy_(state[p + "attention.self.key.bias"])
                layer.value.weight.copy_(state[p + "attention.self.value.weight"])
                layer.value.bias.copy_(state[p + "attention.self.value.bias"])
                layer.attn_out.weight.copy_(state[p + "attention.output.dense.weight"])
                layer.attn_out.bias.copy_(state[p + "attention.output.dense.bias"])
                layer.attn_ln.weight.copy_(state[p + "attention.output.LayerNorm.weight"])
                layer.attn_ln.bias.copy_(state[p + "attention.output.LayerNorm.bias"])
                layer.intermediate.weight.copy_(state[p + "intermediate.dense.weight"])
                layer.intermediate.bias.copy_(state[p + "intermediate.dense.bias"])
                layer.output.weight.copy_(state[p + "output.dense.weight"])
                layer.output.bias.copy_(state[p + "output.dense.bias"])
                layer.output_ln.weight.copy_(state[p + "output.LayerNorm.weight"])
                layer.output_ln.bias.copy_(state[p + "output.LayerNorm.bias"])
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class TemporalEncoder(nn.Module):
    """Mask-aware projection, positional encoding and a one-layer BiGRU."""
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, 50, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = h * obs.unsqueeze(-1).to(h.dtype)
        h, _ = self.gru(h)
        return self.dropout(h * obs.unsqueeze(-1).to(h.dtype))


class TextEncoder(nn.Module):
    def __init__(self, model_dir: Path, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bert = BertEncoder()
        self.bert.load_hf_state(model_dir / "pytorch_model.bin")
        self.temporal = TemporalEncoder(384, hidden, dropout)

    def forward(self, input_ids: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # BERT itself must never see padding as evidence.  For an entirely
        # unavailable text modality, feed a harmless [CLS] token so that no
        # all-masked softmax is produced; the modality mask remains zero.
        bert_ids = torch.where(obs, input_ids, torch.full_like(input_ids, 103))
        bert_mask = obs.clone()
        empty = ~bert_mask.any(dim=1)
        if empty.any():
            bert_ids[empty, 0] = 101
            bert_mask[empty, 0] = True
        h = self.bert(bert_ids, bert_mask.long())
        return self.temporal(h, obs)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    out = F.softmax(scores, dim=dim)
    return out * mask.to(out.dtype)


class CrossModalBlock(nn.Module):
    """Mask-aware cross-modal attention at every aligned time position."""

    def __init__(self, hidden: int, struct_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, 1, hidden))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.struct_proj = nn.Sequential(nn.Linear(struct_dim, hidden), nn.GELU())
        self.norm1 = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        b, m, t, d = h.shape
        if m != 3:
            raise ValueError("CrossModalBlock expects exactly three modalities")
        x = h.permute(0, 2, 1, 3)
        x = x + self.modality_embedding.permute(0, 2, 1, 3)
        x = x + self.struct_proj(q).unsqueeze(2)
        valid = mask.permute(0, 2, 1)
        fallback = ~valid.any(dim=-1, keepdim=True)
        key_valid = valid | fallback
        x_norm = self.norm1(x)
        qh = self.q_proj(x_norm).view(b, t, 3, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kh = self.k_proj(x_norm).view(b, t, 3, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        vh = self.v_proj(x_norm).view(b, t, 3, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~key_valid[:, :, None, None, :], torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = attn * key_valid[:, :, None, None, :].to(attn.dtype)
        context = torch.matmul(attn, vh).permute(0, 1, 3, 2, 4).contiguous().view(b, t, 3, d)
        x = x + self.dropout(self.out_proj(context))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        x = x * valid.unsqueeze(-1).to(x.dtype)
        return x.permute(0, 2, 1, 3)


class TaskHeads(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, 3)
        self.reg = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(z)
        return self.cls(z), 3.0 * torch.tanh(self.reg(z).squeeze(-1))


class Q2Model(nn.Module):
    """Unified model family.

    kind:
      B2: text-only
      B3: simple masked mean concatenation
      B4: BiGRU + ordinary dynamic gate, capacity matched to M0
      B5: BiGRU + low-rank pairwise bilinear fusion
      M0: B4 with continuous missing structure q_t
      M2: M0 plus mask-aware cross-modal attention at each time position
    """
    def __init__(self, kind: str, model_dir: Path, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.kind = kind
        self.hidden = hidden
        self.text = TextEncoder(model_dir, hidden, dropout)
        self.audio = TemporalEncoder(74, hidden, dropout)
        self.vision = TemporalEncoder(35, hidden, dropout)
        self.heads = TaskHeads(hidden, dropout)
        self.struct_dim = 34
        if kind == "B4":
            self.control = nn.ModuleList([nn.Linear(hidden, self.struct_dim) for _ in range(3)])
            self.gate = nn.Sequential(nn.Linear(hidden + self.struct_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            self.time_control = nn.Linear(hidden, self.struct_dim)
            self.time_score = nn.Sequential(nn.Linear(hidden + self.struct_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        elif kind in ("M0", "M2"):
            self.gate = nn.Sequential(nn.Linear(hidden + self.struct_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            self.time_score = nn.Sequential(nn.Linear(hidden + self.struct_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            if kind == "M2":
                self.cross = CrossModalBlock(hidden, self.struct_dim, heads=4, dropout=dropout)
        elif kind == "B5":
            self.rank = 8
            self.pair_left = nn.ModuleList([nn.Linear(hidden, self.rank, bias=False) for _ in range(3)])
            self.pair_right = nn.ModuleList([nn.Linear(hidden, self.rank, bias=False) for _ in range(3)])
            self.fuse = nn.Sequential(nn.Linear(3 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        elif kind == "B3":
            self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())

    @staticmethod
    def _run_stats(obs: torch.Tensor) -> torch.Tensor:
        """Return [B,T,3] with missing run length (only at missing positions)."""
        b, t = obs.shape
        run = torch.zeros((b, t), device=obs.device, dtype=torch.float32)
        cur = torch.zeros((b,), device=obs.device, dtype=torch.float32)
        out = []
        for j in range(t):
            cur = (cur + 1.0) * (~obs[:, j]).to(torch.float32)
            out.append(cur)
        return torch.stack(out, dim=1)

    @staticmethod
    def _boundary_distance(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t = obs.shape
        left = torch.empty((b, t), device=obs.device, dtype=torch.float32)
        right = torch.empty((b, t), device=obs.device, dtype=torch.float32)
        for j in range(t):
            left[:, j] = j + 1
            right[:, t - 1 - j] = j + 1
        for j in range(1, t):
            left[:, j] = torch.where(obs[:, j] == obs[:, j - 1], left[:, j - 1] + 1.0, torch.ones_like(left[:, j]))
        for j in range(t - 2, -1, -1):
            right[:, j] = torch.where(obs[:, j] == obs[:, j + 1], right[:, j + 1] + 1.0, torch.ones_like(right[:, j]))
        return left / t, right / t

    def structure(self, obs: torch.Tensor) -> torch.Tensor:
        """Continuous local/global missing structure q_t, [B,T,34]."""
        b, m, t = obs.shape
        pieces = []
        for i in range(3):
            o = obs[:, i]
            local = F.avg_pool1d(o.float().unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
            runs = self._run_stats(o)
            left, right = self._boundary_distance(o)
            # Equivalent vectorized form: runs[j] is the current
            # consecutive missing length at j, so a run starts exactly when
            # runs[j] == 1.  This removes the per-batch Python loop without
            # changing the q_t definition.
            max_run = runs.max(dim=1).values
            n_runs = ((runs == 1.0) & (runs > 0)).sum(dim=1).float()
            max_run = max_run[:, None].expand(-1, t) / t
            n_runs = n_runs[:, None].expand(-1, t) / t
            pieces.extend([
                o.float().unsqueeze(-1),
                local.unsqueeze(-1),
                (runs / t).unsqueeze(-1),
                left.unsqueeze(-1),
                right.unsqueeze(-1),
                o.float().mean(1, keepdim=True).expand(-1, t).unsqueeze(-1),
                max_run.unsqueeze(-1),
                n_runs.unsqueeze(-1),
            ])
        nobs = obs.sum(1).float()
        pieces.append((nobs / 3.0).unsqueeze(-1))
        pieces.append((1.0 - nobs / 3.0).unsqueeze(-1))
        pieces.append((nobs <= 1.0).float().unsqueeze(-1))
        # 7-way observed subset one-hot.
        idx = (obs[:, 0].long() + 2 * obs[:, 1].long() + 4 * obs[:, 2].long())
        onehot = F.one_hot(idx, num_classes=8).float()
        # Exclude the empty subset from the seven non-empty categories.
        onehot = onehot[..., 1:]
        pieces.append(onehot)
        q = torch.cat(pieces, dim=-1)
        return q

    def encode(self, text_ids: torch.Tensor, obs: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ht = self.text(text_ids, obs[:, 0])
        ha = self.audio(audio, obs[:, 1])
        hv = self.vision(vision, obs[:, 2])
        return ht, ha, hv

    def forward(self, text_ids: torch.Tensor, obs: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> Dict[str, torch.Tensor]:
        ht, ha, hv = self.encode(text_ids, obs, audio, vision)
        h = torch.stack([ht, ha, hv], dim=1)  # [B,3,T,d]
        b, m, t, d = h.shape
        modality_obs = obs.permute(0, 2, 1)  # [B,T,3]
        any_obs = modality_obs.any(dim=2)
        q = self.structure(obs) if self.kind in ("M0", "M2") else None
        if self.kind == "B2":
            zbar = masked_mean(ht, obs[:, 0])
        elif self.kind == "B3":
            pooled = torch.stack([masked_mean(ht, obs[:, 0]), masked_mean(ha, obs[:, 1]), masked_mean(hv, obs[:, 2])], dim=1).reshape(b, -1)
            zbar = self.fuse(pooled)
        elif self.kind in ("B4", "M0", "M2"):
            if self.kind == "B4":
                q_control = torch.stack([torch.tanh(p(h)) for p, h in zip(self.control, [ht, ha, hv])], dim=1)  # [B,3,T,qdim]
                q_use = q_control
            else:
                q_use = q.unsqueeze(1).expand(-1, 3, -1, -1)  # [B,3,T,qdim]
            h_fuse = self.cross(h, obs, q) if self.kind == "M2" else h
            gate_input = torch.cat([h_fuse, q_use], dim=-1)  # [B,3,T,2d]
            r = self.gate(gate_input).squeeze(-1)  # [B,3,T]
            w = masked_softmax(r.permute(0, 2, 1), modality_obs, dim=2)  # [B,T,3]
            z = (w.unsqueeze(-1) * h_fuse.permute(0, 2, 1, 3)).sum(dim=2)  # [B,T,d]
            if self.kind == "B4":
                q_time = torch.tanh(self.time_control(z))  # same-capacity control, no structure
                time_input = torch.cat([z, q_time], dim=-1)
            else:
                time_input = torch.cat([z, q], dim=-1)
            score = self.time_score(time_input).squeeze(-1)
            beta = masked_softmax(score, any_obs, dim=1)
            zbar = (beta.unsqueeze(-1) * z).sum(dim=1)
        elif self.kind == "B5":
            pair_feats = []
            for i in range(3):
                for j in range(i + 1, 3):
                    li = self.pair_left[i](h[:, i])
                    rj = self.pair_right[j](h[:, j])
                    pair = (li * rj).sum(dim=-1)
                    pair = pair * modality_obs[:, :, i] * modality_obs[:, :, j]
                    pair = ((pair * any_obs).sum(dim=1) / any_obs.sum(dim=1).clamp_min(1.0)).unsqueeze(-1)
                    pair_feats.append(pair)
            pooled = [masked_mean(ht, obs[:, 0]), masked_mean(ha, obs[:, 1]), masked_mean(hv, obs[:, 2])]
            feat = torch.cat(pooled + pair_feats, dim=-1)
            zbar = self.fuse(feat)
        else:
            raise ValueError(f"unknown model kind {self.kind}")
        logits, raw = self.heads(zbar)
        # No available evidence: use a zero representation, which is handled by
        # the caller's training-prior fallback when required.
        return {"logits": logits, "raw_intensity": raw, "z": zbar, "obs": obs}

