"""Deployable frozen text adapter; use identically on Attachment2 and Attachment3."""
from pathlib import Path
import torch
from torch import nn
from transformers import BertModel


class FrozenTextEncoder(nn.Module):
    def __init__(self, model_dir=None):
        super().__init__()
        model_dir = model_dir or Path(__file__).resolve().parents[1] / "resources/bert_shared"
        self.bert = BertModel.from_pretrained(str(model_dir), local_files_only=True)
        self.bert.requires_grad_(False)
        self.bert.eval()
        self.output_dim = self.bert.config.hidden_size

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()
        return self

    def forward(self, text_bert, domain, visible):
        """text_bert [B,3,T]; domain/visible [B,T]; masks fixed before corruption."""
        tb = text_bert.long().clone()
        domain, visible = domain.bool(), visible.bool()
        if tb.ndim != 3 or tb.shape[1] != 3 or domain.shape != visible.shape:
            raise ValueError("Invalid text adapter input")
        if tb.shape[0] != len(domain) or tb.shape[2] != domain.shape[1] or (visible & ~domain).any():
            raise ValueError("Text visibility must be inside the original domain")
        hidden = domain & ~visible
        special = ~domain & ((tb[:, 0] == 101) | (tb[:, 0] == 102))
        tb[:, 0] = torch.where(hidden, 103, tb[:, 0])
        tb[:, 0] = torch.where(~domain & ~special, 0, tb[:, 0])
        tb[:, 1] = (visible | special).long()
        tb[:, 2] = torch.where(hidden | (~domain & ~special), 0, tb[:, 2])
        self.bert.eval()
        with torch.no_grad():
            encoded = self.bert(input_ids=tb[:, 0], attention_mask=tb[:, 1],
                                token_type_ids=tb[:, 2]).last_hidden_state
        return encoded * visible.unsqueeze(-1).to(encoded.dtype)
