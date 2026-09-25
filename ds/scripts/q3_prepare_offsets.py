#!/usr/bin/env python3
"""Prepare exact tokenizer offsets for Q3 evidence mapping.

Run with the base environment because the training environment has an
incomplete transformers installation.  This script performs data preparation
only: it recomputes official tokenizer outputs, checks them against every
stored text_bert tensor, and writes offsets for the PyTorch explainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def tokenize_with_offsets(tok: Any, text: str) -> Dict[str, np.ndarray]:
    enc = tok(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=50,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_offsets_mapping=True,
    )
    return {
        "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
        "token_type_ids": np.asarray(enc["token_type_ids"], dtype=np.int64),
        "offsets": np.asarray(enc["offset_mapping"], dtype=np.int32),
    }


def audit_split(tok: Any, raw_texts: List[str], text_bert: np.ndarray, label: str) -> Dict[str, np.ndarray]:
    n = len(raw_texts)
    offsets = np.zeros((n, 50, 2), dtype=np.int32)
    valid = np.zeros((n, 50), dtype=bool)
    special = np.zeros((n, 50), dtype=bool)
    match = np.zeros((n,), dtype=bool)
    errors: List[str] = []
    for i, raw in enumerate(raw_texts):
        enc = tokenize_with_offsets(tok, str(raw))
        stored = np.asarray(text_bert[i], dtype=np.int64)
        ok = (
            np.array_equal(stored[0], enc["input_ids"])
            and np.array_equal(stored[1], enc["attention_mask"])
            and np.array_equal(stored[2], enc["token_type_ids"])
        )
        match[i] = bool(ok)
        if not ok:
            errors.append(f"{label}:{i}:tokenizer_mismatch")
        offsets[i] = enc["offsets"]
        valid[i] = enc["attention_mask"] > 0
        special[i] = np.isin(enc["input_ids"], [101, 102])
    return {
        "offsets": offsets,
        "valid": valid,
        "special": special,
        "match": match,
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q2-data", type=Path, required=True)
    ap.add_argument("--attachment4", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    q2 = np.load(args.q2_data, allow_pickle=False)
    a4 = np.load(args.attachment4, allow_pickle=False)
    payload: Dict[str, np.ndarray] = {}
    audit: Dict[str, Any] = {
        "tokenizer_dir": str(args.model_dir),
        "tokenizer_class": tok.__class__.__name__,
        "q2_data": str(args.q2_data),
        "attachment4": str(args.attachment4),
        "splits": {},
        "errors": [],
    }

    for split in ("train", "valid", "test"):
        res = audit_split(
            tok,
            [str(x) for x in q2[f"{split}_raw_text"].tolist()],
            np.asarray(q2[f"{split}_text_bert"]),
            split,
        )
        payload[f"{split}_offsets"] = res["offsets"]
        payload[f"{split}_valid"] = res["valid"]
        payload[f"{split}_special"] = res["special"]
        payload[f"{split}_match"] = res["match"]
        audit["splits"][split] = {
            "n": int(len(res["match"])),
            "tokenizer_match_count": int(res["match"].sum()),
            "tokenizer_match_rate": float(res["match"].mean()) if len(res["match"]) else 0.0,
            "errors": res["errors"],
        }
        audit["errors"].extend(res["errors"])

    a4_res = audit_split(
        tok,
        [str(x) for x in a4["raw_text"].tolist()],
        np.asarray(a4["text_bert"]),
        "attachment4",
    )
    payload["attachment4_offsets"] = a4_res["offsets"]
    payload["attachment4_valid"] = a4_res["valid"]
    payload["attachment4_special"] = a4_res["special"]
    payload["attachment4_match"] = a4_res["match"]
    audit["splits"]["attachment4"] = {
        "n": int(len(a4_res["match"])),
        "tokenizer_match_count": int(a4_res["match"].sum()),
        "tokenizer_match_rate": float(a4_res["match"].mean()) if len(a4_res["match"]) else 0.0,
        "errors": a4_res["errors"],
    }
    audit["errors"].extend(a4_res["errors"])
    audit["status"] = "PASS" if not audit["errors"] else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "out": str(args.out),
        "audit": str(args.audit_json),
        "splits": audit["splits"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
