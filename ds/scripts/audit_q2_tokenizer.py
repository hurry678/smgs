#!/usr/bin/env python3
"""Full-tokenizer consistency audit for Q2 using the server base environment."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.data, allow_pickle=False)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    report = {'model_dir': str(args.model_dir), 'tokenizer_class': tok.__class__.__name__, 'model_max_length': int(tok.model_max_length), 'splits': {}}
    for split in ('train','valid','test'):
        raw = z[f'{split}_raw_text'].astype(str).tolist()
        stored = z[f'{split}_text_bert'].astype(np.int64)
        enc = tok(raw, add_special_tokens=True, truncation=True, max_length=50, padding='max_length', return_attention_mask=True, return_token_type_ids=True)
        ids = np.asarray(enc['input_ids'], dtype=np.int64)
        att = np.asarray(enc['attention_mask'], dtype=np.int64)
        typ = np.asarray(enc['token_type_ids'], dtype=np.int64)
        report['splits'][split] = {
            'n': len(raw),
            'ids_exact_rate': float((ids == stored[:,0]).mean()),
            'attention_exact_rate': float((att == stored[:,1]).mean()),
            'token_type_exact_rate': float((typ == stored[:,2]).mean()),
            'ids_mismatch_samples': int((ids != stored[:,0]).any(axis=1).sum()),
            'attention_mismatch_samples': int((att != stored[:,1]).any(axis=1).sum()),
            'max_abs_id_diff': int(np.abs(ids - stored[:,0]).max()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
