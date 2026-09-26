#!/usr/bin/env python3
"""Audit and package Q2 data into compact NPZ files.

This script is intentionally dependency-light: it only needs Python, NumPy and
pickle and is meant to run on the workstation that holds the original files.
It does not train or modify any source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def as_str_array(values: Any) -> np.ndarray:
    return np.asarray([str(x) for x in values], dtype=np.str_)


def valid_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.max(np.abs(x), axis=-1) > eps


def run_lengths(mask: np.ndarray) -> list[int]:
    out: list[int] = []
    cur = 0
    for value in mask.astype(bool).tolist():
        if value:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def describe_mask(mask: np.ndarray) -> dict[str, Any]:
    lengths = np.asarray(mask.sum(axis=1), dtype=np.int64)
    all_runs: list[int] = []
    max_runs: list[int] = []
    for row in mask.astype(bool):
        runs = run_lengths(row)
        all_runs.extend(runs)
        max_runs.append(max(runs) if runs else 0)
    return {
        "shape": list(mask.shape),
        "valid_total": int(mask.sum()),
        "valid_fraction": float(mask.mean()),
        "valid_length_min": int(lengths.min()) if len(lengths) else 0,
        "valid_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "valid_length_max": int(lengths.max()) if len(lengths) else 0,
        "empty_samples": int((lengths == 0).sum()),
        "run_count": int(len(all_runs)),
        "run_length_mean": float(np.mean(all_runs)) if all_runs else 0.0,
        "run_length_max": int(max(all_runs)) if all_runs else 0,
        "longest_run_per_sample_mean": float(np.mean(max_runs)) if max_runs else 0.0,
    }


def audit_split(split: str, payload: dict[str, Any], eps: float) -> dict[str, Any]:
    required = [
        "raw_text", "audio", "vision", "id", "text_bert", "classification_labels",
        "regression_labels", "text",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{split}: missing keys {missing}")

    ids = np.asarray([str(x) for x in payload["id"]])
    n = len(ids)
    text_bert = np.asarray(payload["text_bert"])
    audio = np.asarray(payload["audio"])
    vision = np.asarray(payload["vision"])
    text = np.asarray(payload["text"])
    cls = np.asarray(payload["classification_labels"])
    reg = np.asarray(payload["regression_labels"])
    raw_text = np.asarray([str(x) for x in payload["raw_text"]])

    expected = {
        "text_bert": (n, 3, 50),
        "audio": (n, 50, 74),
        "vision": (n, 50, 35),
        "text": (n, 50, 768),
        "classification_labels": (n,),
        "regression_labels": (n,),
        "raw_text": (n,),
    }
    for key, shape in expected.items():
        arr = np.asarray(payload[key])
        if arr.shape != shape:
            raise ValueError(f"{split}.{key}: expected {shape}, got {arr.shape}")

    text_ids = text_bert[:, 0]
    text_att = text_bert[:, 1] > 0
    text_types = text_bert[:, 2]
    audio_valid = valid_rows(audio, eps)
    vision_valid = valid_rows(vision, eps)

    finite = all(np.isfinite(np.asarray(payload[k], dtype=np.float64)).all() for k in ("audio", "vision", "text"))
    video_ids = np.asarray([x.split("$_$", 1)[0] for x in ids])
    id_unique = len(set(ids.tolist())) == n
    video_unique = len(set(video_ids.tolist()))

    first_tokens = text_ids[:, 0]
    last_valid = np.zeros(n, dtype=np.int64)
    for i in range(n):
        positions = np.flatnonzero(text_att[i])
        last_valid[i] = positions[-1] if len(positions) else -1
    last_tokens = np.asarray([text_ids[i, last_valid[i]] if last_valid[i] >= 0 else -1 for i in range(n)])

    return {
        "n": int(n),
        "ids_unique": bool(id_unique),
        "video_count": int(video_unique),
        "video_id_examples": video_ids[:5].tolist(),
        "id_examples": ids[:5].tolist(),
        "raw_text_nonempty": int((np.char.str_len(raw_text.astype(str)) > 0).sum()),
        "raw_text_empty": int((np.char.str_len(raw_text.astype(str)) == 0).sum()),
        "classification_counts": {str(k): int(v) for k, v in sorted(Counter(cls.astype(int).tolist()).items())},
        "classification_min": float(cls.min()),
        "classification_max": float(cls.max()),
        "regression_min": float(reg.min()),
        "regression_max": float(reg.max()),
        "regression_mean": float(reg.mean()),
        "regression_std": float(reg.std()),
        "finite": bool(finite),
        "token_id_min": int(text_ids.min()),
        "token_id_max": int(text_ids.max()),
        "token_type_values": sorted(set(text_types.astype(int).reshape(-1).tolist())),
        "text_first_token_values": sorted(set(first_tokens.astype(int).tolist())),
        "text_last_valid_token_values": sorted(set(last_tokens.astype(int).tolist())),
        "text_attention": describe_mask(text_att),
        "audio_valid": describe_mask(audio_valid),
        "vision_valid": describe_mask(vision_valid),
        "text_valid": describe_mask(text_att),
        "audio_vision_equal_mask_rate": float((audio_valid == vision_valid).mean()),
        "audio_zero_but_attention_valid": int((~audio_valid & text_att).sum()),
        "vision_zero_but_attention_valid": int((~vision_valid & text_att).sum()),
        "text_zero_feature_rows": int((~valid_rows(text, eps)).sum()),
        "split_id_overlap": {},
    }


def load_attachment3(root: Path, eps: float) -> dict[str, Any]:
    files = sorted(root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no attachment-3 pkl files under {root}")
    names: list[str] = []
    text_bert: list[np.ndarray] = []
    audio: list[np.ndarray] = []
    vision: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for path in files:
        payload = pickle.load(path.open("rb"))
        item = payload["test"]
        tb = np.asarray(item["text_bert"], dtype=np.float32)
        au = np.asarray(item["audio"], dtype=np.float32)
        vi = np.asarray(item["vision"], dtype=np.float32)
        if tb.shape != (1, 3, 50) or au.shape != (1, 50, 74) or vi.shape != (1, 50, 35):
            raise ValueError(f"unexpected shapes in {path.name}: {tb.shape}, {au.shape}, {vi.shape}")
        if not all(np.isfinite(x).all() for x in (tb, au, vi)):
            raise ValueError(f"non-finite values in {path.name}")
        ta = tb[0, 1] > 0
        av = valid_rows(au[0], eps)
        vv = valid_rows(vi[0], eps)
        names.append(path.name)
        text_bert.append(tb[0])
        audio.append(au[0])
        vision.append(vi[0])
        audits.append({
            "file": path.name,
            "text_attention": ta.astype(np.uint8).tolist(),
            "audio_valid": av.astype(np.uint8).tolist(),
            "vision_valid": vv.astype(np.uint8).tolist(),
            "text_valid_length": int(ta.sum()),
            "audio_valid_length": int(av.sum()),
            "vision_valid_length": int(vv.sum()),
            "token_id_min": int(tb[0, 0].min()),
            "token_id_max": int(tb[0, 0].max()),
        })
    tb_arr = np.stack(text_bert).astype(np.int16)
    au_arr = np.stack(audio).astype(np.float16)
    vi_arr = np.stack(vision).astype(np.float16)
    return {
        "filenames": np.asarray(names, dtype=np.str_),
        "text_bert": tb_arr,
        "audio": au_arr,
        "vision": vi_arr,
        "audit": {
            "count": len(files),
            "text_attention": describe_mask(tb_arr[:, 1] > 0),
            "audio_valid": describe_mask(valid_rows(au_arr, eps)),
            "vision_valid": describe_mask(valid_rows(vi_arr, eps)),
            "audio_vision_equal_mask_rate": float((valid_rows(au_arr, eps) == valid_rows(vi_arr, eps)).mean()),
            "per_file": audits,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--attachment3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"loading {args.aligned}", flush=True)
    with args.aligned.open("rb") as f:
        data = pickle.load(f)

    audit: dict[str, Any] = {
        "source_aligned": str(args.aligned.resolve()),
        "source_aligned_sha256": sha256_file(args.aligned),
        "source_attachment3_dir": str(args.attachment3_dir.resolve()),
        "eps": args.eps,
        "splits": {},
        "attachment3": {},
    }
    for split in ("train", "valid", "test"):
        print(f"auditing {split}", flush=True)
        audit["splits"][split] = audit_split(split, data[split], args.eps)

    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        ia = set(map(str, data[a]["id"]))
        ib = set(map(str, data[b]["id"]))
        va = {x.split("$_$", 1)[0] for x in ia}
        vb = {x.split("$_$", 1)[0] for x in ib}
        audit["splits"][a]["split_id_overlap"][b] = len(ia & ib)
        audit["splits"][a].setdefault("split_video_overlap", {})[b] = len(va & vb)

    print("packing attachment 3", flush=True)
    a3 = load_attachment3(args.attachment3_dir, args.eps)
    audit["attachment3"] = a3["audit"]

    print("saving compact arrays", flush=True)
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        payload = data[split]
        arrays[f"{split}_text_bert"] = np.asarray(payload["text_bert"], dtype=np.int16)
        arrays[f"{split}_audio"] = np.asarray(payload["audio"], dtype=np.float16)
        arrays[f"{split}_vision"] = np.asarray(payload["vision"], dtype=np.float16)
        arrays[f"{split}_text_feature"] = np.asarray(payload["text"], dtype=np.float16)
        arrays[f"{split}_labels"] = np.asarray(payload["classification_labels"], dtype=np.int8)
        arrays[f"{split}_regression"] = np.asarray(payload["regression_labels"], dtype=np.float16)
        arrays[f"{split}_ids"] = as_str_array(payload["id"])
        arrays[f"{split}_raw_text"] = as_str_array(payload["raw_text"])

    out_npz = args.output_dir / "q2_data.npz"
    np.savez_compressed(out_npz, **arrays)
    a3_npz = args.output_dir / "attachment3_aligned.npz"
    np.savez_compressed(
        a3_npz,
        filenames=a3["filenames"],
        text_bert=a3["text_bert"],
        audio=a3["audio"],
        vision=a3["vision"],
    )

    audit["outputs"] = {
        out_npz.name: {"bytes": out_npz.stat().st_size, "sha256": sha256_file(out_npz)},
        a3_npz.name: {"bytes": a3_npz.stat().st_size, "sha256": sha256_file(a3_npz)},
    }
    audit["elapsed_seconds"] = time.time() - started
    audit_path = args.output_dir / "q2_data_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "audit": str(audit_path), "elapsed": audit["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

