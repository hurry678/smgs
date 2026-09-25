#!/usr/bin/env python3
"""Audit and convert attachment-4 aligned samples into a reproducible NPZ.

This script is intentionally run with the server base environment because the
provided attachment-4 pickles were written with NumPy 2.x.  It does not train
or tune anything.  It records exact text-token offsets and clearly labels
video-time/frame mappings as approximate when no original timestamps exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pickle


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    def as_float(x: Any) -> float | None:
        try:
            return float(x)
        except Exception:
            return None

    fps = None
    rate = stream.get("avg_frame_rate")
    if rate and "/" in str(rate):
        a, b = str(rate).split("/", 1)
        try:
            fps = float(a) / float(b)
        except Exception:
            fps = None
    return {
        "duration": as_float(fmt.get("duration")) or as_float(stream.get("duration")),
        "fps": fps,
        "nb_frames": int(stream["nb_frames"])
        if str(stream.get("nb_frames", "")).isdigit()
        else None,
        "ffprobe_raw": data,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)

    aligned_dir = args.raw_dir / "对齐版本"
    unaligned_dir = args.raw_dir / "未对齐版本"
    pkls = sorted(aligned_dir.glob("*.pkl"))
    if len(pkls) != 20:
        raise SystemExit(f"expected 20 aligned pickles, found {len(pkls)}")

    text_bert: List[np.ndarray] = []
    text_feature: List[np.ndarray] = []
    audio: List[np.ndarray] = []
    vision: List[np.ndarray] = []
    ids: List[str] = []
    filenames: List[str] = []
    raw_texts: List[str] = []
    audio_lengths: List[int] = []
    vision_lengths: List[int] = []
    metadata_samples: List[Dict[str, Any]] = []
    errors: List[str] = []

    for p in pkls:
        with p.open("rb") as f:
            d = pickle.load(f)
        if not isinstance(d, dict):
            raise SystemExit(f"{p} is not a dict")
        required = {"id", "raw_text", "text", "text_bert", "audio", "vision"}
        missing = sorted(required - set(d))
        if missing:
            raise SystemExit(f"{p} missing keys: {missing}")
        sid = str(d["id"])
        if sid != p.stem:
            errors.append(f"id/file mismatch: {sid} != {p.stem}")
        raw = str(d["raw_text"])
        tb = np.asarray(d["text_bert"], dtype=np.int64)
        tf = np.asarray(d["text"], dtype=np.float32)
        au = np.asarray(d["audio"], dtype=np.float32)
        vi = np.asarray(d["vision"], dtype=np.float32)
        if tb.shape != (3, 50):
            raise SystemExit(f"{p}: text_bert shape {tb.shape}")
        if tf.shape != (50, 768):
            raise SystemExit(f"{p}: text shape {tf.shape}")
        if au.shape != (50, 74) or vi.shape != (50, 35):
            raise SystemExit(f"{p}: audio/vision shape {au.shape}/{vi.shape}")
        if (
            not np.isfinite(tf).all()
            or not np.isfinite(au).all()
            or not np.isfinite(vi).all()
        ):
            errors.append(f"{sid}: non-finite feature value")

        enc = tok(
            raw,
            add_special_tokens=True,
            truncation=True,
            max_length=50,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
        )
        expected_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        expected_att = np.asarray(enc["attention_mask"], dtype=np.int64)
        expected_typ = np.asarray(enc["token_type_ids"], dtype=np.int64)
        offsets = np.asarray(enc["offset_mapping"], dtype=np.int64)
        if not np.array_equal(tb[0], expected_ids):
            errors.append(f"{sid}: input_ids tokenizer mismatch")
        if not np.array_equal(tb[1], expected_att):
            errors.append(f"{sid}: attention_mask tokenizer mismatch")
        if not np.array_equal(tb[2], expected_typ):
            errors.append(f"{sid}: token_type_ids tokenizer mismatch")

        semantic = (tb[1] > 0) & (tb[0] != 101) & (tb[0] != 102)
        token_offsets: List[List[int] | None] = []
        for i in range(50):
            if semantic[i] and (offsets[i, 1] > offsets[i, 0]):
                token_offsets.append([int(offsets[i, 0]), int(offsets[i, 1])])
            else:
                token_offsets.append(None)

        video_path = aligned_dir / "videos" / f"{sid}.mp4"
        if not video_path.exists():
            errors.append(f"{sid}: missing video {video_path}")
            probe = {}
        else:
            probe = probe_video(video_path)
        duration = probe.get("duration")
        fps = probe.get("fps")
        if duration is None or duration <= 0:
            errors.append(f"{sid}: invalid video duration")
        if fps is None or fps <= 0:
            errors.append(f"{sid}: invalid fps")

        # Approximate position-to-time map.  The provided aligned features are
        # token-position aligned, not timestamp aligned.  We use the exact
        # tokenizer character span centre as the ordering coordinate and scale
        # it to the measured video duration.  This is explicitly approximate.
        position_to_time: List[Dict[str, Any] | None] = []
        for i, off in enumerate(token_offsets):
            if off is None or duration is None:
                position_to_time.append(None)
                continue
            a, b = off
            center = 0.5 * (a + b)
            t = float(duration) * (center / max(1, len(raw)))
            position_to_time.append(
                {
                    "token_position": i,
                    "char_start": int(a),
                    "char_end": int(b),
                    "time_center": t,
                    "time_start_est": max(
                        0.0,
                        t - 0.5 * (float(duration) / max(1, len(raw))) * max(1, b - a),
                    ),
                    "time_end_est": min(
                        float(duration),
                        t + 0.5 * (float(duration) / max(1, len(raw))) * max(1, b - a),
                    ),
                    "frame_center_est": int(round(t * float(fps))) if fps else None,
                    "status": "approximate_proportional_token_to_video",
                }
            )

        text_bert.append(tb.astype(np.int16))
        text_feature.append(tf.astype(np.float16))
        audio.append(au.astype(np.float16))
        vision.append(vi.astype(np.float16))
        ids.append(sid)
        filenames.append(p.name)
        raw_texts.append(raw)
        audio_lengths.append(int(d.get("audio_lengths", 0)))
        vision_lengths.append(int(d.get("vision_lengths", 0)))

        # Cross-check the unaligned companion and record its valid lengths.
        up = unaligned_dir / p.name
        u = pickle.load(up.open("rb")) if up.exists() else {}
        audio_len = int(u.get("audio_lengths", d.get("audio_lengths", 0)) or 0)
        vision_len = int(u.get("vision_lengths", d.get("vision_lengths", 0)) or 0)

        metadata_samples.append(
            {
                "sample_id": sid,
                "feature_file": str(p.relative_to(args.raw_dir)),
                "feature_sha256": sha256_file(p),
                "video_file": str(video_path.relative_to(args.raw_dir))
                if video_path.exists()
                else None,
                "video_sha256": sha256_file(video_path)
                if video_path.exists()
                else None,
                "raw_text": raw,
                "raw_char_length": len(raw),
                "text_tokenizer": tok.__class__.__name__,
                "text_bert_shape": list(tb.shape),
                "semantic_token_positions": [int(i) for i in np.flatnonzero(semantic)],
                "token_offsets": token_offsets,
                "audio_valid_positions": [
                    int(i) for i in np.flatnonzero(np.any(np.abs(au) > 1e-8, axis=1))
                ],
                "vision_valid_positions": [
                    int(i) for i in np.flatnonzero(np.any(np.abs(vi) > 1e-8, axis=1))
                ],
                "audio_lengths_unaligned": audio_len,
                "vision_lengths_unaligned": vision_len,
                "video_duration_seconds": duration,
                "video_fps": fps,
                "video_nb_frames": probe.get("nb_frames"),
                "position_to_time": position_to_time,
                "mapping_precision": {
                    "text": "exact_character_offsets_from_official_tokenizer",
                    "audio": "approximate_proportional_token_position_to_measured_video_time",
                    "vision": "approximate_proportional_token_position_to_measured_video_time_and_frame",
                    "warning": "No original token-level timestamps were provided with attachment 4; audio/vision locations are estimates, not claimed exact ground truth.",
                },
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "attachment4_aligned.npz",
        text_bert=np.stack(text_bert).astype(np.int16),
        text_feature=np.stack(text_feature).astype(np.float16),
        audio=np.stack(audio).astype(np.float16),
        vision=np.stack(vision).astype(np.float16),
        ids=np.asarray(ids, dtype=str),
        filenames=np.asarray(filenames, dtype=str),
        raw_text=np.asarray(raw_texts, dtype=str),
        audio_lengths=np.asarray(audio_lengths, dtype=np.int32),
        vision_lengths=np.asarray(vision_lengths, dtype=np.int32),
    )
    audit = {
        "question": 3,
        "attachment": 4,
        "n": len(ids),
        "source_dir": "external://" + args.raw_dir.name,
        "feature_version": "attachment4_aligned_token50_v1",
        "tokenizer_dir": "external://" + args.model_dir.name,
        "tokenizer_class": tok.__class__.__name__,
        "npz": "run://attachment4_data/attachment4_aligned.npz",
        "npz_sha256": sha256_file(args.out_dir / "attachment4_aligned.npz"),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "mapping_policy": {
            "text": "exact character offsets from official tokenizer",
            "audio": "approximate proportional token-position-to-time; no original timestamps",
            "vision": "approximate proportional token-position-to-time/frame; no original timestamps",
        },
        "samples": metadata_samples,
    }
    (args.out_dir / "attachment4_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "n": len(ids),
                "errors": errors,
                "npz": str(args.out_dir / "attachment4_aligned.npz"),
                "audit": str(args.out_dir / "attachment4_audit.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
