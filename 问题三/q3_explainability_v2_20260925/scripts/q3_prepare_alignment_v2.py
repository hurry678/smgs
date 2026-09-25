#!/usr/bin/env python3
"""Create auditable MFA-primary token-to-time mappings for attachment 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        [str(part) for part in command],
        env=env,
        text=True,
        capture_output=True,
    )
    record = {
        "command": [str(part) for part in command],
        "returncode": int(result.returncode),
        "elapsed_seconds": float(time.perf_counter() - started),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if check and result.returncode:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return record


def replace_paths(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        result = value
        for source, target in replacements:
            result = result.replace(source, target)
        return result
    if isinstance(value, list):
        return [replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, replacements) for key, item in value.items()}
    return value


def probe_video(path: Path) -> dict[str, Any]:
    command = [
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
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    file_format = payload.get("format") or {}

    def optional_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    duration = optional_float(file_format.get("duration")) or optional_float(
        stream.get("duration")
    )
    fps = None
    rate = str(stream.get("avg_frame_rate") or "")
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        if float(denominator):
            fps = float(numerator) / float(denominator)
    return {
        "duration": duration,
        "fps": fps,
        "nb_frames": (
            int(stream["nb_frames"])
            if str(stream.get("nb_frames", "")).isdigit()
            else None
        ),
    }


def normalized(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def map_mfa_entries_to_words(
    text: str,
    entries: Sequence[Sequence[Any]],
    duration: float,
) -> list[dict[str, Any]]:
    """Map MFA labels back to disjoint spans in the original transcript."""
    aligned = [
        (float(entry[0]), float(entry[1]), str(entry[2]))
        for entry in entries
        if len(entry) >= 3 and str(entry[2]).strip() and str(entry[2]) != "<eps>"
    ]
    normalized_chars = [
        (index, character.lower())
        for index, character in enumerate(text)
        if character.isalnum()
    ]
    normalized_text = "".join(character for _, character in normalized_chars)
    cursor = 0
    raw_cursor = 0
    words: list[dict[str, Any]] = []
    for word_index, (start, end, label) in enumerate(aligned):
        target = normalized(label)
        if label == "[bracketed]":
            match = re.search(r"\[[^\]]+\]", text[raw_cursor:])
            if match is None:
                raise ValueError("MFA emitted [bracketed] without matching source text")
            char_start = raw_cursor + match.start()
            char_end = raw_cursor + match.end()
            cursor += len(normalized(text[char_start:char_end]))
        elif label == "<unk>":
            if cursor >= len(normalized_chars):
                raise ValueError("MFA emitted unmatched <unk>")
            char_start = normalized_chars[cursor][0]
            token = re.match(r"\S+", text[char_start:])
            if token is None:
                raise ValueError("cannot map MFA <unk>")
            char_end = char_start + len(token.group())
            cursor += len(normalized(text[char_start:char_end]))
        else:
            found = normalized_text.find(target, cursor)
            if not target or found < cursor:
                raise ValueError(f"cannot map MFA token {word_index}: {label!r}")
            if normalized_text[cursor:found]:
                raise ValueError(f"MFA skipped source text before {label!r}")
            char_start = normalized_chars[found][0]
            char_end = normalized_chars[found + len(target) - 1][0] + 1
            cursor = found + len(target)
        raw_cursor = char_end
        if not (0.0 <= start < end <= duration + 0.05):
            raise ValueError(
                f"invalid MFA timing for {label!r}: {start}, {end}, duration={duration}"
            )
        words.append(
            {
                "word_index": int(word_index),
                "label": label,
                "char_start": int(char_start),
                "char_end": int(char_end),
                "time_start_seconds": max(0.0, start),
                "time_end_seconds": min(float(duration), end),
            }
        )
    if not words:
        raise ValueError("MFA output contains no aligned words")
    if cursor != len(normalized_chars):
        raise ValueError(
            f"MFA did not cover normalized transcript: {cursor}/{len(normalized_chars)}"
        )
    words[0]["char_start"] = 0
    for current, following in zip(words, words[1:]):
        current["char_end"] = int(following["char_start"])
    words[-1]["char_end"] = len(text)
    for word in words:
        word["text"] = text[word["char_start"] : word["char_end"]]
    return words


def token_time_rows(
    offsets: np.ndarray,
    valid: np.ndarray,
    words: Sequence[dict[str, Any]],
    duration: float,
    fps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token_position, ((char_start, char_end), is_valid) in enumerate(
        zip(offsets.tolist(), valid.tolist())
    ):
        if not is_valid or int(char_end) <= int(char_start):
            continue
        overlapping = [
            word
            for word in words
            if int(word["char_end"]) > int(char_start)
            and int(word["char_start"]) < int(char_end)
        ]
        if not overlapping:
            raise ValueError(
                f"token {token_position} span {(char_start, char_end)} has no MFA word"
            )
        start = float(overlapping[0]["time_start_seconds"])
        end = float(overlapping[-1]["time_end_seconds"])
        if not (0.0 <= start < end <= duration + 0.05):
            raise ValueError(
                f"token {token_position} has invalid mapped timing {start}, {end}"
            )
        rows.append(
            {
                "token_position": int(token_position),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "time_start_seconds": start,
                "time_end_seconds": end,
                "frame_start": int(max(0, np.floor(start * fps))),
                "frame_end": int(max(0, np.ceil(end * fps))),
                "status": "mfa_forced_alignment",
                "confidence": 1.0,
                "confidence_definition": "exact_character_overlap_with_mfa_word_span",
            }
        )
    return rows


def proportional_rows(
    offsets: np.ndarray,
    valid: np.ndarray,
    text_length: int,
    duration: float,
    fps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    denominator = max(1, int(text_length))
    for token_position, ((char_start, char_end), is_valid) in enumerate(
        zip(offsets.tolist(), valid.tolist())
    ):
        if not is_valid or int(char_end) <= int(char_start):
            continue
        start = float(duration) * int(char_start) / denominator
        end = float(duration) * int(char_end) / denominator
        rows.append(
            {
                "token_position": int(token_position),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "time_start_est": start,
                "time_end_est": end,
                "frame_center_est": int(round(0.5 * (start + end) * fps)),
                "status": "approximate_proportional_token_to_video",
                "confidence": 0.0,
                "confidence_definition": "fallback_not_an_alignment_confidence",
            }
        )
    return rows


def locate_mfa_output(output_dir: Path, stem: str) -> Path | None:
    matches = sorted(output_dir.rglob(f"{stem}.json"))
    if len(matches) > 1:
        raise ValueError(f"multiple MFA outputs found for {stem}: {matches}")
    return matches[0] if matches else None


def write_corpus(
    ids: Sequence[str],
    texts: Sequence[str],
    videos_dir: Path,
    corpus_dir: Path,
    audio_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    layout: dict[str, str] = {}
    executions: list[dict[str, Any]] = []
    speaker_dir = corpus_dir / "speaker_000"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample_id, text) in enumerate(zip(ids, texts)):
        stem = f"sample_{index:03d}"
        layout[str(sample_id)] = stem
        video_path = videos_dir / f"{sample_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"missing attachment-4 video: {video_path}")
        audio_path = audio_dir / f"{stem}.wav"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
        execution = run_command(command)
        execution["stage"] = "extract_audio"
        execution["sample_id"] = str(sample_id)
        executions.append(execution)
        if execution["returncode"] or not audio_path.is_file():
            raise RuntimeError(f"ffmpeg extraction failed for {sample_id}")
        (speaker_dir / f"{stem}.wav").symlink_to(audio_path.resolve())
        (speaker_dir / f"{stem}.lab").write_text(str(text) + "\n", encoding="utf-8")
    return layout, executions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--attachment4", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mfa-executable", type=Path, required=True)
    parser.add_argument("--mfa-root-dir", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--acoustic-model", type=Path, required=True)
    parser.add_argument("--g2p-model", type=Path, required=True)
    parser.add_argument("--expected-mfa-version", default="3.4.1")
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--min-mfa-sample-rate", type=float, default=0.8)
    parser.add_argument("--allow-low-coverage", action="store_true")
    args = parser.parse_args()

    required = [
        args.attachment4,
        args.offsets,
        args.base_audit,
        args.mfa_executable,
        args.dictionary,
        args.acoustic_model,
        args.g2p_model,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    if not 0.0 <= args.min_mfa_sample_rate <= 1.0:
        raise SystemExit("--min-mfa-sample-rate must be in [0,1]")

    with np.load(args.attachment4, allow_pickle=False) as attachment:
        ids = [str(value) for value in attachment["ids"].tolist()]
        texts = [str(value) for value in attachment["raw_text"].tolist()]
    with np.load(args.offsets, allow_pickle=False) as offset_pack:
        offsets = np.asarray(offset_pack["attachment4_offsets"], dtype=np.int32)
        valid = np.asarray(offset_pack["attachment4_valid"], dtype=bool) & ~np.asarray(
            offset_pack["attachment4_special"], dtype=bool
        )
        tokenizer_match = np.asarray(offset_pack["attachment4_match"], dtype=bool)
    if offsets.shape != (len(ids), 50, 2) or valid.shape != (len(ids), 50):
        raise SystemExit("attachment-4 offset shapes are invalid")
    if not bool(tokenizer_match.all()):
        raise SystemExit("attachment-4 tokenizer offsets do not match stored tensors")

    base_audit = json.loads(args.base_audit.read_text(encoding="utf-8"))
    base_samples = {
        str(sample["sample_id"]): sample for sample in base_audit.get("samples", [])
    }
    if set(base_samples) != set(ids):
        raise SystemExit("base audit sample IDs do not match attachment-4 NPZ")

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    corpus_dir = args.work_dir / "corpus"
    output_dir = args.work_dir / "output"
    audio_dir = args.work_dir / "audio"
    output_dir.mkdir(parents=True)
    videos_dir = args.raw_dir / "\u5bf9\u9f50\u7248\u672c" / "videos"
    layout, executions = write_corpus(ids, texts, videos_dir, corpus_dir, audio_dir)

    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(args.mfa_root_dir)
    env["PATH"] = str(args.mfa_executable.parent) + os.pathsep + env.get("PATH", "")
    version_result = subprocess.run(
        [str(args.mfa_executable), "version"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    version = version_result.stdout.strip()
    if version != args.expected_mfa_version:
        raise SystemExit(
            f"MFA version mismatch: expected {args.expected_mfa_version}, got {version}"
        )
    command = [
        str(args.mfa_executable),
        "align",
        str(corpus_dir),
        str(args.dictionary),
        str(args.acoustic_model),
        str(output_dir),
        "--output_format",
        "json",
        "--include_original_text",
        "--clean",
        "--overwrite",
        "--num_jobs",
        str(args.num_jobs),
        "--g2p_model_path",
        str(args.g2p_model),
        "--single_speaker",
    ]
    batch = run_command(command, env=env)
    batch["stage"] = "mfa_batch_align"
    executions.append(batch)

    output_samples: list[dict[str, Any]] = []
    aligned_count = 0
    fallback_ids: list[str] = []
    for index, (sample_id, text) in enumerate(zip(ids, texts)):
        source = dict(base_samples[sample_id])
        video_path = videos_dir / f"{sample_id}.mp4"
        probe = probe_video(video_path)
        duration = float(probe["duration"] or 0.0)
        fps = float(probe["fps"] or 0.0)
        if duration <= 0.0 or fps <= 0.0:
            raise SystemExit(f"invalid video metadata for {sample_id}: {probe}")
        mfa_path = locate_mfa_output(output_dir, layout[sample_id])
        alignment_error = None
        words: list[dict[str, Any]] = []
        try:
            if mfa_path is None:
                raise FileNotFoundError("MFA JSON output not found")
            mfa_payload = json.loads(mfa_path.read_text(encoding="utf-8"))
            entries = mfa_payload["tiers"]["words"]["entries"]
            words = map_mfa_entries_to_words(text, entries, duration)
            position_to_time = token_time_rows(
                offsets[index], valid[index], words, duration, fps
            )
            route = "mfa_forced_alignment"
            aligned_count += 1
        except Exception as exc:
            alignment_error = f"{type(exc).__name__}: {exc}"
            position_to_time = proportional_rows(
                offsets[index], valid[index], len(text), duration, fps
            )
            route = "approximate_proportional_fallback"
            fallback_ids.append(sample_id)
        source.update(
            {
                "video_duration_seconds": duration,
                "video_fps": fps,
                "video_nb_frames": probe["nb_frames"],
                "alignment_route": route,
                "alignment_error": alignment_error,
                "mfa_output": (
                    str(mfa_path.relative_to(args.work_dir))
                    if mfa_path is not None
                    else None
                ),
                "mfa_output_sha256": (
                    sha256_file(mfa_path) if mfa_path is not None else None
                ),
                "mfa_words": words,
                "position_to_time": position_to_time,
                "mapping_precision": {
                    "text": "exact_character_offsets_from_official_tokenizer",
                    "audio": (
                        "mfa_word_timing_mapped_by_character_overlap"
                        if route == "mfa_forced_alignment"
                        else "approximate_proportional_fallback"
                    ),
                    "vision": (
                        "mfa_word_timing_then_video_fps"
                        if route == "mfa_forced_alignment"
                        else "approximate_proportional_fallback_then_video_fps"
                    ),
                },
            }
        )
        output_samples.append(source)

    coverage = aligned_count / len(ids) if ids else 0.0
    coverage_ok = coverage >= args.min_mfa_sample_rate
    status = "PASS" if coverage_ok or args.allow_low_coverage else "FAIL"
    q1_resource_root = Path(
        os.path.commonpath(
            [
                str(args.mfa_executable.resolve()),
                str(args.dictionary.resolve()),
                str(args.acoustic_model.resolve()),
                str(args.g2p_model.resolve()),
            ]
        )
    )
    replacements = [
        (str(args.work_dir.resolve()), "run://mfa_work"),
        (str(args.raw_dir.resolve()), "attachment4://"),
        (str(q1_resource_root), "q1-mfa://"),
    ]
    audit = {
        "schema": "q3v2-mfa-alignment-audit-v1",
        "status": status,
        "policy": "mfa_3.4.1_primary_with_explicit_proportional_fallback",
        "fallback_policy": (
            "Per-sample proportional fallback is descriptive only; no "
            "energy/pause dynamic-programming fallback is implemented."
        ),
        "mfa": {
            "version": version,
            "executable_sha256": sha256_file(args.mfa_executable),
            "dictionary_sha256": sha256_file(args.dictionary),
            "acoustic_model_sha256": sha256_file(args.acoustic_model),
            "g2p_model_sha256": sha256_file(args.g2p_model),
            "num_jobs": int(args.num_jobs),
            "batch_returncode": int(batch["returncode"]),
        },
        "inputs": {
            "attachment4_sha256": sha256_file(args.attachment4),
            "offsets_sha256": sha256_file(args.offsets),
            "base_audit_sha256": sha256_file(args.base_audit),
        },
        "n_samples": len(ids),
        "mfa_aligned_samples": aligned_count,
        "fallback_samples": len(fallback_ids),
        "fallback_sample_ids": fallback_ids,
        "mfa_sample_rate": coverage,
        "minimum_mfa_sample_rate": float(args.min_mfa_sample_rate),
        "low_coverage_override": bool(args.allow_low_coverage),
        "executions": replace_paths(executions, replacements),
        "samples": output_samples,
    }
    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "mfa_aligned_samples": aligned_count,
                "fallback_samples": len(fallback_ids),
                "mfa_sample_rate": coverage,
                "audit": str(args.out_audit),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if status != "PASS":
        raise SystemExit(
            "MFA coverage gate failed; inspect audit before using attachment 4"
        )


if __name__ == "__main__":
    main()
