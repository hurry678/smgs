"""Deterministic MFA-primary alignment with stable-ts and VAD evidence."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from prepare import ROOT, dump, sha256


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mfa_environment(config):
    executable = ROOT / config["mfa_executable"]
    if not executable.exists():
        raise FileNotFoundError(
            f"MFA executable not found: {executable}. See README.md for installation.")
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(ROOT / config["mfa_root_dir"])
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    return executable, env


def _safe_layout(rows):
    speakers = {name: f"speaker_{i:03d}" for i, name in enumerate(
        sorted({row["video_id"] for row in rows}))}
    return {
        row["sample_id"]: {
            "speaker": speakers[row["video_id"]],
            "stem": f"sample_{i:03d}",
        }
        for i, row in enumerate(sorted(rows, key=lambda item: item["sample_id"]))
    }


def _portable(value):
    return str(value).replace(str(ROOT), ".").replace(str(Path.home()), "<home>")


def _execute(command, env, stage, sample_id=None):
    started = time.perf_counter()
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + "\n" + result.stderr)
    record = {
        "stage": stage, "sample_id": sample_id,
        "command": [_portable(part) for part in command],
        "returncode": result.returncode, "seconds": time.perf_counter() - started,
        "output_excerpt": _portable(plain[-5000:]),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    }
    return result, record


def ensure_mfa_alignments(rows, config, force=False):
    """Build a safe-name corpus and run one reproducible MFA batch."""
    c = config["automatic_alignment"]
    base = ROOT / c["work_dir"]
    corpus, output = base / "corpus", base / "output"
    mapping_path, report_path = base / "mapping.json", base / "run.json"
    usable = [row for row in rows if np.any(
        __import__("soundfile").read(
            ROOT / "intermediate/audio" / (row["sample_id"] + ".wav"), dtype="float32")[0] != 0)]
    layout = _safe_layout(usable)
    acoustic = ROOT / c["mfa_acoustic_model"]
    dictionary = ROOT / c["mfa_dictionary"]
    g2p = ROOT / c["mfa_g2p_model"]
    fingerprint = _digest({
        "policy": c["policy"], "mfa_version": c["mfa_version"],
        "mfa_num_jobs": c["mfa_num_jobs"],
        "mfa_single_speaker_mode": c["mfa_single_speaker_mode"],
        "acoustic_sha256": sha256(acoustic), "dictionary_sha256": sha256(dictionary),
        "g2p_sha256": sha256(g2p),
        "runner_sha256": sha256(ROOT / "src/automatic_alignment.py"),
        "samples": [{
            "sample_id": row["sample_id"], "text": row["text"],
            "audio_sha256": sha256(ROOT / "intermediate/audio" / (row["sample_id"] + ".wav")),
            **layout[row["sample_id"]],
        } for row in usable],
    })
    expected = {
        sid: output / item["speaker"] / (item["stem"] + ".json")
        for sid, item in layout.items()
    }
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    if (not force and previous.get("fingerprint") == fingerprint
            and previous.get("execution_schema") == "mfa-execution-2"
            and mapping_path.exists()
            and json.loads(mapping_path.read_text()) == layout
            and all((path.exists() and sha256(path) == previous.get("output_sha256", {}).get(sid))
                    or (not path.exists() and sid in previous.get("mfa_failed_samples", []))
                    for sid, path in expected.items())):
        return previous
    run_begin = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_id = uuid.uuid4().hex
    for path in (corpus, output):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    for row in usable:
        item = layout[row["sample_id"]]
        speaker = corpus / item["speaker"]
        speaker.mkdir(parents=True, exist_ok=True)
        source = ROOT / "intermediate/audio" / (row["sample_id"] + ".wav")
        (speaker / (item["stem"] + ".wav")).symlink_to(source.resolve())
        (speaker / (item["stem"] + ".lab")).write_text(row["text"] + "\n", encoding="utf-8")
    dump(mapping_path, layout)
    executable, env = _mfa_environment(c)
    version = subprocess.check_output([str(executable), "version"], env=env, text=True).strip()
    if version != c["mfa_version"]:
        raise ValueError(f"MFA version mismatch: expected {c['mfa_version']}, got {version}")
    command = [
        str(executable), "align", str(corpus), str(dictionary), str(acoustic), str(output),
        "--output_format", "json", "--include_original_text", "--clean", "--overwrite",
        "--num_jobs", str(c["mfa_num_jobs"]), "--g2p_model_path", str(g2p),
    ]
    if c["mfa_single_speaker_mode"]:
        command.append("--single_speaker")
    completed, batch_execution = _execute(command, env, "align")
    executions = [batch_execution]
    (base / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (base / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    # MFA's corpus exporter can fail for one speaker group even when individual
    # utterances aligned correctly. Re-run only missing items without adaptation.
    batch_output_count = sum(path.exists() for path in expected.values())
    batch_samples = {sid for sid, path in expected.items() if path.exists()}
    if completed.returncode and not batch_output_count:
        raise RuntimeError(f"MFA failed with exit code {completed.returncode}; see {base}")
    individual_failures = {}
    for sid, path in expected.items():
        if path.exists():
            continue
        item = layout[sid]
        path.parent.mkdir(parents=True, exist_ok=True)
        single_command = [
            str(executable), "align_one",
            str(corpus / item["speaker"] / (item["stem"] + ".wav")),
            str(corpus / item["speaker"] / (item["stem"] + ".lab")),
            str(dictionary), str(acoustic), str(path),
            "--output_format", "json", "--clean", "--overwrite",
            "--g2p_model_path", str(g2p),
        ]
        result, execution = _execute(single_command, env, "align_one", sid)
        execution["output_created"] = path.exists()
        executions.append(execution)
        (base / f"{item['stem']}_align_one.log").write_text(
            result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode or not path.exists():
            # A failed process must not leave an apparently usable partial output.
            path.unlink(missing_ok=True)
            individual_failures[sid] = {
                "returncode": result.returncode,
                "log": str((base / f"{item['stem']}_align_one.log").relative_to(ROOT)),
            }
    report = {
        "execution_schema": "mfa-execution-2", "run_id": run_id,
        "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds": time.perf_counter() - run_begin,
        "executions": executions,
        "samples": [{
            "sample_id": sid,
            "result": "failed" if sid in individual_failures else "aligned",
            "route": "batch" if sid in batch_samples else "align_one",
            "output": str(path.relative_to(ROOT)) if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        } for sid, path in sorted(expected.items())],
        "output_sha256": {sid: sha256(path) for sid, path in expected.items() if path.exists()},
        "recovered_sample_ids": sorted(
            sid for sid, path in expected.items() if path.exists() and sid not in batch_samples),
        "fingerprint": fingerprint, "policy": c["policy"], "mfa_version": version,
        "mfa_acoustic_model": c["mfa_acoustic_model"],
        "mfa_acoustic_sha256": sha256(acoustic),
        "mfa_dictionary": c["mfa_dictionary"],
        "mfa_dictionary_sha256": sha256(dictionary),
        "mfa_g2p_model": c["mfa_g2p_model"], "mfa_g2p_sha256": sha256(g2p),
        "mfa_num_jobs": c["mfa_num_jobs"],
        "mfa_single_speaker_mode": c["mfa_single_speaker_mode"],
        "mfa_batch_returncode": completed.returncode,
        "aligned_samples": len(usable) - len(individual_failures),
        "mfa_failed_samples": sorted(individual_failures),
        "mfa_individual_fallback_samples": (
            sum(path.exists() for path in expected.values()) - batch_output_count),
        "excluded_digital_silence_samples": len(rows) - len(usable),
        "command": [str(Path(value).relative_to(ROOT)) if str(value).startswith(str(ROOT)) else value
                    for value in command],
    }
    dump(report_path, report)
    return report


def _normalized(value):
    return "".join(ch.lower() for ch in value if ch.isalnum())


def map_mfa_words(text, entries, offset, duration):
    """Map MFA-normalized words back to exact original character ranges."""
    aligned = [(float(a) + offset, float(b) + offset, str(label))
               for a, b, label in entries
               if str(label).strip() and str(label) != "<eps>"]
    normalized_chars = [(index, ch.lower()) for index, ch in enumerate(text) if ch.isalnum()]
    normalized_text = "".join(ch for _, ch in normalized_chars)
    cursor, raw_cursor = 0, 0
    words = []
    for index, (start, end, label) in enumerate(aligned):
        target = _normalized(label)
        if label == "[bracketed]":
            match = re.search(r"\[[^\]]+\]", text[raw_cursor:])
            if match is None:
                raise ValueError("MFA emitted [bracketed] without bracketed source text")
            char_start, char_end = raw_cursor + match.start(), raw_cursor + match.end()
            consumed = _normalized(text[char_start:char_end])
            cursor += len(consumed)
        elif label == "<unk>":
            if cursor >= len(normalized_chars):
                raise ValueError("MFA emitted an unmatched <unk>")
            char_start = normalized_chars[cursor][0]
            token = re.match(r"\S+", text[char_start:])
            char_end = char_start + len(token.group())
            cursor += len(_normalized(text[char_start:char_end]))
        else:
            found = normalized_text.find(target, cursor)
            if not target or found < cursor:
                raise ValueError(f"MFA token cannot be mapped at {index}: {label!r}")
            skipped = normalized_text[cursor:found]
            if skipped:
                raise ValueError(
                    f"MFA skipped source characters before {label!r}: {skipped!r}")
            char_start = normalized_chars[found][0]
            char_end = normalized_chars[found + len(target) - 1][0] + 1
            cursor = found + len(target)
        raw_cursor = char_end
        if not (0 <= start < end <= duration + 0.05):
            raise ValueError(f"Invalid MFA word time {start}, {end}, duration {duration}")
        clipped_end = min(duration, end)
        words.append({
            "word_index": index, "text": text[char_start:char_end],
            "char_start": char_start, "char_end": char_end,
            "start_s": max(0.0, start), "end_s": clipped_end,
            "raw_start_s": start, "raw_end_s": end,
            "probability": None, "instant": False, "end_clipped": clipped_end != end,
            "alignment_source": "mfa-3.4.1/english_mfa-3.1.0",
            "mfa_label": label,
        })
    if cursor != len(normalized_chars):
        raise ValueError(
            f"MFA did not cover normalized transcript: {cursor}/{len(normalized_chars)}")
    # Assign punctuation and whitespace deterministically to the preceding token,
    # while keeping a disjoint partition that covers the original transcript.
    words[0]["char_start"] = 0
    for current, following in zip(words, words[1:]):
        current["char_end"] = following["char_start"]
    words[-1]["char_end"] = len(text)
    for word in words:
        word["text"] = text[word["char_start"]:word["char_end"]]
    return words


def load_mfa_words(row, config):
    c = config["automatic_alignment"]
    layout = json.loads((ROOT / c["work_dir"] / "mapping.json").read_text())
    item = layout[row["sample_id"]]
    path = ROOT / c["work_dir"] / "output" / item["speaker"] / (item["stem"] + ".json")
    data = json.loads(path.read_text())
    return map_mfa_words(
        row["text"], data["tiers"]["words"]["entries"],
        row["audio_start_s"], row["duration_s"])


_VAD_MODEL = None


def vad_intervals(waveform, sample_rate, config):
    """Return deterministic Silero speech intervals in clip-relative seconds."""
    global _VAD_MODEL
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad(onnx=True)
    c = config["automatic_alignment"]
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.asarray(waveform, dtype=np.float32)), _VAD_MODEL,
        sampling_rate=sample_rate, threshold=c["vad_threshold"],
        min_speech_duration_ms=c["vad_min_speech_ms"],
        min_silence_duration_ms=c["vad_min_silence_ms"],
        speech_pad_ms=c["vad_speech_pad_ms"], return_seconds=True)
    return [[float(item["start"]), float(item["end"])] for item in timestamps]


def _secondary_span(words, char_start, char_end):
    selected = [word for word in words
                if word["char_end"] > char_start and word["char_start"] < char_end]
    if not selected:
        return None
    return [selected[0]["start_s"], selected[-1]["end_s"]]


def _interval_overlap(a, b, intervals):
    return sum(max(0.0, min(b, end) - max(a, start)) for start, end in intervals)


def add_automatic_evidence(phrases, stable_words, vad, config):
    """Attach independent-algorithm differences without calling them ground-truth errors."""
    deltas, unavailable = [], []
    for phrase in phrases:
        secondary = _secondary_span(
            stable_words, phrase["char_start"], phrase["char_end"])
        valid = (secondary is not None and np.isfinite(secondary).all()
                 and 0 <= secondary[0] < secondary[1])
        differences = ([abs(phrase["start_s"] - secondary[0]),
                        abs(phrase["end_s"] - secondary[1])] if valid else [])
        deltas.extend(differences)
        if not valid:
            unavailable.append(phrase["phrase_index"])
        duration = phrase["end_s"] - phrase["start_s"]
        phrase["automatic_boundary_evidence"] = {
            "mfa_s": [phrase["start_s"], phrase["end_s"]],
            "stable_ts_s": secondary,
            "secondary_status": "available" if valid else "missing_or_nonpositive_span",
            "absolute_difference_s": differences,
            "vad_overlap_fraction": (
                _interval_overlap(phrase["start_s"], phrase["end_s"], vad) / duration),
        }
    p90 = float(np.quantile(deltas, .9)) if deltas else None
    median = float(np.median(deltas)) if deltas else None
    c = config["automatic_alignment"]
    grade = ("C" if unavailable else
             "A" if p90 is not None and p90 <= c["grade_a_p90_difference_s"] else
             "B" if p90 is not None and p90 <= c["grade_b_p90_difference_s"] else "C")
    primary = phrases[0]["alignment_source"] if phrases else "unavailable"
    return {
        "policy": c["policy"], "primary": primary,
        "secondary": "stable-ts-2.19.1/tiny.en", "vad": "silero-vad-6.2.2",
        "endpoint_count": len(deltas), "median_absolute_difference_s": median,
        "secondary_unavailable_phrase_indices": unavailable,
        "secondary_audited_phrase_count": len(phrases),
        "p90_absolute_difference_s": p90, "maximum_absolute_difference_s": max(deltas, default=None),
        "vad_intervals_s": vad, "confidence_grade": grade,
        "metric_meaning": "Cross-aligner disagreement, not error against human ground truth.",
    }


def attach_protocol_checks(evidence, phrases, primary_complete, manual=False):
    context_only = bool(phrases and phrases[0].get("context_only", False))
    checks = {
        "primary_complete_and_quality_passed": primary_complete,
        "secondary_endpoint_coverage": (
            None if context_only or manual else evidence["endpoint_count"] == 2 * len(phrases)),
        "secondary_evidence_accounted": (
            None if context_only or manual else
            evidence.get("secondary_audited_phrase_count") == len(phrases)),
        "clip_context_policy_applied": context_only,
        "manual_correction": manual,
    }
    evidence["protocol_checks"] = checks
    evidence["protocol_passed"] = bool(
        not manual and primary_complete
        and (context_only or checks["secondary_evidence_accounted"]))


def clip_context_phrase(row, audio_missing=True, reason="digital_silence"):
    """Represent a clip without observable transcript timing at clip granularity."""
    flags = ["speech_timing_not_observable", "clip_context"]
    if audio_missing:
        flags.append("audio_modality_missing")
    else:
        flags.append("text_audio_alignment_unavailable")
    return [{
        "phrase_index": 0, "start_s": 0.0, "end_s": row["duration_s"],
        "char_start": 0, "char_end": len(row["text"]), "text": row["text"],
        "word_indices": [], "alignment_source": f"clip-context/{reason}",
        "flags": flags, "human_reviewed": False, "context_only": True,
        "audio_missing": audio_missing,
    }]
