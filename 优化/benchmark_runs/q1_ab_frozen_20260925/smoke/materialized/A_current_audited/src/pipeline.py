"""Resumable Q1 extraction. Run through ../run.py."""
import csv
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
import numpy as np
from prepare import ROOT, dump, sha256
from modalities import TextEncoder, decode_audio, audio_features, vision_features
from alignment import map_words, phrase_groups, assess_alignment, aggregate
from automatic_alignment import (
    add_automatic_evidence, clip_context_phrase, ensure_mfa_alignments,
    load_mfa_words, vad_intervals,
)


def log_event(event):
    with (ROOT / "logs/events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}, ensure_ascii=False) + "\n")


def json_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def provenance(config):
    input_records = json.loads((ROOT / "resources/input_manifest.json").read_text())
    input_hashes = {row["path"]: row["sha256"] for row in input_records}
    model_records = json.loads((ROOT / "resources/model_sources.json").read_text())
    from download_resources import expected_hashes
    locked = expected_hashes(ROOT)
    expected_models = {path for path in locked if path.startswith("resources/models/")}
    recorded_models = [row["path"] for row in model_records
                       if row["path"].startswith("resources/models/")]
    if set(recorded_models) != expected_models or len(recorded_models) != len(expected_models):
        raise ValueError("Model source manifest differs from the locked model set")
    model_hashes = {}
    for row in model_records:
        if not row["path"].startswith("resources/models/"):
            continue
        path = ROOT / row["path"]
        current = sha256(path)
        if current != row["sha256"] or current != locked[row["path"]]:
            raise ValueError(f"Model fingerprint mismatch: {row['path']}")
        model_hashes[row["path"]] = current
    code_paths = [
        ROOT / "src/pipeline.py", ROOT / "src/modalities.py",
        ROOT / "src/alignment.py", ROOT / "src/prepare.py", ROOT / "src/reaggregate.py",
        ROOT / "src/automatic_alignment.py",
        ROOT / "src/download_resources.py",
    ]
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    manual = ROOT / "annotations/manual_phrases.jsonl"
    base = {
        "config_sha256": json_hash(config),
        "manifest_sha256": sha256(ROOT / "resources/manifest.json"),
        "alignment_retry_sha256": sha256(ROOT / "configs/alignment_retry.json"),
        "requirements_lock_sha256": sha256(ROOT / "requirements.lock.txt"),
        "manual_annotations_sha256": sha256(manual) if manual.exists() else None,
        "model_sha256": model_hashes,
        "resource_lock_sha256": sha256(ROOT / "resources/resource_lock.json"),
        "code_sha256": code_hashes,
    }
    alignment_base = {
        "alignment": config["alignment"],
        "automatic_alignment": config["automatic_alignment"],
        "alignment_retry_sha256": base["alignment_retry_sha256"],
        "whisper_sha256": model_hashes["resources/models/whisper/tiny.en.pt"],
        "stable_ts_version": importlib.metadata.version("stable-ts"),
        "whisper_version": importlib.metadata.version("openai-whisper"),
    }
    return base, alignment_base, input_hashes


def manual_phrases(row):
    path = ROOT / "annotations/manual_phrases.jsonl"
    if not path.exists():
        return None
    found = [json.loads(line) for line in path.read_text().splitlines() if line.strip()
             and json.loads(line)["sample_id"] == row["sample_id"]]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("Duplicate manual annotation")
    entry = found[0]
    if not str(entry.get("reviewer", "")).strip() or not str(entry.get("reason", "")).strip():
        raise ValueError("Manual annotation requires reviewer and reason")
    result = entry["phrases"]
    prev_t, prev_c = 0, 0
    for i, p in enumerate(result):
        a, b, ca, cb = p["start_s"], p["end_s"], p["char_start"], p["char_end"]
        if (not all(isinstance(t, (int, float)) and not isinstance(t, bool) and np.isfinite(t)
                    for t in (a, b))
                or not all(type(c) is int for c in (ca, cb))):
            raise ValueError("Manual times must be finite numbers and character offsets must be integers")
        if not (prev_t <= a < b <= row["duration_s"] and prev_c <= ca < cb <= len(row["text"])):
            raise ValueError("Invalid manual interval/character range")
        if row["text"][prev_c:ca].strip():
            raise ValueError("Manual annotation skipped transcript")
        p.update({"phrase_index": i, "text": row["text"][ca:cb], "word_indices": [],
                  "alignment_source": "manual", "flags": [], "human_reviewed": False,
                  "human_annotated": True,
                  "reviewer": entry["reviewer"], "review_reason": entry["reason"]})
        prev_t, prev_c = b, cb
    if row["text"][prev_c:].strip():
        raise ValueError("Manual annotation skipped transcript tail")
    return result


def environment(config):
    import torch
    packages = {}
    for name in ("numpy", "torch", "torchaudio", "transformers", "librosa", "mediapipe",
                 "stable-ts", "openai-whisper", "opencv-contrib-python", "soundfile", "numba",
                 "silero-vad", "onnxruntime"):
        packages[name] = importlib.metadata.version(name)
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages,
            "device_used": "cpu", "mps_available": torch.backends.mps.is_available(),
            "torch_threads": config["torch_threads"],
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "ffprobe": subprocess.check_output(["ffprobe", "-version"], text=True).splitlines()[0],
            "config_sha256": json_hash(config)}
    dump(ROOT / "reports/environment.json", info)


def run(config, selected=None, force=False):
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_begin = time.perf_counter()
    import torch
    import stable_whisper
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    environment(config)
    rows = json.loads((ROOT / "resources/manifest.json").read_text())
    if selected is not None:
        rows = [r for r in rows if r["sample_id"] in selected]
    config_hash = json_hash(config)
    provenance_base, alignment_base, input_hashes = provenance(config)
    prepare_begin = time.perf_counter()
    # MFA aligns a corpus in one batch, so ensure all selected waveforms exist first.
    for row in rows:
        source_sha = sha256(ROOT / row["source_file"])
        if source_sha != input_hashes.get(row["source_file"]):
            raise ValueError(f"Input fingerprint mismatch: {row['source_file']}")
        decode_audio(row, config["audio"], source_sha)
    audio_prepare_s = time.perf_counter() - prepare_begin
    mfa_begin = time.perf_counter()
    mfa_report = ensure_mfa_alignments(rows, config, force=force)
    mfa_call_s = time.perf_counter() - mfa_begin
    enc, aligner = None, None
    errors, run_records = [], []
    for index, row in enumerate(rows):
        sid = row["sample_id"]
        output = ROOT / "outputs/features" / (sid + ".npz")
        detail = ROOT / "outputs/details" / (sid + ".json")
        source_path = ROOT / row["source_file"]
        source_sha = sha256(source_path)
        if source_sha != input_hashes.get(row["source_file"]):
            raise ValueError(f"Input fingerprint mismatch: {row['source_file']}")
        fingerprint_components = {**provenance_base, "source_sha256": source_sha}
        artifact_fingerprint = json_hash(fingerprint_components)
        alignment_fingerprint = json_hash({**alignment_base, "source_sha256": source_sha,
                                          "text": row["text"], "audio_start_s": row["audio_start_s"],
                                          "duration_s": row["duration_s"]})
        previous = json.loads(detail.read_text()) if detail.exists() else {}
        if not force and output.exists() and detail.exists():
            if previous.get("artifact_fingerprint") == artifact_fingerprint:
                print(f"[{index+1}/{len(rows)}] cached {sid}", flush=True)
                run_records.append({"sample_id": sid, "status": "cached",
                                    "artifact_fingerprint": artifact_fingerprint})
                continue
        begin, stages = time.perf_counter(), {}
        print(f"[{index+1}/{len(rows)}] start {sid}", flush=True)
        try:
            cache = ROOT / "intermediate/samples" / sid
            cache.mkdir(parents=True, exist_ok=True)
            y, _ = decode_audio(row, config["audio"], source_sha)
            if len(y) / config["audio"]["sample_rate"] > row["duration_s"] + 0.1:
                raise ValueError("Decoded audio exceeds container duration")
            start = time.perf_counter()
            aligned_path = cache / "alignment_raw.json"
            manual = manual_phrases(row)
            digitally_silent = not np.any(y != 0)
            stable_words = []
            if digitally_silent:
                if manual is not None:
                    raise ValueError("Digital-silent input uses the fixed clip-context policy")
                words, phrases = [], clip_context_phrase(row)
                alignment_evidence = {
                    "policy": config["automatic_alignment"]["policy"],
                    "primary": "clip-context/audio-missing", "secondary": "not-applicable",
                    "vad": config["automatic_alignment"]["vad_model"], "endpoint_count": 0,
                    "median_absolute_difference_s": None,
                    "p90_absolute_difference_s": None,
                    "maximum_absolute_difference_s": None, "vad_intervals_s": [],
                    "confidence_grade": "M",
                    "metric_meaning": "Audio is unavailable; interval denotes clip context, not speech timing.",
                }
                log_event({"sample_id": sid, "status": "clip_context_audio_missing",
                           "reason": "Digital silence is represented by an explicit missing-audio mask."})
            elif manual is None:
                alignment_cache_valid = (
                    aligned_path.exists()
                    and previous.get("alignment_fingerprint") == alignment_fingerprint
                )
                if not alignment_cache_valid or force:
                    if aligner is None:
                        load_time = time.perf_counter()
                        aligner = stable_whisper.load_model(
                            config["alignment"]["model"], device="cpu",
                            download_root=str(ROOT / config["alignment"]["model_dir"]))
                        log_event({"stage": "load_alignment_model", "seconds": time.perf_counter() - load_time})
                    result = aligner.align(
                        y, row["text"], language="en", verbose=None,
                        token_step=100, remove_instant_words=False,
                        regroup=False, suppress_silence=config["alignment"]["suppress_silence"],
                        vad=config["alignment"]["vad"], q_levels=config["alignment"]["q_levels"],
                        k_size=config["alignment"]["k_size"], fast_mode=False, stream=False,
                        max_word_dur=3.0, word_dur_factor=2.0, nonspeech_skip=5.0)
                    if result is None:
                        raise ValueError("Alignment returned None; manual annotation required")
                    dump(aligned_path, result.to_dict())
                try:
                    stable_words = map_words(
                        row["text"], json.loads(aligned_path.read_text()),
                        row["audio_start_s"], row["duration_s"])
                except ValueError as first_error:
                    retry_path = cache / "alignment_retry.json"
                    retry_config = json.loads((ROOT / "configs/alignment_retry.json").read_text())
                    retry_cache_valid = (
                        retry_path.exists()
                        and previous.get("alignment_fingerprint") == alignment_fingerprint
                    )
                    if not retry_cache_valid or force:
                        if aligner is None:
                            aligner = stable_whisper.load_model(
                                config["alignment"]["model"], device="cpu",
                                download_root=str(ROOT / config["alignment"]["model_dir"]))
                        result = aligner.align(y, row["text"], language="en", verbose=None,
                                               **retry_config["options"])
                        dump(retry_path, result.to_dict())
                    stable_words = map_words(
                        row["text"], json.loads(retry_path.read_text()),
                        row["audio_start_s"], row["duration_s"])
                    log_event({"sample_id": sid, "status": "alignment_retry_recovered",
                               "first_error": str(first_error), "retry_options": retry_config["options"]})
                try:
                    words = load_mfa_words(row, config)
                    phrases = phrase_groups(
                        words, row["text"], config["alignment"], config["alignment_quality"])
                except (FileNotFoundError, ValueError) as mfa_error:
                    words = []
                    phrases = clip_context_phrase(
                        row, audio_missing=False, reason="mfa-primary-unavailable")
                    log_event({"sample_id": sid, "status": "mfa_primary_fallback",
                               "error": str(mfa_error)})
                vad = vad_intervals(y, config["audio"]["sample_rate"], config)
                if phrases[0].get("context_only", False):
                    alignment_evidence = {
                        "policy": config["automatic_alignment"]["policy"],
                        "primary": phrases[0]["alignment_source"],
                        "secondary": "stable-ts-2.19.1/tiny.en",
                        "vad": config["automatic_alignment"]["vad_model"], "endpoint_count": 0,
                        "median_absolute_difference_s": None,
                        "p90_absolute_difference_s": None,
                        "maximum_absolute_difference_s": None, "vad_intervals_s": vad,
                        "confidence_grade": "C",
                        "metric_meaning": (
                            "No reliable word-level primary alignment; interval denotes clip context."),
                    }
                else:
                    alignment_evidence = add_automatic_evidence(
                        phrases, stable_words, vad, config)
                alignment_evidence["mfa_batch_fingerprint"] = mfa_report["fingerprint"]
            else:
                words, phrases = [], manual
                alignment_evidence = {
                    "policy": config["automatic_alignment"]["policy"],
                    "primary": "manual-correction", "secondary": "not-applicable",
                    "vad": config["automatic_alignment"]["vad_model"], "endpoint_count": 0,
                    "median_absolute_difference_s": None,
                    "p90_absolute_difference_s": None,
                    "maximum_absolute_difference_s": None, "vad_intervals_s": [],
                    "confidence_grade": "manual",
                    "metric_meaning": "Manual correction; excluded from automatic disagreement statistics.",
                }
            alignment_evidence["mfa_batch_fingerprint"] = mfa_report["fingerprint"]
            automatic_words = copy.deepcopy(
                words if manual is None else previous.get("automatic_words", previous.get("words", [])))
            automatic_phrases = copy.deepcopy(
                phrases if manual is None else previous.get("automatic_phrases", previous.get("phrases", [])))
            automatic_quality = assess_alignment(
                automatic_words, automatic_phrases, config["alignment_quality"])
            if manual is not None:
                alignment_quality = assess_alignment([], manual, config["alignment_quality"])
            else:
                alignment_quality = assess_alignment(words, phrases, config["alignment_quality"])
            context_only = bool(phrases and phrases[0].get("context_only", False))
            primary_complete = bool(phrases and alignment_quality["hard_passed"])
            from automatic_alignment import attach_protocol_checks
            attach_protocol_checks(
                alignment_evidence, phrases, primary_complete, manual is not None)
            stages["alignment_s"] = time.perf_counter() - start
            start = time.perf_counter()
            if enc is None:
                enc = TextEncoder(config["text"])
            text = enc.encode(row["text"])
            np.savez_compressed(cache / "text.npz", **text)
            stages["text_s"] = time.perf_counter() - start
            start = time.perf_counter()
            acoustic = audio_features(y, row["audio_start_s"], config["audio"])
            np.savez_compressed(cache / "audio.npz", **acoustic)
            stages["audio_s"] = time.perf_counter() - start
            start = time.perf_counter()
            visual = vision_features(row, config["vision"])
            np.savez_compressed(cache / "vision.npz", **visual)
            stages["vision_s"] = time.perf_counter() - start
            save_features(output, row, config, phrases, alignment_quality, text, acoustic, visual)
            if digitally_silent:
                alignment_status = "aligned_clip_context_audio_missing"
            elif context_only:
                alignment_status = "aligned_clip_context_speech_unavailable"
            elif manual is not None:
                alignment_status = "aligned_manual_pending_approval"
            elif alignment_evidence["protocol_passed"]:
                alignment_status = "aligned_auto_protocol_passed"
            else:
                alignment_status = "aligned_auto_protocol_failed"
            meta = {**row, "config_sha256": config_hash,
                    "artifact_fingerprint": artifact_fingerprint,
                    "alignment_fingerprint": alignment_fingerprint,
                    "fingerprint_components": fingerprint_components,
                    "alignment_status": alignment_status,
                    "alignment_quality": alignment_quality,
                    "automatic_boundary_evidence": alignment_evidence,
                    "digitally_silent_audio": digitally_silent,
                    "speech_alignment_available": not context_only,
                    "audio_observed": not digitally_silent,
                    "sample_flags": (
                        ["digital_silence", "audio_modality_missing", "clip_context"]
                        if digitally_silent else
                        ["text_audio_alignment_unavailable", "clip_context"]
                        if context_only else []),
                    "feature_file": str(output.relative_to(ROOT)),
                    "schema_version": config["schema_version"], "phrases": phrases, "words": words,
                    "automatic_words": automatic_words, "automatic_phrases": automatic_phrases,
                    "secondary_alignment_words": stable_words,
                    "automatic_alignment_quality": automatic_quality,
                    "token_count": len(text["ids"]), "text_windows": int(text["windows"]),
                    "decoded_audio_duration_s": len(y) / config["audio"]["sample_rate"],
                    "audio_window_count": len(acoustic["values"]), "pitch_window_count": len(acoustic["f0"]),
                    "pitch_voiced_rate": float(acoustic["f0_valid"].mean()),
                    "pitch_boundary_hits": int(acoustic["pitch_boundary_hits"]),
                    "sampled_frame_count": len(visual["time"]),
                    "face_valid_rate": float(visual["valid"].all(1).mean()),
                    "face_crop_fallback_frames": int(visual["crop_fallback_used"].sum()),
                    "face_crop_fallback_valid_frames": int(
                        (visual["crop_fallback_used"] & visual["valid"].all(1)).sum()),
                    "face_geometry_rejected_frames": int(visual["geometry_rejected"].sum()),
                    "multi_face_frames": int((visual["faces"] > 1).sum()),
                    "ambiguous_face_frames": int(visual["ambiguous"].sum()),
                    "stage_seconds": stages, "elapsed_s": time.perf_counter() - begin}
            dump(detail, meta)
            from reaggregate import seal_native
            seal_native(row, config, provenance_base, source_sha)
            log_event({"sample_id": sid, "status": "extracted", "phrases": len(phrases),
                       "seconds": meta["elapsed_s"], "stage_seconds": stages})
            run_records.append({
                "sample_id": sid, "status": "extracted", "seconds": meta["elapsed_s"],
                "stage_seconds": stages, "artifact_fingerprint": artifact_fingerprint,
                "feature_sha256": sha256(output), "alignment_status": alignment_status})
            print(f"  saved {len(phrases)} intervals; face={meta['face_valid_rate']:.1%}; "
                  f"{meta['elapsed_s']:.2f}s", flush=True)
        except Exception as exc:
            error = {"sample_id": sid, "status": "failed", "error": str(exc),
                     "traceback": traceback.format_exc(), "stage_seconds": stages}
            errors.append(error)
            run_records.append({"sample_id": sid, "status": "failed",
                                "error": str(exc).replace(str(ROOT), ".")})
            log_event(error)
            print(error["traceback"], flush=True)
    summarize()
    dump(ROOT / "reports/last_run.json", {
        "schema": "q1-execution-2", "started_at": run_started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested": len(rows), "force": force,
        "errors": [r for r in run_records if r["status"] == "failed"],
        "wall_seconds": time.perf_counter() - run_begin,
        "audio_preparation_seconds": audio_prepare_s, "mfa_call_seconds": mfa_call_s,
        "mfa_run_id": mfa_report["run_id"], "mfa_batch_fingerprint": mfa_report["fingerprint"],
        "provenance_base_sha256": json_hash(provenance_base), "samples": run_records,
        "timing_scope": "extract call including imports, audio preparation, MFA, all selected feature stages and summary; excludes installation, download, prepare/probing, validation, audit and figures.",
    })
    if errors:
        raise RuntimeError(f"{len(errors)} samples failed; see logs/events.jsonl")


def save_features(output, row, config, phrases, quality, text, acoustic, visual):
    """Shared writer: extraction and correction reaggregation have the same numeric contract."""
    x, mask = (aggregate(phrases, text, acoustic, visual) if phrases else
               (np.empty((0, 150), np.float32), np.empty((0, 150), bool)))
    if x.shape[1] != 150 or not np.isfinite(x).all():
        raise ValueError("Invalid final feature matrix")
    context_only = bool(phrases and all(p.get("context_only", False) for p in phrases))
    audio_observed = bool(phrases and not any(p.get("audio_missing", False) for p in phrases))
    np.savez_compressed(
        output, features=x, valid=mask,
        intervals=np.array([[p["start_s"], p["end_s"]] for p in phrases], dtype=np.float64).reshape(-1, 2),
        char_ranges=np.array([[p["char_start"], p["char_end"]] for p in phrases], dtype=np.int32).reshape(-1, 2),
        sample_id=np.array(row["sample_id"]), duration_s=np.array(row["duration_s"], dtype=np.float64),
        text_token_ids=text["ids"], text_token_offsets=text["offsets"],
        sampled_frame_indices=visual["frame_index"], sampled_frame_pts=visual["pts"],
        alignment_available=np.array(bool(phrases)), alignment_quality_passed=np.array(quality["hard_passed"]),
        speech_alignment_available=np.array(bool(phrases) and not context_only),
        audio_observed=np.array(audio_observed),
        granularity=np.array("clip_context" if context_only else "phrase"),
        schema_version=np.array(config["schema_version"]))


def summarize():
    paths = sorted((ROOT / "outputs/details").glob("*.json"))
    items = [json.loads(p.read_text()) for p in paths]
    with (ROOT / "outputs/alignment.jsonl").open("w", encoding="utf-8") as f:
        for row in items:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = []
    for row in items:
        with np.load(ROOT / row["feature_file"], allow_pickle=False) as z:
            valid = z["valid"]
            intervals = z["intervals"]
            flags = sorted({flag for p in row["phrases"] for flag in p["flags"]} | set(row.get("sample_flags", [])))
            summary.append({
                "sample_id": row["sample_id"], "video_id": row["video_id"], "clip_id": row["clip_id"],
                "duration_s": row["duration_s"], "modalities": "text/audio/vision",
                "text_dim": 128, "audio_dim": 16, "vision_dim": 6,
                "granularity": str(z["granularity"]), "phrases": len(valid),
                "alignment_available": bool(len(valid)),
                "speech_alignment_available": bool(z["speech_alignment_available"]),
                "audio_observed": bool(z["audio_observed"]),
                "alignment_quality_passed": row["alignment_quality"]["hard_passed"],
                "automatic_protocol_passed": row["automatic_boundary_evidence"]["protocol_passed"],
                "automatic_confidence_grade": row["automatic_boundary_evidence"]["confidence_grade"],
                "alignment_status": row["alignment_status"],
                "phrase_time_coverage": float(np.diff(intervals, axis=1).sum() / row["duration_s"]),
                "text_valid_rate": float(valid[:, :128].mean()) if len(valid) else "",
                "audio_valid_rate": float(valid[:, 128:144].mean()) if len(valid) else "",
                "vision_valid_rate": float(valid[:, 144:].mean()) if len(valid) else "",
                "frame_face_valid_rate": row["face_valid_rate"],
                "alignment_source": ";".join(sorted({p["alignment_source"] for p in row["phrases"]})) if len(valid) else "unavailable",
                "flags": ";".join(flags), "feature_bytes": (ROOT / row["feature_file"]).stat().st_size,
                "feature_file": row["feature_file"],
            })
    if summary:
        with (ROOT / "outputs/q1_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return summary
