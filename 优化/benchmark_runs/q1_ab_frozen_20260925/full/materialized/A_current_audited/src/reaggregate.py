"""Reuse verified native features and automatic word evidence after boundary corrections."""
import copy
import hashlib
import importlib.metadata
import io
import json
import subprocess
import time
import zipfile
import numpy as np
from prepare import ROOT, dump, sha256
from alignment import aggregate, assess_alignment, phrase_groups
from pipeline import json_hash, manual_phrases, provenance, save_features, summarize, log_event
from automatic_alignment import (
    add_automatic_evidence, clip_context_phrase, ensure_mfa_alignments,
    load_mfa_words, vad_intervals,
)

SNAPSHOT = "intermediate/acceptance_revision_20260924/before.zip"
NATIVE_FILES = ("text.npz", "audio.npz", "vision.npz")


def native_dependencies(row, config, base, source_sha):
    return {
        "row": row, "source_sha256": source_sha,
        "config": {k: config[k] for k in ("text", "audio", "vision", "seed", "torch_threads")},
        "model_sha256": base["model_sha256"],
        "modality_code_sha256": base["code_sha256"]["src/modalities.py"],
        "prepare_code_sha256": base["code_sha256"]["src/prepare.py"],
        "requirements_lock_sha256": base["requirements_lock_sha256"],
        "probe_sha256": sha256(ROOT / "intermediate/probes" / (row["sample_id"] + ".json")),
        "runtime": {name: importlib.metadata.version(name) for name in (
            "numpy", "torch", "transformers", "librosa", "mediapipe", "opencv-contrib-python",
            "soundfile", "numba")},
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }


def seal_native(row, config, base, source_sha, migration=None):
    cache = ROOT / "intermediate/samples" / row["sample_id"]
    meta = json.loads((ROOT / "outputs/details" / (row["sample_id"] + ".json")).read_text())
    seal = {
        "dependencies": native_dependencies(row, config, base, source_sha),
        "files": {name: sha256(cache / name) for name in NATIVE_FILES},
        "automatic_words_sha256": json_hash(meta.get("automatic_words", meta["words"])),
        "alignment_config": config["alignment"],
        "alignment_retry_sha256": base["alignment_retry_sha256"],
        "migration": migration,
    }
    dump(cache / "native_provenance.json", seal)


def check_sealed_native(seal, dependencies, cache, words, config, base):
    if seal["dependencies"] != dependencies:
        raise ValueError("Native extraction dependencies changed; run extract instead")
    if seal["automatic_words_sha256"] != json_hash(words):
        raise ValueError("Automatic word evidence changed")
    if (seal["alignment_config"] != config["alignment"]
            or seal["alignment_retry_sha256"] != base["alignment_retry_sha256"]):
        raise ValueError("Automatic alignment configuration changed; run extract instead")
    for name in NATIVE_FILES:
        if sha256(cache / name) != seal["files"][name]:
            raise ValueError(f"Native feature changed: {name}")


def verify_legacy(row, previous, config, base, source_sha, native):
    """One-time migration from the preserved, numerically audited pre-revision snapshot."""
    with zipfile.ZipFile(ROOT / SNAPSHOT) as snapshot:
        archived = json.loads(snapshot.read(f"outputs/details/{row['sample_id']}.json"))
        if previous != archived:
            raise ValueError("Unsealed legacy metadata differs from preserved snapshot")
        components = previous["fingerprint_components"]
        if previous["artifact_fingerprint"] != json_hash(components):
            raise ValueError("Legacy artifact fingerprint is inconsistent")
        old_config = json.loads(snapshot.read("configs/q1.json"))
        if components["config_sha256"] != hashlib.sha256(snapshot.read("configs/q1.json")).hexdigest():
            raise ValueError("Legacy configuration is not the archived configuration")
        if config != old_config:
            raise ValueError("Legacy migration requires the original configuration")
        for name in ("model_sha256", "requirements_lock_sha256", "alignment_retry_sha256"):
            if components[name] != base[name]:
                raise ValueError(f"Legacy dependency mismatch: {name}")
        for name in ("src/modalities.py", "src/prepare.py"):
            if components["code_sha256"][name] != base["code_sha256"][name]:
                raise ValueError(f"Legacy modality dependency mismatch: {name}")
        if components["source_sha256"] != source_sha or any(previous[k] != v for k, v in row.items()):
            raise ValueError("Legacy input, text or time coordinates changed")
        environment = json.loads(snapshot.read("reports/environment.json"))
        current = native_dependencies(row, config, base, source_sha)
        if any(environment["packages"][k] != v for k, v in current["runtime"].items()):
            raise ValueError("Legacy runtime package versions changed")
        if environment["ffmpeg"] != current["ffmpeg"]:
            raise ValueError("Legacy ffmpeg changed")
        with np.load(io.BytesIO(snapshot.read(previous["feature_file"])), allow_pickle=False) as original:
            tx, au, vi = native
            if previous["phrases"]:
                rebuilt, mask = aggregate(copy.deepcopy(previous["phrases"]), tx, au, vi)
                np.testing.assert_array_equal(rebuilt, original["features"])
                np.testing.assert_array_equal(mask, original["valid"])
            else:
                for field, value in {
                    "unaligned_text_embeddings": tx["embeddings"],
                    "unaligned_audio_values": au["values"],
                    "unaligned_audio_intervals": au["intervals"],
                    "unaligned_f0": au["f0"], "unaligned_f0_valid": au["f0_valid"],
                    "unaligned_f0_intervals": au["f0_intervals"],
                    "unaligned_vision_values": vi["values"],
                    "unaligned_vision_valid": vi["valid"], "unaligned_vision_times": vi["time"],
                }.items():
                    np.testing.assert_array_equal(value, original[field])
            np.testing.assert_array_equal(tx["ids"], original["text_token_ids"])
            np.testing.assert_array_equal(tx["offsets"], original["text_token_offsets"])
            np.testing.assert_array_equal(vi["pts"], original["sampled_frame_pts"])
            np.testing.assert_array_equal(vi["frame_index"], original["sampled_frame_indices"])
    return {"snapshot": SNAPSHOT, "check": "original dependencies and exact numeric reconstruction",
            "limitation": "Legacy native arrays had no individual historical hashes; sealed after these checks."}


def main(config, selected=None):
    begin = time.perf_counter()
    base, alignment_base, input_hashes = provenance(config)
    rows = json.loads((ROOT / "resources/manifest.json").read_text())
    if selected is not None:
        rows = [r for r in rows if r["sample_id"] in selected]
        if set(selected) != {r["sample_id"] for r in rows}:
            raise ValueError("Unknown selected sample")
    mfa_report = ensure_mfa_alignments(rows, config)
    records = []
    # All caches are checked before writing any output.
    for row in rows:
        sid = row["sample_id"]
        source_sha = sha256(ROOT / row["source_file"])
        if source_sha != input_hashes[row["source_file"]]:
            raise ValueError(f"Source hash mismatch: {sid}")
        cache = ROOT / "intermediate/samples" / sid
        previous = json.loads((ROOT / "outputs/details" / (sid + ".json")).read_text())
        native = []
        for name in NATIVE_FILES:
            with np.load(cache / name, allow_pickle=False) as z:
                native.append({k: z[k] for k in z.files})
        seal_path = cache / "native_provenance.json"
        if seal_path.exists():
            check_sealed_native(json.loads(seal_path.read_text()),
                                native_dependencies(row, config, base, source_sha), cache,
                                previous.get("automatic_words", previous["words"]), config, base)
        else:
            migration = verify_legacy(row, previous, config, base, source_sha, native)
            seal_native(row, config, base, source_sha, migration)
        # Validate all manual edits before replacing any features.
        manual = manual_phrases(row)
        if previous["digitally_silent_audio"] and manual is not None:
            raise ValueError("Digital-silent input cannot acquire fabricated acoustic boundaries")
    for index, row in enumerate(rows):
        start = time.perf_counter()
        sid = row["sample_id"]
        detail = ROOT / "outputs/details" / (sid + ".json")
        meta = json.loads(detail.read_text())
        cache = ROOT / "intermediate/samples" / sid
        native = []
        for name in NATIVE_FILES:
            with np.load(cache / name, allow_pickle=False) as z:
                native.append({k: z[k] for k in z.files})
        stable_words = copy.deepcopy(meta.get(
            "secondary_alignment_words", meta.get("automatic_words", meta["words"])))
        digitally_silent = meta["digitally_silent_audio"]
        if digitally_silent:
            words, automatic = [], clip_context_phrase(row)
            evidence = {
                "policy": config["automatic_alignment"]["policy"],
                "primary": "clip-context/audio-missing", "secondary": "not-applicable",
                "vad": config["automatic_alignment"]["vad_model"], "endpoint_count": 0,
                "median_absolute_difference_s": None, "p90_absolute_difference_s": None,
                "maximum_absolute_difference_s": None, "vad_intervals_s": [],
                "confidence_grade": "M",
                "metric_meaning": "Audio is unavailable; interval denotes clip context, not speech timing.",
            }
        else:
            try:
                words = load_mfa_words(row, config)
                automatic = phrase_groups(
                    words, row["text"], config["alignment"], config["alignment_quality"])
            except (FileNotFoundError, ValueError):
                words = []
                automatic = clip_context_phrase(
                    row, audio_missing=False, reason="mfa-primary-unavailable")
            import soundfile as sf
            waveform, sample_rate = sf.read(
                ROOT / "intermediate/audio" / (sid + ".wav"), dtype="float32")
            vad = vad_intervals(waveform, sample_rate, config)
            if automatic[0].get("context_only", False):
                evidence = {
                    "policy": config["automatic_alignment"]["policy"],
                    "primary": automatic[0]["alignment_source"],
                    "secondary": "stable-ts-2.19.1/tiny.en",
                    "vad": config["automatic_alignment"]["vad_model"], "endpoint_count": 0,
                    "median_absolute_difference_s": None, "p90_absolute_difference_s": None,
                    "maximum_absolute_difference_s": None, "vad_intervals_s": vad,
                    "confidence_grade": "C",
                    "metric_meaning": (
                        "No reliable word-level primary alignment; interval denotes clip context."),
                }
            else:
                evidence = add_automatic_evidence(
                    automatic, stable_words, vad, config)
        evidence["mfa_batch_fingerprint"] = mfa_report["fingerprint"]
        manual = manual_phrases(row)
        phrases = copy.deepcopy(automatic) if manual is None else manual
        if manual is not None:
            evidence = {
                "policy": config["automatic_alignment"]["policy"],
                "primary": "manual-correction", "secondary": "not-applicable",
                "vad": config["automatic_alignment"]["vad_model"], "endpoint_count": 0,
                "median_absolute_difference_s": None,
                "p90_absolute_difference_s": None,
                "maximum_absolute_difference_s": None, "vad_intervals_s": [],
                "confidence_grade": "manual",
                "metric_meaning": (
                    "Manual correction; excluded from automatic disagreement statistics."),
                "mfa_batch_fingerprint": mfa_report["fingerprint"],
            }
        quality = assess_alignment(words if manual is None else [], phrases, config["alignment_quality"])
        context_only = bool(phrases and phrases[0].get("context_only", False))
        primary_complete = bool(phrases and quality["hard_passed"])
        from automatic_alignment import attach_protocol_checks
        attach_protocol_checks(evidence, phrases, primary_complete, manual is not None)
        output = ROOT / meta["feature_file"]
        with np.load(output, allow_pickle=False) as z:
            before = {k: z[k] for k in z.files}
        save_features(output, row, config, phrases, quality, *native)
        with np.load(output, allow_pickle=False) as z:
            changed_fields = [k for k in z.files if k not in before or not np.array_equal(z[k], before[k])]
        source_sha = input_hashes[row["source_file"]]
        components = {**base, "source_sha256": source_sha}
        meta.update({
            **row, "config_sha256": json_hash(config), "fingerprint_components": components,
            "artifact_fingerprint": json_hash(components),
            "schema_version": config["schema_version"],
            "alignment_fingerprint": json_hash({**alignment_base, "source_sha256": source_sha,
                                              "text": row["text"], "audio_start_s": row["audio_start_s"],
                                              "duration_s": row["duration_s"]}),
            "words": words if manual is None else [], "phrases": phrases,
            "automatic_words": words, "automatic_phrases": automatic,
            "secondary_alignment_words": stable_words,
            "automatic_alignment_quality": assess_alignment(words, automatic, config["alignment_quality"]),
            "alignment_quality": quality,
            "automatic_boundary_evidence": evidence,
            "speech_alignment_available": not context_only,
            "audio_observed": not digitally_silent,
            "sample_flags": (
                ["digital_silence", "audio_modality_missing", "clip_context"]
                if digitally_silent else
                ["text_audio_alignment_unavailable", "clip_context"]
                if context_only else []),
            "alignment_status": (
                "aligned_clip_context_audio_missing" if digitally_silent else
                "aligned_clip_context_speech_unavailable" if context_only else
                "aligned_manual_pending_approval" if manual is not None else
                "aligned_auto_protocol_passed" if evidence["protocol_passed"] else
                "aligned_auto_protocol_failed"),
            "face_crop_fallback_valid_frames": int(
                (native[2]["crop_fallback_used"] & native[2]["valid"].all(1)).sum()),
            "reaggregation": {"seconds": time.perf_counter() - start, "native_features_reused": True,
                             "manual_correction_applied": manual is not None,
                             "changed_npz_fields": changed_fields},
        })
        dump(detail, meta)
        records.append({"sample_id": sid, **meta["reaggregation"]})
        log_event({"sample_id": sid, "status": "reaggregated", **meta["reaggregation"]})
        print(f"[{index+1}/{len(rows)}] reaggregated {sid}; changed={changed_fields}", flush=True)
    summarize()
    report = {"samples": len(records), "elapsed_s": time.perf_counter() - begin,
              "manual_corrections_applied": sum(r["manual_correction_applied"] for r in records),
              "unchanged_npz_arrays": sum(not r["changed_npz_fields"] for r in records),
              "records": records}
    dump(ROOT / "reports/reaggregation.json", report)
