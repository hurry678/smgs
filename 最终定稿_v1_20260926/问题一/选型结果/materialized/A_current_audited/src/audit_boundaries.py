"""Audio-only ASR audit; no transcript prompt, no fabricated human references."""
import importlib.metadata
import json
import re
import subprocess
import time

import numpy as np

from prepare import ROOT, dump, sha256
from pipeline import json_hash
from download_resources import fetch

OPTIONS = {
    "language": "en", "task": "transcribe", "temperature": 0.0, "beam_size": 5,
    "condition_on_previous_text": False, "initial_prompt": None,
    "word_timestamps": True, "fp16": False, "verbose": None,
}


def tokens(text):
    return [(m.group().lower().replace("’", "'"), m.start(), m.end())
            for m in re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", text)]


def align_tokens(reference, predicted):
    """Unit-cost word edit distance, plus monotone exact matches."""
    n, m = len(reference), len(predicted)
    costs = np.zeros((n + 1, m + 1), dtype=int)
    costs[:, 0], costs[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i, j] = min(
                costs[i - 1, j - 1] + (reference[i - 1] != predicted[j - 1]),
                costs[i - 1, j] + 1, costs[i, j - 1] + 1)
    i, j, matches = n, m, {}
    while i or j:
        if i and j and costs[i, j] == costs[i - 1, j - 1] + (reference[i - 1] != predicted[j - 1]):
            if reference[i - 1] == predicted[j - 1]:
                matches[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i and costs[i, j] == costs[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return int(costs[n, m]), matches


def compare(meta, raw):
    reference = tokens(meta["text"])
    words = []
    for segment in raw.get("segments", []):
        for word in segment.get("words", []):
            for token, _, _ in tokens(word["word"]):
                words.append({"text": token, "start_s": word["start"] + meta["audio_start_s"],
                              "end_s": word["end"] + meta["audio_start_s"]})
    edits, matches = align_tokens([t[0] for t in reference], [w["text"] for w in words])
    phrases = []
    for phrase in meta["phrases"]:
        indices = [i for i, (_, a, b) in enumerate(reference)
                   if a >= phrase["char_start"] and b <= phrase["char_end"]]
        matched = [i for i in indices if i in matches]
        record = {
            "phrase_index": phrase["phrase_index"], "text": phrase["text"],
            "primary_s": [phrase["start_s"], phrase["end_s"]],
            "reference_word_count": len(indices), "exact_matched_words": len(matched),
            "asr_s": None, "absolute_difference_s": [],
        }
        if (meta["speech_alignment_available"] and indices and matched == indices):
            a, b = words[matches[indices[0]]]["start_s"], words[matches[indices[-1]]]["end_s"]
            if 0 <= a < b <= meta["duration_s"] + 0.05:
                record["asr_s"] = [a, b]
                record["absolute_difference_s"] = [
                    abs(a - phrase["start_s"]), abs(b - phrase["end_s"])]
        phrases.append(record)
    return {
        "sample_id": meta["sample_id"], "source_sha256": meta["fingerprint_components"]["source_sha256"],
        "artifact_fingerprint": meta["artifact_fingerprint"],
        "text": meta["text"], "asr_text": raw.get("text", ""),
        "status": raw["status"], "word_edit_count": edits,
        "reference_word_count": len(reference), "exact_matched_words": len(matches),
        "word_error_ratio": edits / len(reference) if reference else None,
        "phrases": phrases,
    }


def main(force=False):
    import soundfile as sf
    import torch
    import whisper
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    np.random.seed(2026)
    asset = fetch(whisper._MODELS["base.en"], ROOT / "resources/audit_models/base.en.pt")
    dump(ROOT / "resources/audit_model_source.json", {
        **asset, "model": "OpenAI Whisper base.en", "license": "MIT",
        "use": "Audio-only independent-model audit; not used to fit or replace final boundaries.",
        "package_version": importlib.metadata.version("openai-whisper")})
    manifest = json.loads((ROOT / "resources/manifest.json").read_text())
    model, results, raw_hashes = None, [], {}
    begin = time.perf_counter()
    cache = ROOT / "intermediate/boundary_audit"
    cache.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(manifest):
        sid = row["sample_id"]
        audio_path = ROOT / "intermediate/audio" / (sid + ".wav")
        audio_sha = sha256(audio_path)
        identity = json_hash({
            "audio_sha256": audio_sha, "model_sha256": asset["sha256"], "options": OPTIONS,
            "whisper": importlib.metadata.version("openai-whisper"), "seed": 2026,
        })
        raw_path = cache / (sid + ".json")
        raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
        if force or raw.get("fingerprint") != identity:
            y, sr = sf.read(audio_path, dtype="float32")
            if sr != 16000:
                raise ValueError("ASR requires the 16 kHz prepared waveform")
            start = time.perf_counter()
            if np.any(y != 0):
                if model is None:
                    model = whisper.load_model(str(ROOT / asset["path"]), device="cpu")
                decoded = model.transcribe(y, **OPTIONS)
                status = "decoded_without_transcript_prompt"
            else:
                decoded = {"text": "", "segments": []}
                status = "digital_silence_not_decoded"
            raw = {
                "sample_id": sid, "fingerprint": identity, "audio_sha256": audio_sha,
                "model_sha256": asset["sha256"], "status": status,
                "seconds": time.perf_counter() - start, **decoded,
            }
            dump(raw_path, raw)
        raw_hashes[sid] = sha256(raw_path)
        meta = json.loads((ROOT / "outputs/details" / (sid + ".json")).read_text())
        if sha256(ROOT / row["source_file"]) != meta["fingerprint_components"]["source_sha256"]:
            raise ValueError(f"Changed source: {sid}")
        results.append(compare(meta, raw))
        print(f"[{i+1}/{len(manifest)}] ASR audit {sid}: {raw['text']}", flush=True)
    differences = [d for r in results for p in r["phrases"] for d in p["absolute_difference_s"]]
    decoded = [r for r in results if r["status"] == "decoded_without_transcript_prompt"]
    report = {
        "schema": "q1-audio-only-audit-1", "samples": results,
        "sample_count": len(results), "decoded_samples": len(decoded),
        "digital_silence_skipped": len(results) - len(decoded),
        "options": OPTIONS, "model": asset, "wall_seconds": time.perf_counter() - begin,
        "code_sha256": sha256(ROOT / "src/audit_boundaries.py"),
        "raw_output_sha256": raw_hashes, "input_transcript_used_for_decoding": False,
        "matching_rule": "Unit-cost monotone edit alignment after decoding; compare timing only when every reference word of a phrase matches exactly and ASR span is positive.",
        "matched_phrase_count": len(differences) // 2,
        "comparable_endpoint_count": len(differences),
        "absolute_difference_s": {
            "median": float(np.median(differences)) if differences else None,
            "p90": float(np.quantile(differences, .9)) if differences else None,
            "maximum": max(differences, default=None),
        },
        "transcript_word_error_ratio_decoded_samples": (
            sum(r["word_edit_count"] for r in decoded) / sum(r["reference_word_count"] for r in decoded)),
        "human_reference_count": 0,
        "interpretation": "Independent weights and audio-only decoding provide additional model evidence, not human ground truth. Timing statistics are conditional on exact lexical matches and cannot estimate accuracy on unmatched phrases. ASR errors and transcript differences both contribute to the edit ratio.",
    }
    dump(ROOT / "reports/boundary_audit.json", report)
    # Small replayable evidence set: pilot clips, largest disagreements, VAD conflicts, merged cases.
    metas = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/details").glob("*.json"))]
    chosen = set(json.loads((ROOT / "configs/pilot.json").read_text()))
    chosen.update(r["sample_id"] for r in sorted(metas, key=lambda r: (
        r["automatic_boundary_evidence"]["maximum_absolute_difference_s"] or 0), reverse=True)[:3])
    chosen.update(r["sample_id"] for r in metas
                  if not r["automatic_boundary_evidence"]["vad_intervals_s"])
    chosen.update(r["sample_id"] for r in metas
                  if any("automatic_quality_merge" in p["flags"] for p in r["phrases"]))
    out = ROOT / "outputs/audit_audio"
    out.mkdir(parents=True, exist_ok=True)
    audio_manifest = []
    for meta in metas:
        sid = meta["sample_id"]
        if sid not in chosen:
            continue
        target = out / (sid + ".m4a")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / meta["source_file"]),
                        "-vn", "-c:a", "aac", "-b:a", "64k", str(target)], check=True)
        audio_manifest.append({
            "sample_id": sid, "file": str(target.relative_to(ROOT)), "sha256": sha256(target),
            "source_sha256": meta["fingerprint_components"]["source_sha256"],
            "scope": "Complete source audio track re-encoded for optional listening, not a human annotation.",
        })
    dump(out / "manifest.json", audio_manifest)


if __name__ == "__main__":
    main()
