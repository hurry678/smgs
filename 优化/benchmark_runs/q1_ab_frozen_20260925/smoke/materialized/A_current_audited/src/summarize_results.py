"""Compute final Q1 statistics only from persisted, validated real outputs."""
import csv
import json
import numpy as np
from prepare import ROOT, dump


def q(values):
    values = np.asarray(values, dtype=float)
    return {name: float(value) for name, value in zip(
        ("min", "q25", "median", "q75", "max"),
        np.quantile(values, [0, .25, .5, .75, 1]))}


def main():
    with (ROOT / "outputs/q1_summary.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    details = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/details").glob("*.json"))]
    validation = json.loads((ROOT / "reports/validation.json").read_text())
    aligned = [r for r in rows if r["alignment_available"] == "True"]
    words = [w for d in details for w in d["words"]]
    scored_words = [w for w in words if w.get("probability") is not None]
    frames = sum(d["sampled_frame_count"] for d in details)
    face_frames = sum(round(d["face_valid_rate"] * d["sampled_frame_count"]) for d in details)
    stages = ("alignment_s", "text_s", "audio_s", "vision_s")
    result = {
        "sample_coverage": {
            "input_samples": 100, "feature_files": len(rows),
            "mapped_samples": validation["mapped_samples"],
            "automatic_quality_passed_samples": validation["automatic_quality_passed_samples"],
            "strictly_accepted_samples": validation["strictly_accepted_samples"],
            "automatically_accepted_samples": validation["automatically_accepted_samples"],
            "speech_aligned_samples": validation["speech_aligned_samples"],
            "clip_context_samples": validation["clip_context_samples"],
            "audio_missing_samples": validation["audio_missing_samples"],
            "verified_source_exception_samples": validation["verified_source_exception_samples"],
            "acceptance_status_counts": validation["acceptance_status_counts"],
            "unalignable_samples": validation["unalignable_samples"],
            "all_files_numeric_validation_passed": validation["structural_passed"],
            "strict_release_passed": validation["strict_release_passed"],
        },
        "source": {
            "total_duration_s": sum(float(r["duration_s"]) for r in rows),
            "duration_s_quantiles": q([r["duration_s"] for r in rows]),
        },
        "aligned_sequences": {
            "phrases": sum(int(r["phrases"]) for r in aligned),
            "phrases_per_aligned_sample": q([r["phrases"] for r in aligned]),
            "phrase_time_coverage": q([r["phrase_time_coverage"] for r in aligned]),
            "word_records": len(words),
            "instant_word_records": sum(w["instant"] for w in words),
            "probability_scored_word_records": len(scored_words),
            "low_probability_word_records": sum(w["probability"] < .2 for w in scored_words),
            "samples_with_instant_words": sum(any(w["instant"] for w in d["words"]) for d in details),
            "samples_with_low_probability_words": sum(
                any(w.get("probability") is not None and w["probability"] < .2
                    for w in d["words"]) for d in details),
            "human_reviewed_boundaries": validation["human_boundary_evaluation"]["reviewed_boundaries"],
        },
        "automatic_boundary_protocol": validation["automatic_boundary_evaluation"],
        "visual": {
            "sampled_frames": frames, "valid_face_frames": face_frames,
            "weighted_face_valid_rate": face_frames / frames,
            "sample_face_valid_rate_quantiles": q([d["face_valid_rate"] for d in details]),
            "zero_face_samples": [d["sample_id"] for d in details if d["face_valid_rate"] == 0],
            "crop_fallback_frames": sum(d.get("face_crop_fallback_frames", 0) for d in details),
            "crop_fallback_valid_frames": sum(d.get("face_crop_fallback_valid_frames", 0) for d in details),
            "samples_using_crop_fallback": sum(d.get("face_crop_fallback_frames", 0) > 0 for d in details),
            "geometry_rejected_frames": sum(d.get("face_geometry_rejected_frames", 0) for d in details),
            "samples_with_geometry_rejections": sum(
                d.get("face_geometry_rejected_frames", 0) > 0 for d in details),
            "ambiguous_face_frames": sum(d["ambiguous_face_frames"] for d in details),
            "multi_face_frames": sum(d["multi_face_frames"] for d in details),
        },
        "audio": {
            "digitally_silent_samples": [d["sample_id"] for d in details if d["digitally_silent_audio"]],
            "pitch_voiced_rate_quantiles": q([d["pitch_voiced_rate"] for d in details]),
            "pitch_boundary_hits": sum(d["pitch_boundary_hits"] for d in details),
        },
        "storage": {
            "feature_npz_bytes": sum((ROOT / r["feature_file"]).stat().st_size for r in rows),
            "alignment_jsonl_bytes": (ROOT / "outputs/alignment.jsonl").stat().st_size,
            "summary_csv_bytes": (ROOT / "outputs/q1_summary.csv").stat().st_size,
        },
        "latest_full_extraction_timing": {
            "sum_elapsed_s": sum(d["elapsed_s"] for d in details),
            "per_sample_elapsed_s": q([d["elapsed_s"] for d in details]),
            "stage_sum_s": {s: sum(d["stage_seconds"].get(s, 0) for d in details) for s in stages},
            "note": "Sum of per-sample feature stages only; excludes imports, initial audio preparation, corpus MFA, installation, downloads, audit and validation. See last_run.json for extract wall time.",
            "extract_run": json.loads((ROOT / "reports/last_run.json").read_text()),
        },
        "reaggregation_timing": {
            "samples_with_reaggregation_metadata": sum("reaggregation" in d for d in details),
            "sum_sample_seconds": sum(d.get("reaggregation", {}).get("seconds", 0) for d in details),
            "note": "Only current detail records produced by reaggregate are counted; historical runs remain in reaggregation.json.",
        },
        "validation": {
            "independent_reconstruction_max_abs_error": max(
                r["max_reconstruction_abs_error"] for r in validation["samples"]),
            "roadmap_all_100_phrase_aligned": validation["roadmap_all_100_phrase_aligned"],
            "scope_all_samples_sequence_mapped": validation["scope_all_samples_sequence_mapped"],
            "roadmap_human_review_complete": validation["roadmap_human_review_complete"],
        },
    }
    dump(ROOT / "reports/result_statistics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
