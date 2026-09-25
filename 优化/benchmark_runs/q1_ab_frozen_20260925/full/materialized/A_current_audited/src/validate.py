"""Validate final Q1 sequences, boundary evidence, and feature reconstruction."""
import csv
import copy
import json
from collections import Counter

import numpy as np

from acceptance import (
    POLICY_VERSION,
    boundary_rows,
    decide_sample,
    evaluate_reviews,
    merge_review_sheet,
    release_ready,
    validation_fingerprint,
    verify_digital_silence,
)
from alignment import assess_alignment, visual_observation_state
from automatic_alignment import add_automatic_evidence, attach_protocol_checks
from pipeline import json_hash, provenance
from prepare import ROOT, dump, sha256


def _quantiles(values):
    if not values:
        return {"median": None, "p90": None, "maximum": None}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(np.max(values)),
    }


def main(pilot=False, diagnostic=False, config=None):
    config = config or json.loads((ROOT / "configs/q1.json").read_text())
    provenance_base, _, input_hashes = provenance(config)
    manifest = {
        row["sample_id"]: row
        for row in json.loads((ROOT / "resources/manifest.json").read_text())
    }
    expected = (
        set(json.loads((ROOT / "configs/pilot.json").read_text()))
        if pilot else set(manifest)
    )
    with (ROOT / "outputs/q1_summary.csv").open(encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in summary]
    errors, records, boundaries, metadata = [], [], [], {}
    missing_audio_evidence = {}
    mapping = [
        json.loads(line)
        for line in (ROOT / "outputs/alignment.jsonl").read_text().splitlines()
    ]
    mapping_by_id = {row["sample_id"]: row for row in mapping}
    if len(mapping) != len(mapping_by_id) or set(mapping_by_id) != set(ids):
        errors.append("Mapping ID coverage differs from summary")
    if len(ids) != len(set(ids)):
        errors.append("Duplicate summary IDs")
    if not expected <= set(ids) or (not pilot and set(ids) != expected):
        errors.append(
            f"Coverage mismatch: missing={sorted(expected-set(ids))}, "
            f"extra={sorted(set(ids)-set(manifest))}")

    for sid in sorted(expected & set(ids)):
        try:
            source = manifest[sid]
            meta = json.loads((ROOT / "outputs/details" / (sid + ".json")).read_text())
            metadata[sid] = meta
            assert meta == mapping_by_id[sid], "JSONL mapping differs from detail"
            assert all(meta[key] == value for key, value in source.items()), (
                "Source metadata differs")
            source_sha = sha256(ROOT / source["source_file"])
            assert source_sha == input_hashes[source["source_file"]]
            expected_fingerprint = json_hash({
                **provenance_base, "source_sha256": source_sha})
            assert meta["artifact_fingerprint"] == expected_fingerprint
            assert meta["schema_version"] == config["schema_version"]

            probe = json.loads(
                (ROOT / "intermediate/probes" / (sid + ".json")).read_text())
            raw_pts = np.array([
                float(frame["best_effort_timestamp_time"])
                for frame in probe["frames"]
            ])
            with np.load(ROOT / meta["feature_file"], allow_pickle=False) as data:
                features = data["features"]
                valid = data["valid"]
                intervals = data["intervals"]
                char_ranges = data["char_ranges"]
                phrases = meta["phrases"]
                assert str(data["sample_id"]) == sid
                assert float(data["duration_s"]) == source["duration_s"]
                assert len(phrases) >= 1, "Every sample must have a final sequence item"
                assert features.shape == valid.shape == (len(phrases), 150)
                assert features.dtype == np.float32 and valid.dtype == bool
                assert np.isfinite(features).all() and np.all(features[~valid] == 0)
                assert bool(data["alignment_available"])
                assert bool(data["alignment_quality_passed"]) == (
                    meta["alignment_quality"]["hard_passed"])
                assert bool(data["speech_alignment_available"]) == (
                    meta["speech_alignment_available"])
                assert bool(data["audio_observed"]) == meta["audio_observed"]
                assert np.all(intervals[:, 0] >= 0)
                assert np.all(intervals[:, 1] <= source["duration_s"])
                assert np.all(intervals[:, 1] > intervals[:, 0])
                assert np.all(intervals[1:, 0] >= intervals[:-1, 1])
                assert np.array_equal(
                    data["sampled_frame_pts"],
                    raw_pts[data["sampled_frame_indices"]])

                context_values = [bool(p.get("context_only", False)) for p in phrases]
                assert all(context_values) or not any(context_values), (
                    "Cannot mix phrase and clip-context granularity")
                context_only = all(context_values)
                assert str(data["granularity"]) == (
                    "clip_context" if context_only else "phrase")
                assert meta["speech_alignment_available"] == (not context_only)
                audio_missing = any(p.get("audio_missing", False) for p in phrases)
                assert audio_missing == all(
                    p.get("audio_missing", False) for p in phrases)
                assert meta["audio_observed"] == (not audio_missing)
                assert meta["digitally_silent_audio"] == audio_missing

                recomputed_quality = assess_alignment(
                    meta["words"], phrases, config["alignment_quality"])
                assert recomputed_quality == meta["alignment_quality"]
                recomputed_automatic_quality = assess_alignment(
                    meta["automatic_words"], meta["automatic_phrases"],
                    config["alignment_quality"])
                assert recomputed_automatic_quality == meta["automatic_alignment_quality"]

                evidence = meta["automatic_boundary_evidence"]
                assert evidence["policy"] == config["automatic_alignment"]["policy"]
                manual = any(p.get("human_annotated", False) for p in phrases)
                expected_evidence = copy.deepcopy(evidence)
                attach_protocol_checks(expected_evidence, phrases,
                                       bool(phrases and recomputed_quality["hard_passed"]), manual)
                assert evidence["protocol_checks"] == expected_evidence["protocol_checks"]
                assert evidence["protocol_passed"] == expected_evidence["protocol_passed"]

                if context_only:
                    assert len(phrases) == 1
                    phrase = phrases[0]
                    assert phrase["start_s"] == 0 and phrase["end_s"] == source["duration_s"]
                    assert phrase["char_start"] == 0
                    assert phrase["char_end"] == len(source["text"])
                    assert phrase["text"] == source["text"]
                    assert evidence["endpoint_count"] == 0
                    if audio_missing:
                        assert meta["alignment_status"] == "aligned_clip_context_audio_missing"
                        assert not valid[:, 128:144].any()
                        assert np.all(features[:, 128:144] == 0)
                        import soundfile as sf
                        waveform, _ = sf.read(
                            ROOT / "intermediate/audio" / (sid + ".wav"))
                        assert len(waveform) > 0 and not np.any(waveform != 0)
                        proof = verify_digital_silence(
                            ROOT / source["source_file"], source_sha, sha256)
                        assert proof["verified"], (
                            "Original audio channels are not digital silence")
                        missing_audio_evidence[sid] = proof
                    else:
                        assert meta["alignment_status"] == (
                            "aligned_clip_context_speech_unavailable")
                        assert not meta["digitally_silent_audio"]
                else:
                    assert meta["alignment_status"] in {
                        "aligned_auto_protocol_passed",
                        "aligned_manual_pending_approval",
                    }
                    if manual:
                        assert evidence["endpoint_count"] == 0
                    else:
                        rebuilt_phrases = copy.deepcopy(phrases)
                        rebuilt_evidence = add_automatic_evidence(
                            rebuilt_phrases, meta["secondary_alignment_words"],
                            evidence["vad_intervals_s"], config)
                        assert all(evidence[k] == v for k, v in rebuilt_evidence.items())
                        for phrase, rebuilt_phrase in zip(phrases, rebuilt_phrases):
                            automatic = phrase["automatic_boundary_evidence"]
                            assert automatic == rebuilt_phrase["automatic_boundary_evidence"]
                            assert automatic["mfa_s"] == [
                                phrase["start_s"], phrase["end_s"]]
                            if automatic["secondary_status"] == "available":
                                assert automatic["stable_ts_s"][0] < automatic["stable_ts_s"][1]
                                assert len(automatic["absolute_difference_s"]) == 2
                            else:
                                assert automatic["absolute_difference_s"] == []
                            assert 0 <= automatic["vad_overlap_fraction"] <= 1 + 1e-9

                chars_covered = np.zeros(len(source["text"]), dtype=bool)
                token_indices = []
                for phrase, interval, chars in zip(
                        phrases, intervals, char_ranges):
                    assert np.allclose(
                        interval, [phrase["start_s"], phrase["end_s"]],
                        atol=1e-9, rtol=0)
                    assert chars.tolist() == [
                        phrase["char_start"], phrase["char_end"]]
                    assert source["text"][chars[0]:chars[1]] == phrase["text"]
                    assert not chars_covered[chars[0]:chars[1]].any()
                    chars_covered[chars[0]:chars[1]] = True
                    token_indices += phrase["token_indices"]
                    frame_pts = raw_pts[phrase["frame_indices"]]
                    assert np.array_equal(frame_pts, phrase["frame_pts_s"])
                    assert np.all(frame_pts - source["origin_s"] >= interval[0])
                    assert np.all(frame_pts - source["origin_s"] < interval[1])
                assert all(
                    chars_covered[index] or char.isspace()
                    for index, char in enumerate(source["text"]))
                assert sorted(token_indices) == list(range(len(data["text_token_ids"])))

                cache = ROOT / "intermediate/samples" / sid
                with (
                    np.load(cache / "text.npz") as text,
                    np.load(cache / "audio.npz") as audio,
                    np.load(cache / "vision.npz") as vision,
                ):
                    np.testing.assert_array_equal(data["text_token_ids"], text["ids"])
                    np.testing.assert_array_equal(
                        data["text_token_offsets"], text["offsets"])
                    np.testing.assert_array_equal(
                        data["sampled_frame_indices"], vision["frame_index"])
                    np.testing.assert_array_equal(
                        data["sampled_frame_pts"], vision["pts"])
                    largest_error = 0.0
                    for index, phrase in enumerate(phrases):
                        start, end = intervals[index]
                        rebuilt = np.zeros(150)
                        rebuilt_valid = np.zeros(150, bool)
                        selected_tokens = phrase["token_indices"]
                        if selected_tokens:
                            rebuilt[:128] = np.mean(
                                text["embeddings"][selected_tokens], axis=0)
                            rebuilt_valid[:128] = True
                        if phrase.get("audio_missing", False):
                            assert phrase["audio_window_indices"] == []
                            assert phrase["pitch_window_indices"] == []
                        else:
                            for dimension in range(16):
                                pitch = dimension == 15
                                native_intervals = (
                                    audio["f0_intervals"] if pitch
                                    else audio["intervals"])
                                values = (
                                    audio["f0"] if pitch
                                    else audio["values"][:, dimension])
                                observed = (
                                    audio["f0_valid"] if pitch
                                    else np.isfinite(values))
                                weights = np.clip(
                                    np.minimum(native_intervals[:, 1], end)
                                    - np.maximum(native_intervals[:, 0], start),
                                    0, None)
                                active = observed & (weights > 0)
                                if active.any():
                                    rebuilt[128 + dimension] = np.average(
                                        values[active], weights=weights[active])
                                    rebuilt_valid[128 + dimension] = True
                        frame_active = (
                            (vision["time"] >= start) & (vision["time"] < end))
                        assert phrase["visual_observation_state"] == (
                            visual_observation_state(
                                vision, np.flatnonzero(frame_active)))
                        for dimension in range(6):
                            active = frame_active & vision["valid"][:, dimension]
                            if active.any():
                                rebuilt[144 + dimension] = np.mean(
                                    vision["values"][active, dimension],
                                    dtype=np.float64)
                                rebuilt_valid[144 + dimension] = True
                        assert np.array_equal(rebuilt_valid, valid[index]), (
                            "Rebuilt mask differs")
                        np.testing.assert_allclose(
                            rebuilt, features[index], atol=3e-5, rtol=2e-6)
                        largest_error = max(
                            largest_error,
                            float(np.max(np.abs(rebuilt - features[index]))))

                records.append({
                    "sample_id": sid,
                    "passed": True,
                    "phrases": len(features),
                    "alignment_available": True,
                    "speech_alignment_available": meta["speech_alignment_available"],
                    "audio_observed": meta["audio_observed"],
                    "granularity": "clip_context" if context_only else "phrase",
                    "alignment_quality_passed": meta["alignment_quality"]["hard_passed"],
                    "automatic_protocol_passed": evidence["protocol_passed"],
                    "max_reconstruction_abs_error": largest_error,
                })
        except Exception as exc:
            errors.append(f"{sid}: {type(exc).__name__}: {exc}")

    # The review table remains an optional audit/correction interface. Only an
    # imported manual correction requires explicit approval before publication.
    for path in sorted((ROOT / "outputs/details").glob("*.json")):
        meta = json.loads(path.read_text())
        boundaries.extend(boundary_rows(
            meta, sha256(ROOT / meta["feature_file"])))
    reviews, stale_reviews = merge_review_sheet(
        ROOT / "annotations/boundary_review.csv", boundaries)
    if pilot:
        boundaries = [
            row for row in boundaries if row["sample_id"] in expected]
        reviews = evaluate_reviews(boundaries)
        with (
            ROOT / "annotations/pilot_boundary_review.csv"
        ).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(boundaries[0]))
            writer.writeheader()
            writer.writerows(boundaries)

    required_review = {
        sid for sid, meta in metadata.items()
        if any(p.get("human_annotated", False) for p in meta["phrases"])
    }
    review_complete = {
        sid: reviews["samples"].get(sid, {}).get("approved", False)
        for sid in expected
    }
    valid_ids = {record["sample_id"] for record in records}
    decisions = []
    for sid, meta in sorted(metadata.items()):
        decision = decide_sample(
            meta,
            sid in valid_ids,
            reviews["samples"].get(sid, {}),
            sid in required_review,
        )
        decision.update({
            "feature_sha256": sha256(ROOT / meta["feature_file"]),
            "audio_missing_evidence": missing_audio_evidence.get(sid),
        })
        decisions.append(decision)

    structural_passed = not errors and len(records) == len(expected)
    human_review_complete = all(
        review_complete[sid] for sid in required_review)
    strict_release_passed = release_ready(
        decisions, len(expected), structural_passed)
    review_errors = [
        row["absolute_error_s"] for row in reviews["independent"]]
    automatic_reference_differences = [
        row["automatic_abs_difference_s"]
        for row in reviews["independent"]
        if "automatic_abs_difference_s" in row
    ]
    cross_aligner_differences = [
        difference
        for meta in metadata.values()
        for phrase in meta["phrases"]
        for difference in phrase.get(
            "automatic_boundary_evidence", {}).get(
                "absolute_difference_s", [])
    ]

    if not pilot:
        with (ROOT / "outputs/acceptance.jsonl").open(
                "w", encoding="utf-8") as handle:
            for decision in decisions:
                handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
        advisory_queue = []
        for sid, meta in sorted(metadata.items()):
            evidence = meta["automatic_boundary_evidence"]
            flags = sorted({
                flag for phrase in meta["phrases"] for flag in phrase["flags"]
            })
            if (
                evidence["confidence_grade"] == "C"
                or "automatic_quality_merge" in flags
                or not meta["speech_alignment_available"]
            ):
                advisory_queue.append({
                    "sample_id": sid,
                    "status": "optional_audit",
                    "release_blocking": False,
                    "confidence_grade": evidence["confidence_grade"],
                    "alignment_hard_issues": (
                        meta["alignment_quality"]["hard_issues"]),
                    "flags": flags,
                })
        dump(ROOT / "reports/review_queue.json", advisory_queue)
        dump(ROOT / "reports/source_exceptions.json", [])
        dump(
            ROOT / "reports/audio_missing_evidence.json",
            [{"sample_id": sid, **proof}
             for sid, proof in sorted(missing_audio_evidence.items())],
        )

    flags = Counter(row["flags"] for row in summary)
    evidence_rows = [
        meta["automatic_boundary_evidence"] for meta in metadata.values()]
    context_records = [
        record for record in records if record["granularity"] == "clip_context"]
    report = {
        "scope": "pilot" if pilot else "all_100",
        "expected_samples": len(expected),
        "acceptance_policy": POLICY_VERSION,
        "provenance_base_sha256": json_hash(provenance_base),
        "validation_fingerprint": validation_fingerprint(ROOT, sha256),
        "checked_samples": len(records),
        "passed": strict_release_passed,
        "structural_passed": structural_passed,
        "strict_release_passed": strict_release_passed,
        "errors": errors,
        "passed_meaning": (
            "Every sample has a validated final sequence and is accepted by "
            "the deterministic boundary or clip-context protocol."),
        "mapped_samples": sum(
            record["alignment_available"] for record in records),
        "speech_aligned_samples": sum(
            record["speech_alignment_available"] for record in records),
        "clip_context_samples": len(context_records),
        "audio_missing_samples": sum(
            not record["audio_observed"] for record in records),
        "automatic_protocol_passed_samples": sum(
            decision["automatic_protocol_passed"] for decision in decisions),
        "automatic_quality_passed_samples": sum(
            decision["automatic_quality_passed"] for decision in decisions),
        "quality_failed_samples": [
            record["sample_id"] for record in records
            if not record["alignment_quality_passed"]
        ],
        "strictly_accepted_samples": sum(
            decision["aligned_accepted"] for decision in decisions),
        "automatically_accepted_samples": sum(
            decision["status"] in {
                "aligned_auto_accepted",
                "clip_context_speech_unavailable_accepted",
                "clip_context_audio_missing_accepted",
            }
            for decision in decisions
        ),
        "verified_source_exception_samples": sum(
            decision["source_exception_verified"] for decision in decisions),
        "release_eligible_samples": sum(
            decision["release_eligible"] for decision in decisions),
        "acceptance_status_counts": dict(Counter(
            decision["status"] for decision in decisions)),
        "unalignable_samples": [],
        "speech_alignment_unavailable_samples": [
            record["sample_id"] for record in records
            if not record["speech_alignment_available"]
        ],
        "roadmap_all_100_phrase_aligned": (
            None if pilot else len(records) == 100 and all(
                record["speech_alignment_available"] for record in records)
        ),
        "scope_all_samples_sequence_mapped": (
            len(records) == len(expected)
            and all(record["alignment_available"] for record in records)
        ),
        "scope_all_samples_phrase_aligned": (
            len(records) == len(expected)
            and all(record["speech_alignment_available"] for record in records)
        ),
        "roadmap_human_review_complete": human_review_complete,
        "phrase_count": sum(record["phrases"] for record in records),
        "samples": records,
        "acceptance": decisions,
        "flag_combinations": dict(flags),
        "checks": [
            "ID coverage",
            "at least one final sequence item per sample",
            "150 dimensions and mask semantics",
            "finite values",
            "ordered true time ranges",
            "non-whitespace transcript coverage",
            "every token assigned once",
            "frame index to exact ffprobe PTS",
            "independent native-rate feature aggregation",
            "automatic boundary protocol",
            "digital-silence audio-missing evidence",
        ],
        "automatic_boundary_evaluation": {
            "policy": config["automatic_alignment"]["policy"],
            "protocol_passed_samples": sum(
                evidence["protocol_passed"] for evidence in evidence_rows),
            "primary_source_counts": dict(Counter(
                evidence["primary"] for evidence in evidence_rows)),
            "confidence_grade_counts": dict(Counter(
                evidence["confidence_grade"] for evidence in evidence_rows)),
            "cross_aligner_endpoint_count": len(cross_aligner_differences),
            "secondary_unavailable_phrases": sum(
                len(evidence.get("secondary_unavailable_phrase_indices", []))
                for evidence in evidence_rows),
            "secondary_incomplete_samples": sum(
                bool(evidence.get("secondary_unavailable_phrase_indices", []))
                for evidence in evidence_rows),
            "cross_aligner_absolute_difference_s": _quantiles(
                cross_aligner_differences),
            "samples_with_vad_speech": sum(
                bool(evidence["vad_intervals_s"]) for evidence in evidence_rows),
            "mfa_word_aligned_samples": sum(
                meta["speech_alignment_available"]
                for meta in metadata.values()),
            "clip_context_speech_unavailable_samples": sum(
                not meta["speech_alignment_available"] and meta["audio_observed"]
                for meta in metadata.values()),
            "clip_context_audio_missing_samples": sum(
                not meta["audio_observed"] for meta in metadata.values()),
            "metric_meaning": (
                "MFA is the primary boundary. Stable-ts endpoint differences "
                "and Silero VAD are cross-model consistency evidence, not "
                "errors against human ground truth."),
            "human_ground_truth_accuracy": (
                "unavailable" if not review_errors
                else "reported in human_boundary_evaluation"),
        },
        "human_boundary_evaluation": {
            "policy_role": (
                "Optional audit for automatic outputs; required only after "
                "manual boundary correction."),
            "required_samples": len(required_review),
            "fully_reviewed_required_samples": sum(
                review_complete[sid] for sid in required_review),
            "fully_checked_required_samples": sum(
                reviews["samples"].get(sid, {}).get("checked", False)
                for sid in required_review),
            "reviewed_boundaries": reviews["checked_boundaries"],
            "approved_boundaries": reviews["approved_boundaries"],
            "invalid_reference_rows": reviews["invalid_rows"],
            "independent_evaluation_boundaries": len(review_errors),
            "correction_reference_boundaries": len(reviews["corrections"]),
            "median_abs_error_s": (
                float(np.median(review_errors)) if review_errors else None),
            "p90_abs_error_s": (
                float(np.quantile(review_errors, .9))
                if review_errors else None),
            "automatic_median_abs_error_s": (
                float(np.median(automatic_reference_differences))
                if automatic_reference_differences else None),
            "automatic_p90_abs_error_s": (
                float(np.quantile(automatic_reference_differences, .9))
                if automatic_reference_differences else None),
            "correction_median_current_difference_s": (
                float(np.median([
                    row["absolute_error_s"] for row in reviews["corrections"]
                ]))
                if reviews["corrections"] else None),
            "stale_review_rows_invalidated": stale_reviews,
            "error_metric_meaning": (
                "Only explicitly independent references; corrections are "
                "reported separately."),
            "status": (
                "not_required" if not required_review
                else "complete" if human_review_complete
                else "manual correction approval incomplete"),
        },
    }
    output = ROOT / "reports" / (
        "pilot_validation.json" if pilot else "validation.json")
    dump(output, report)
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("samples", "flag_combinations", "acceptance")
    }, ensure_ascii=False, indent=2))
    if not structural_passed:
        raise ValueError(
            f"{len(errors)} validation failures; see {output.name}")
    if not diagnostic and not strict_release_passed:
        raise RuntimeError(
            "Strict Q1 release criteria are not met; rerun with "
            "--diagnostic to inspect known gaps")


if __name__ == "__main__":
    main()
