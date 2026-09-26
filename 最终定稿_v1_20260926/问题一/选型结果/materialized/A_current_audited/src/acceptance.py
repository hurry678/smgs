"""One release policy for validation and packaging; review never creates references."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict

POLICY_VERSION = "q1-acceptance-4"
EDITABLE = ("reference_s", "reviewer", "review_status", "reference_source",
            "reference_id", "decision_note")
STATES = {"pending", "reviewed", "approved", "needs_revision"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def boundary_rows(meta, feature_sha256):
    automatic = {(p["char_start"], p["char_end"]): p
                 for p in meta.get("automatic_phrases", meta["phrases"])}
    identity = digest({
        "policy": POLICY_VERSION, "sample_id": meta["sample_id"], "text": meta["text"],
        "duration_s": meta["duration_s"], "source_sha256": meta["fingerprint_components"]["source_sha256"],
        "feature_sha256": feature_sha256, "quality": meta["alignment_quality"],
        "phrases": [{k: p.get(k) for k in ("start_s", "end_s", "char_start", "char_end",
                                          "text", "alignment_source", "flags")}
                    for p in meta["phrases"]],
    })
    rows = []
    for p in meta["phrases"]:
        original = automatic.get((p["char_start"], p["char_end"]), {})
        for endpoint in ("start_s", "end_s"):
            rows.append({
                "sample_id": meta["sample_id"], "phrase_index": p["phrase_index"],
                "endpoint": endpoint, "current_s": p[endpoint],
                "automatic_s": original.get(endpoint, ""), "duration_s": meta["duration_s"],
                "char_start": p["char_start"], "char_end": p["char_end"], "text": p["text"],
                "alignment_source": p["alignment_source"], "annotator": p.get("reviewer", ""),
                "review_fingerprint": identity, "reference_s": "", "reviewer": "",
                "review_status": "pending", "reference_source": "", "reference_id": "",
                "decision_note": "", "flags": ";".join(p["flags"]),
                "validation_state": "pending", "validation_error": "",
            })
    return rows


def key(row):
    return row["sample_id"], str(row["phrase_index"]), row["endpoint"]


def evaluate_reviews(rows):
    """Check references as complete ordered intervals; approved must accept current output."""
    by_sample = defaultdict(list)
    by_phrase = defaultdict(list)
    errors = defaultdict(list)
    values = {}
    counts = Counter(key(r) for r in rows)
    for i, row in enumerate(rows):
        by_sample[row["sample_id"]].append(i)
        by_phrase[(row["sample_id"], str(row["phrase_index"]))].append(i)
        state = row["review_status"].strip().lower()
        entered = bool(row["reference_s"].strip() or row["reviewer"].strip() or state != "pending")
        if not entered:
            continue
        if counts[key(row)] != 1:
            errors[i].append("duplicate_boundary")
        if state not in STATES:
            errors[i].append("stale_or_invalid_status")
        if not row["reviewer"].strip():
            errors[i].append("missing_reviewer")
        if row["reference_source"] not in {"correction", "independent"}:
            errors[i].append("missing_reference_source")
        if row["reference_source"] == "independent":
            if not row["reference_id"].strip():
                errors[i].append("missing_independent_reference_id")
            if row.get("annotator") and row["annotator"].strip() == row["reviewer"].strip():
                errors[i].append("annotator_cannot_independently_evaluate_own_correction")
        try:
            reference = float(row["reference_s"])
            if not math.isfinite(reference) or not 0 <= reference <= float(row["duration_s"]):
                raise ValueError
            values[i] = reference
        except (ValueError, TypeError):
            errors[i].append("reference_not_finite_or_out_of_range")
    intervals = defaultdict(list)
    for (sid, phrase_index), indices in by_phrase.items():
        endpoints = {rows[i]["endpoint"]: i for i in indices}
        if not any(i in values or i in errors for i in indices):
            continue
        if len(indices) != 2 or set(endpoints) != {"start_s", "end_s"}:
            for i in indices:
                errors[i].append("incomplete_or_duplicate_interval")
            continue
        start, end = endpoints["start_s"], endpoints["end_s"]
        if start not in values or end not in values:
            for i in indices:
                errors[i].append("incomplete_reference_interval")
        elif values[start] >= values[end]:
            for i in indices:
                errors[i].append("reference_interval_not_positive")
        else:
            intervals[sid].append((int(phrase_index), values[start], values[end], indices))
    for group in intervals.values():
        ordered = sorted(group)
        for previous, current in zip(ordered, ordered[1:]):
            if previous[2] > current[1]:
                for i in previous[3] + current[3]:
                    errors[i].append("reference_intervals_overlap_or_reversed")
    independent, corrections = [], []
    for i, row in enumerate(rows):
        state = row["review_status"].strip().lower()
        row["validation_error"] = ";".join(sorted(set(errors.get(i, []))))
        if errors.get(i):
            row["validation_state"] = "invalid"
        elif i not in values or state == "pending":
            row["validation_state"] = "pending"
        else:
            difference = abs(values[i] - float(row["current_s"]))
            row["validation_state"] = (
                "needs_revision" if state == "needs_revision" or difference > 1e-9 else state)
            item = {**row, "absolute_error_s": difference}
            if row["automatic_s"] != "":
                item["automatic_abs_difference_s"] = abs(values[i] - float(row["automatic_s"]))
            (independent if row["reference_source"] == "independent" else corrections).append(item)
    samples = {}
    for sid, indices in by_sample.items():
        states = [rows[i]["validation_state"] for i in indices]
        approved = bool(states) and all(s == "approved" for s in states)
        samples[sid] = {
            "approved": approved,
            "checked": bool(states) and all(s in {"reviewed", "approved", "needs_revision"} for s in states),
            "needs_revision": "needs_revision" in states,
            "invalid": "invalid" in states,
            "has_review_activity": any(rows[i]["reference_s"].strip() or rows[i]["reviewer"].strip()
                                       or rows[i]["review_status"] != "pending" for i in indices),
            "override_documented": approved and all(rows[i]["decision_note"].strip() for i in indices),
        }
    return {"samples": samples, "independent": independent, "corrections": corrections,
            "invalid_rows": sum(r["validation_state"] == "invalid" for r in rows),
            "approved_boundaries": sum(r["validation_state"] == "approved" for r in rows),
            "checked_boundaries": len(independent) + len(corrections)}


def merge_review_sheet(path, boundaries):
    old = defaultdict(list)
    if path.exists():
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                old[key(row)].append(row)
    stale, retired = 0, []
    for row in boundaries:
        previous_rows = old.pop(key(row), [])
        if not previous_rows:
            continue
        previous = previous_rows[0]
        for field in EDITABLE:
            row[field] = previous.get(field, "")
        row["review_status"] = row["review_status"] or "pending"
        activity = any(previous.get(f, "").strip() for f in ("reference_s", "reviewer")) or (
            previous.get("review_status", "pending") != "pending")
        if len(previous_rows) > 1:
            row["review_status"] = "invalid_duplicate"
            retired.extend(previous_rows)
        elif activity and previous.get("review_fingerprint") != row["review_fingerprint"]:
            row["review_status"] = "stale"
            stale += 1
            retired.append(previous)
    retired.extend(r for group in old.values() for r in group)
    if retired:
        with path.with_suffix(".history.jsonl").open("a", encoding="utf-8") as handle:
            for row in retired:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = evaluate_reviews(boundaries)
    if boundaries:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(boundaries[0]))
            writer.writeheader()
            writer.writerows(boundaries)
    return result, stale


def decide_sample(meta, structural_ok, review, required, exception=None):
    aligned = bool(meta["phrases"])
    manual = any(p.get("human_annotated", False) for p in meta["phrases"])
    protocol = meta.get("automatic_boundary_evidence", {}).get(
        "protocol_passed", meta["alignment_quality"]["hard_passed"])
    human = bool(review.get("approved")) and (
        meta["alignment_quality"]["hard_passed"] or review.get("override_documented", False))
    automatic = (aligned and not manual and not required and protocol
                 and not review.get("has_review_activity", False))
    accepted = structural_ok and aligned and (human or automatic)
    source_exception = bool(structural_ok and not aligned and exception and exception["verified"])
    if accepted:
        if meta.get("speech_alignment_available", True):
            status = "aligned_human_approved" if human else "aligned_auto_accepted"
        elif meta.get("audio_observed", True):
            status = "clip_context_speech_unavailable_accepted"
        else:
            status = "clip_context_audio_missing_accepted"
    elif source_exception:
        status = "verified_source_exception"
    elif not structural_ok:
        status = "structural_failure"
    elif review.get("needs_revision"):
        status = "aligned_needs_revision"
    elif any(p.get("human_annotated") for p in meta["phrases"]):
        status = "manual_reaggregated_awaiting_approval"
    else:
        status = "aligned_awaiting_review" if aligned else "unverified_unaligned"
    return {
        "sample_id": meta["sample_id"], "status": status, "structural_passed": structural_ok,
        "alignment_available": aligned, "aligned_accepted": accepted,
        "source_exception_verified": source_exception, "release_eligible": accepted or source_exception,
        "human_approved": accepted and human, "review_required": required,
        "automatic_protocol_passed": protocol,
        "automatic_quality_passed": meta.get("automatic_alignment_quality", meta["alignment_quality"])["hard_passed"],
        "current_quality_passed": meta["alignment_quality"]["hard_passed"],
    }


def release_ready(decisions, expected_count, structural_passed):
    return (structural_passed and len(decisions) == expected_count
            and len({d["sample_id"] for d in decisions}) == expected_count
            and all(d["release_eligible"] for d in decisions))


def validation_fingerprint(root, file_hash):
    """Bind approvals to actual artifacts, native evidence, reviews, inputs and policy."""
    paths = []
    for directory, pattern in (
        ("src", "*.py"), ("configs", "*.json"), ("outputs/features", "*.npz"),
        ("outputs/details", "*.json"), ("annotations", "*.csv"), ("annotations", "*.jsonl"),
        ("intermediate/samples", "*/*.npz"), ("intermediate/samples", "*/native_provenance.json"),
        ("intermediate/probes", "*.json"),
    ):
        paths.extend((root / directory).glob(pattern))
    paths.extend(root / name for name in (
        "run.py", "requirements.lock.txt", "resources/manifest.json",
        "resources/input_manifest.json", "resources/model_sources.json",
        "resources/resource_lock.json", "intermediate/mfa/run.json",
        "reports/last_run.json", "reports/boundary_audit.json",
        "outputs/alignment.jsonl", "outputs/q1_summary.csv", "outputs/acceptance.jsonl"))
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(set(paths)) if p.is_file()})


def verify_digital_silence(path, expected_sha256, file_hash):
    """Decode original channels, so cancellation in a mono downmix cannot prove silence."""
    import av
    import numpy as np
    if file_hash(path) != expected_sha256:
        raise ValueError("Source changed while verifying digital silence")
    samples, nonzero, channels = 0, 0, set()
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            data = frame.to_ndarray()
            samples += int(data.size)
            nonzero += int(np.count_nonzero(data))
            channels.add(len(frame.layout.channels))
    return {"verified": samples > 0 and nonzero == 0, "type": "original_digital_silence",
            "source_sha256": expected_sha256, "decoded_scalar_samples": samples,
            "nonzero_scalar_samples": nonzero, "channel_counts": sorted(channels),
            "decoder": f"PyAV {av.__version__}", "native_features_preserved": True}
