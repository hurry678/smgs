"""Given-transcript alignment, lossless character mapping, and interval aggregation."""
import numpy as np


def map_words(text, result, offset, duration):
    """Map exact non-whitespace characters; never edit the provided transcript."""
    original = [(i, ch) for i, ch in enumerate(text) if not ch.isspace()]
    cursor, words = 0, []
    for segment in result["segments"]:
        for word in segment["words"]:
            letters = [ch for ch in word["word"] if not ch.isspace()]
            if not letters:
                continue
            end = cursor + len(letters)
            if [ch for _, ch in original[cursor:end]] != letters:
                raise ValueError(f"Forced alignment changed text at character {cursor}: {word['word']}")
            a, b = float(word["start"]) + offset, float(word["end"]) + offset
            if a < -1e-6 or b > duration + 0.05 or b < a:
                raise ValueError(f"Invalid word time {a}, {b}, duration {duration}")
            # Only trim sub-50-ms container/audio-end rounding, and record any adjustment.
            a2, b2 = max(0.0, a), min(duration, b)
            words.append({
                "word_index": len(words), "text": text[original[cursor][0]:original[end - 1][0] + 1],
                "char_start": original[cursor][0], "char_end": original[end - 1][0] + 1,
                "start_s": a2, "end_s": b2, "raw_start_s": a, "raw_end_s": b,
                "probability": float(word.get("probability", 0)),
                "instant": b2 <= a2, "end_clipped": b2 != b,
            })
            cursor = end
    if cursor != len(original):
        raise ValueError(f"Unaligned transcript tail: {cursor}/{len(original)} characters")
    return words


def _group_quality_passes(group, words, quality):
    duration = words[group[-1]]["end_s"] - words[group[0]]["start_s"]
    if duration <= 0:
        return False
    instant_fraction = sum(words[i]["instant"] for i in group) / len(group)
    return (
        duration >= quality["minimum_phrase_duration_s"]
        and len(group) / duration <= quality["maximum_words_per_second"]
        and instant_fraction <= quality["maximum_instant_word_fraction"]
    )


def _merge_invalid_groups(groups, words, quality):
    """Merge numerically unstable tiny groups into their nearest time neighbor."""
    regularized = set()
    while len(groups) > 1:
        invalid = next(
            (i for i, group in enumerate(groups)
             if not _group_quality_passes(group, words, quality)),
            None,
        )
        if invalid is None:
            break
        if invalid == 0:
            neighbor = 1
        elif invalid == len(groups) - 1:
            neighbor = invalid - 1
        else:
            left_gap = max(
                0.0, words[groups[invalid][0]]["start_s"]
                - words[groups[invalid - 1][-1]]["end_s"])
            right_gap = max(
                0.0, words[groups[invalid + 1][0]]["start_s"]
                - words[groups[invalid][-1]]["end_s"])
            neighbor = invalid - 1 if left_gap <= right_gap else invalid + 1
        start, end = sorted((invalid, neighbor))
        merged = groups[start] + groups[end]
        regularized.update(merged)
        groups[start:end + 1] = [merged]
    return groups, regularized


def phrase_groups(words, text, c, quality=None):
    groups, current = [], []
    for i, w in enumerate(words):
        current.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        elapsed = w["end_s"] - words[current[0]]["start_s"]
        split = (nxt is None or w["text"].rstrip().endswith((".", ",", ";", ":", "!", "?"))
                 or nxt["start_s"] - w["end_s"] >= c["gap_s"]
                 or len(current) >= c["max_phrase_words"] or elapsed >= c["max_phrase_s"])
        if split:
            groups.append(current)
            current = []
    # Instantaneous words have no defensible separate interval. Merge adjacent groups
    # using their observed endpoint union; retain an uncertainty flag and original word times.
    i = 0
    while i < len(groups):
        g = groups[i]
        if words[g[-1]]["end_s"] <= words[g[0]]["start_s"]:
            if len(groups) == 1:
                raise ValueError("All words have zero-duration alignment; manual timing required")
            if i + 1 < len(groups):
                groups[i + 1] = g + groups[i + 1]
                groups.pop(i)
            else:
                groups[i - 1] += g
                groups.pop(i)
            continue
        i += 1
    regularized = set()
    if quality is not None:
        groups, regularized = _merge_invalid_groups(groups, words, quality)
    phrases = []
    for i, g in enumerate(groups):
        ws = [words[k] for k in g]
        ca, cb = ws[0]["char_start"], ws[-1]["char_end"]
        a, b = ws[0]["start_s"], ws[-1]["end_s"]
        flags = []
        if any(w["instant"] for w in ws):
            flags.append("instant_word_merged")
        if any(w.get("probability") is not None
               and w["probability"] < c["low_word_probability"] for w in ws):
            flags.append("low_alignment_probability")
        if any(w["end_clipped"] for w in ws):
            flags.append("container_end_clipped")
        if any(index in regularized for index in g):
            flags.append("automatic_quality_merge")
        if i and a < phrases[-1]["end_s"] - 1e-6:
            raise ValueError("Overlapping phrase times require review")
        phrases.append({"phrase_index": i, "start_s": a, "end_s": b, "char_start": ca,
                        "char_end": cb, "text": text[ca:cb], "word_indices": g,
                        "alignment_source": ws[0].get(
                            "alignment_source", "stable-ts/tiny.en"), "flags": flags,
                        "human_reviewed": False})
    return phrases


def assess_alignment(words, phrases, quality):
    phrase_checks = []
    hard_issues = []
    for p in phrases:
        selected = [words[i] for i in p["word_indices"]]
        duration = p["end_s"] - p["start_s"]
        no_word_timing = p.get("human_annotated", False) or p.get("context_only", False)
        count = len(p["text"].split()) if no_word_timing else len(selected)
        instant_fraction = (None if no_word_timing else
                            sum(w["instant"] for w in selected) / count if count else 0.0)
        words_per_second = count / duration if duration > 0 else float("inf")
        issues = []
        if duration < quality["minimum_phrase_duration_s"]:
            issues.append("phrase_too_short")
        if words_per_second > quality["maximum_words_per_second"]:
            issues.append("implausible_word_rate")
        if instant_fraction is not None and instant_fraction > quality["maximum_instant_word_fraction"]:
            issues.append("excess_instant_words")
        hard_issues.extend(f"p{p['phrase_index']}:{issue}" for issue in issues)
        phrase_checks.append({
            "phrase_index": p["phrase_index"], "duration_s": duration, "word_count": count,
            "words_per_second": words_per_second, "instant_word_fraction": instant_fraction,
            "hard_issues": issues,
        })
    scored_words = [w for w in words if w.get("probability") is not None]
    low_probability_fraction = (
        sum(w["probability"] < quality["low_word_probability"] for w in scored_words)
        / len(scored_words) if scored_words else 0.0)
    review_reasons = list(hard_issues)
    if any(p.get("human_annotated") for p in phrases):
        review_reasons.append("manual_annotation_requires_explicit_approval")
    if low_probability_fraction > quality["review_low_probability_fraction"]:
        review_reasons.append("contains_low_probability_words")
    return {
        "hard_passed": bool(phrases) and not hard_issues,
        "hard_issues": hard_issues,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "low_probability_word_fraction": low_probability_fraction,
        "phrase_checks": phrase_checks,
    }


def weighted_overlap(intervals, values, valid, a, b):
    overlap = np.maximum(0, np.minimum(b, intervals[:, 1]) - np.maximum(a, intervals[:, 0]))
    weights = overlap[:, None] * valid
    denom = weights.sum(0)
    values = np.where(valid, values, 0)
    result = np.divide((weights * values).sum(0), denom, out=np.zeros(values.shape[1]), where=denom > 0)
    return result, denom > 0, np.flatnonzero(overlap > 0).tolist()


def visual_observation_state(vision, indices):
    """Observed failure mode, without pretending detector failure proves physical absence."""
    if not len(indices):
        return "sampling_no_coverage"
    valid = vision["valid"][indices].all(1)
    if not valid.any():
        return ("detected_geometry_rejected" if vision["geometry_rejected"][indices].any()
                else "no_reliable_face_detected")
    return ("observed_with_subject_ambiguity" if (vision["ambiguous"][indices] & valid).any()
            else "observed")


def aggregate(phrases, text_features, audio, vision):
    features, masks = [], []
    for p in phrases:
        generated_flags = {"no_valid_face_in_phrase", "ambiguous_face_selection",
                           "face_crop_fallback", "invalid_face_geometry"}
        p["flags"] = [f for f in p["flags"] if f not in generated_flags]
        a, b = p["start_s"], p["end_s"]
        offsets = text_features["offsets"]
        mid = offsets.mean(1)
        token_indices = np.flatnonzero((mid >= p["char_start"]) & (mid < p["char_end"]))
        tx = (text_features["embeddings"][token_indices].mean(0) if len(token_indices) else np.zeros(128))
        if p.get("audio_missing", False):
            ac, am, audio_indices = np.zeros(15), np.zeros(15, bool), []
            f0, fm, pitch_indices = np.zeros(1), np.zeros(1, bool), []
        else:
            ac, am, audio_indices = weighted_overlap(
                audio["intervals"], audio["values"], np.isfinite(audio["values"]), a, b)
            f0, fm, pitch_indices = weighted_overlap(
                audio["f0_intervals"], audio["f0"][:, None],
                audio["f0_valid"][:, None], a, b)
        vi = np.flatnonzero((vision["time"] >= a) & (vision["time"] < b))
        vv, vm = vision["values"][vi], vision["valid"][vi]
        denom = vm.sum(0)
        vx = np.divide(np.where(vm, vv, 0).sum(0), denom, out=np.zeros(6), where=denom > 0)
        valid = np.concatenate([np.full(128, len(token_indices) > 0), am, fm, denom > 0])
        values = np.concatenate([tx, ac, f0, vx])
        features.append(np.where(valid, values, 0))
        masks.append(valid)
        p.update({"token_indices": token_indices.tolist(), "audio_window_indices": audio_indices,
                  "pitch_window_indices": pitch_indices, "sampled_frame_positions": vi.tolist(),
                  "frame_indices": vision["frame_index"][vi].tolist(),
                  "frame_pts_s": vision["pts"][vi].tolist(),
                  "frame_times_s": vision["time"][vi].tolist(),
                  "visual_observation_state": visual_observation_state(vision, vi),
                  "valid_face_frames": int(vision["valid"][vi].all(1).sum())})
        if not (denom > 0).all():
            p["flags"].append("no_valid_face_in_phrase")
        if vision["ambiguous"][vi].any():
            p["flags"].append("ambiguous_face_selection")
        if vision["crop_fallback_used"][vi].any():
            p["flags"].append("face_crop_fallback")
        if vision["geometry_rejected"][vi].any():
            p["flags"].append("invalid_face_geometry")
    return np.asarray(features, dtype=np.float32), np.asarray(masks, dtype=bool)
