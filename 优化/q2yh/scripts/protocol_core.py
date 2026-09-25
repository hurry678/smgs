"""NumPy reference implementation of q2-mask-domain-1; no training or file writes."""
from __future__ import annotations

import hashlib
import json
import numpy as np

CLASSES = ("Negative", "Neutral", "Positive")
PROTOCOL = "q2-mask-domain-1"


def mask_hash(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def stable_index(parts, count):
    key = json.dumps([PROTOCOL, *parts], ensure_ascii=False, separators=(",", ":"))
    number = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return number % count


def interface_masks(text_bert, audio, vision):
    """Return immutable-domain semantics and original observation states.

    UNK=100 is an observed unknown token, not an oracle missing indicator.
    """
    tb = np.asarray(text_bert)
    a, v = np.asarray(audio), np.asarray(vision)
    if tb.ndim != 3 or tb.shape[1] != 3:
        raise ValueError("text_bert must have shape [N,3,T]")
    n, _, t = tb.shape
    if a.shape[:2] != (n, t) or v.shape[:2] != (n, t) or a.ndim != 3 or v.ndim != 3:
        raise ValueError("audio/vision must have matching [N,T,D] shapes")
    if not all(np.isfinite(x).all() for x in (tb, a, v)):
        raise ValueError("Nonfinite source values; do not silently sanitize")
    if not np.array_equal(tb, np.rint(tb)):
        raise ValueError("text_bert channels must be integral")
    ids = tb[:, 0].astype(np.int64)
    if np.any(ids < 0) or np.any(ids >= 30522):
        raise ValueError("Token ID outside frozen vocabulary")
    if not np.isin(tb[:, 1:], (0, 1)).all():
        raise ValueError("Attention/type channels must be binary")
    if not np.all(ids[:, 0] == 101) or not np.all((ids == 102).sum(1) == 1):
        raise ValueError("Require CLS at index 0 and exactly one SEP")
    sep = (ids == 102).argmax(1)
    pos = np.arange(t)[None, :]
    domain = (pos > 0) & (pos < sep[:, None])
    if not domain.any(1).all():
        raise ValueError("Empty semantic domain requires explicit interface revision")
    text_obs = domain & (tb[:, 1] == 1) & ~np.isin(ids, (0, 103))
    obs = np.stack((text_obs, domain & np.any(a != 0, -1),
                    domain & np.any(v != 0, -1)), axis=1)
    return domain, obs


def sanitize_text(text_bert, domain, visible):
    """Mask before BERT; special tokens are internal support, never fusion evidence."""
    tb = np.asarray(text_bert, dtype=np.int64).copy()
    d = np.asarray(domain, dtype=bool)
    m = np.asarray(visible, dtype=bool)
    if m.shape != d.shape or tb.shape != (len(d), 3, d.shape[1]) or np.any(m & ~d):
        raise ValueError("Invalid text visibility/domain")
    # Domain is fixed before masking; never infer it from changed token contents.
    hidden = d & ~m
    tb[:, 0][hidden] = 103
    special = ~d & np.isin(tb[:, 0], (101, 102))
    tb[:, 1] = (m | special).astype(np.int64)
    tb[:, 0][~d & ~special] = 0
    tb[:, 2][~d & ~special] = 0
    tb[:, 2][hidden] = 0
    return tb


def fit_normalizer(x, observed):
    vals = np.asarray(x, dtype=np.float64)[np.asarray(observed, dtype=bool)]
    if not len(vals) or not np.isfinite(vals).all():
        raise ValueError("No finite observed training rows for normalization")
    return vals.mean(0), np.maximum(vals.std(0), 1e-5)


def normalize(x, visible, mean, std):
    values = (np.asarray(x, dtype=np.float32) - mean) / std
    return np.where(np.asarray(visible, dtype=bool)[..., None], values, 0).astype(np.float32)


def make_missing(domain, observed, sample_ids, subset, ratio, *, seed=20260925,
                 split="valid", replica=0, placement="random", coupling="independent"):
    """Sample physical index intervals without collapsing natural missing holes.

    Returns final visibility plus per-selected-modality audit rows. No input mutation.
    """
    d = np.asarray(domain, dtype=bool)
    o = np.asarray(observed, dtype=bool)
    if d.ndim != 2 or o.shape != (len(d), 3, d.shape[1]) or np.any(o & ~d[:, None]):
        raise ValueError("Expected O subset of D with [N,3,T] and [N,T]")
    if len(sample_ids) != len(d) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Unique sample IDs must match batch length")
    subset = tuple(sorted(subset))
    if not subset or len(set(subset)) != len(subset) or any(m not in (0, 1, 2) for m in subset):
        raise ValueError("Invalid modality subset")
    if not 0 < ratio < 1 or placement not in ("random", "start", "middle", "end"):
        raise ValueError("Invalid ratio/placement")
    if coupling not in ("independent", "synchronized"):
        raise ValueError("Invalid coupling")
    out, records = o.copy(), []
    for i, sid in enumerate(sample_ids):
        indices = np.flatnonzero(d[i])
        length = len(indices)
        if length and not np.all(np.diff(indices) == 1):
            raise ValueError("Domain must be contiguous; do not compress observed positions")
        first = int(indices[0]) if length else 0
        k = min(length - 1, max(1, int(np.floor(ratio * length + 0.5)))) if length >= 2 else 0
        eligible = [m for m in subset if o[i, m].sum() >= 2 and k > 0]
        def candidates(modalities):
            return [s for s in range(first, first + length - k + 1)
                    if all(0 < o[i, m, s:s+k].sum() < o[i, m].sum() for m in modalities)]
        def choose(starts, modkey):
            if not starts:
                return None
            if placement == "start":
                return starts[0]
            if placement == "end":
                return starts[-1]
            if placement == "middle":
                return min(starts, key=lambda s: (abs(s + k / 2 - (first + length / 2)), s))
            parts = (seed, split, sid, subset, float(ratio), placement, replica, coupling, modkey)
            return starts[stable_index(parts, len(starts))]
        shared = choose(candidates(eligible), "shared") if coupling == "synchronized" and eligible else None
        for m in subset:
            n_original = int(o[i, m].sum())
            start = None
            status = "skipped_lt2_observed"
            if m in eligible:
                start = shared if coupling == "synchronized" else choose(candidates([m]), m)
                status = "applied" if start is not None else "skipped_no_eligible_window"
            if start is not None:
                out[i, m, start:start+k] = False
            n_visible = int(out[i, m].sum())
            records.append({
                "sample_id": str(sid), "modality": m, "domain_length": length,
                "original_observed": n_original, "visible": n_visible,
                "requested_ratio": float(ratio), "start": start,
                "stop": start + k if start is not None else None, "status": status,
                "window_ratio": k / length if start is not None else 0.0,
                "removed_observed_ratio": (n_original - n_visible) / n_original if n_original else None,
                "native_missing_ratio": 1 - n_original / length if length else None,
                "final_missing_ratio": 1 - n_visible / length if length else None,
            })
    return out, records


def gap_structure(domain, visible):
    """[N,T,26]: six per-modality channels plus eight observed-subset indicators."""
    d = np.asarray(domain, dtype=bool)
    m = np.asarray(visible, dtype=bool)
    if m.shape != (len(d), 3, d.shape[1]) or np.any(m & ~d[:, None]):
        raise ValueError("Invalid visibility/domain")
    q = np.zeros((len(d), d.shape[1], 26), dtype=np.float32)
    for i in range(len(d)):
        positions = np.flatnonzero(d[i])
        length = len(positions)
        if not length:
            continue
        for mod in range(3):
            observed = np.flatnonzero(m[i, mod])
            for t in positions:
                local = positions[np.abs(positions - t) <= 2]
                run = 0
                if not m[i, mod, t]:
                    left, right = t, t
                    while left > 0 and d[i, left-1] and not m[i, mod, left-1]:
                        left -= 1
                    while right + 1 < len(d[i]) and d[i, right+1] and not m[i, mod, right+1]:
                        right += 1
                    run = right - left + 1
                before, after = observed[observed <= t], observed[observed >= t]
                dl = (t - before[-1]) / length if len(before) else 1.0
                dr = (after[0] - t) / length if len(after) else 1.0
                q[i, t, 6*mod:6*mod+6] = (
                    m[i, mod, t], m[i, mod, local].mean(), run / length,
                    dl, dr, len(observed) / length)
        states = m[i, 0].astype(int) + 2*m[i, 1] + 4*m[i, 2]
        q[i, positions, 18 + states[positions]] = 1
    return q


def publish_c2(probabilities, raw, epsilon=1e-4):
    p, r = np.asarray(probabilities), np.asarray(raw, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or r.shape != (len(p),):
        raise ValueError("Invalid prediction shapes")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite prediction")
    if (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError("Expected probabilities, not logits")
    c = p.argmax(1)
    r = r.clip(-3, 3)
    result = np.where(c == 0, np.minimum(r, -epsilon),
                      np.where(c == 2, np.maximum(r, epsilon), 0.0))
    return c, result
