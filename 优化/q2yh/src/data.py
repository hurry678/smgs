"""Data preparation and mask banks for q2-plan-1.0 (server task T01).

Reads the competition pickles read-only, writes split archives under a new run
directory. Training code never receives the test or Attachment-3 path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

CLASSES = ("Negative", "Neutral", "Positive")
SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
RATIOS = (0.1, 0.3, 0.5)
MODALITIES = ("text", "audio", "vision")
FIELDS = ("text_bert", "audio", "vision")
POSITION_PLACEMENTS = ("start", "middle", "end")
COUPLING_SUBSETS = ((0, 1), (0, 2), (1, 2), (0, 1, 2))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def video_id(sample_id):
    return str(sample_id).split("$_$", 1)[0]


def pack_masks(masks):
    """[C,N,3,T] bool -> uint8 [C,N,3,ceil(T/8)] bit-packed along the last axis."""
    m = np.ascontiguousarray(np.asarray(masks, dtype=bool))
    packed = np.packbits(m, axis=-1)
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_masks(packed, n_bits):
    m = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, count=int(n_bits))
    return m.astype(bool)


def sample_mask_sha(mask_3xT):
    return sha256_bytes(np.ascontiguousarray(np.asarray(mask_3xT, dtype=bool), dtype=np.uint8).tobytes())


def condition_sha(masks):
    """Order-sensitive digest of one condition's [N,3,T] visibility tensor."""
    m = np.ascontiguousarray(np.asarray(masks, dtype=bool))
    return sha256_bytes(m.astype(np.uint8).tobytes())


def bank_sha(packed, condition_ids):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(packed, dtype=np.uint8).tobytes())
    for cid in condition_ids:
        h.update(b"\x00" + str(cid).encode())
    return h.hexdigest()


def condition_id(bank, subset, ratio, placement, coupling, replica):
    modstr = "".join(str(m) for m in subset)
    return f"{bank}:{modstr}:{ratio:.2f}:{placement}:{coupling}:{int(replica)}"


def _subset_of(modstr):
    return tuple(int(c) for c in modstr)


def split_condition_id(cid):
    bank, modstr, ratio, placement, coupling, replica = str(cid).split(":")
    return {
        "bank": bank, "subset": _subset_of(modstr), "ratio": float(ratio),
        "placement": placement, "coupling": coupling, "replica": int(replica),
    }


# --------------------------------------------------------------------------
# split archives
# --------------------------------------------------------------------------

def load_aligned(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_attachment3(attachment3_dir):
    """30 aligned pickles, each with a leading N=1 axis; returns stacked arrays."""
    paths = sorted(Path(attachment3_dir).glob("*.pkl"))
    expected = [f"附件3_{i:02d}.pkl" for i in range(1, 31)]
    if [p.name for p in paths] != expected:
        raise ValueError("Attachment3 filenames must be exactly 附件3_01.pkl ... 附件3_30.pkl")
    out = {"id": [], "text_bert": [], "audio": [], "vision": [], "file": [], "sha256": []}
    for path in paths:
        with path.open("rb") as f:
            item = pickle.load(f)["test"]
        if set(item) != set(FIELDS):
            raise ValueError(f"Unexpected Attachment3 fields in {path.name}: {sorted(item)}")
        out["id"].append(path.stem)
        out["file"].append(path.name)
        out["sha256"].append(sha256_file(path))
        for field in FIELDS:
            arr = np.asarray(item[field])
            if arr.shape[0] != 1:
                raise ValueError(f"{path.name}/{field}: expected leading N=1 axis, got {arr.shape}")
            out[field].append(arr[0])
    return {
        "id": np.asarray(out["id"]),
        "source_file": np.asarray(out["file"]),
        "sha256": np.asarray(out["sha256"]),
        "text_bert": np.stack(out["text_bert"]).astype(np.int64),
        "audio": np.stack(out["audio"]).astype(np.float32),
        "vision": np.stack(out["vision"]).astype(np.float32),
    }


def split_payload(raw, name, with_labels=True):
    """Convert one competition split into the stored archive payload."""
    ids = np.asarray([str(s) for s in raw["id"]])
    payload = {
        "id": ids,
        "video_id": np.asarray([video_id(s) for s in ids]),
        "text_bert": np.asarray(raw["text_bert"], dtype=np.int64),
        "audio": np.asarray(raw["audio"], dtype=np.float32),
        "vision": np.asarray(raw["vision"], dtype=np.float32),
        "split": np.asarray(name),
    }
    if with_labels:
        cls = np.asarray(raw["classification_labels"], dtype=np.float64)
        reg = np.asarray(raw["regression_labels"], dtype=np.float64)
        if not np.isin(cls, (0, 1, 2)).all():
            raise ValueError(f"{name}: classification labels outside 0/1/2")
        if not np.isfinite(reg).all() or (np.abs(reg) > 3).any():
            raise ValueError(f"{name}: invalid regression labels")
        payload["classification_labels"] = cls.astype(np.int8)
        payload["regression_labels"] = reg.astype(np.float32)
    return payload


def save_npz(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # open a handle: np.savez(path, **payload) breaks when a payload key is literally "file"
    with path.open("wb") as handle:
        np.savez(handle, **payload)
    return sha256_file(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
# normalizer (train observations only)
# --------------------------------------------------------------------------

def fit_normalizer(audio, vision, observed):
    """Per-dimension train statistics over rows with O=1 (float64 accumulation)."""
    obs = np.asarray(observed, dtype=bool)
    stats = {}
    for name, x, mask in (("audio", audio, obs[:, 1]), ("vision", vision, obs[:, 2])):
        vals = np.asarray(x, dtype=np.float64)[mask]
        if not len(vals) or not np.isfinite(vals).all():
            raise ValueError(f"No finite observed train rows for {name}")
        stats[name] = {"mean": vals.mean(0), "std": np.maximum(vals.std(0), 1e-5)}
    return stats


def save_normalizer(path, stats, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             audio_mean=stats["audio"]["mean"].astype(np.float64),
             audio_std=stats["audio"]["std"].astype(np.float64),
             vision_mean=stats["vision"]["mean"].astype(np.float64),
             vision_std=stats["vision"]["std"].astype(np.float64),
             provenance=np.asarray(json.dumps(provenance, ensure_ascii=False)))
    return sha256_file(path)


def load_normalizer(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "audio": {"mean": z["audio_mean"], "std": z["audio_std"]},
            "vision": {"mean": z["vision_mean"], "std": z["vision_std"]},
            "provenance": json.loads(str(z["provenance"])),
        }


def apply_normalizer(x, visible, mean, std):
    """Normalize then re-zero unobserved rows; M comes from raw values, never from 0."""
    values = (np.asarray(x, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return np.where(np.asarray(visible, dtype=bool)[..., None], values, 0).astype(np.float32)


# --------------------------------------------------------------------------
# mask banks
# --------------------------------------------------------------------------

def _make(domain, observed, ids, subset, ratio, *, seed, split, replica, placement, coupling):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from protocol_core import make_missing
    return make_missing(domain, observed, ids, subset, ratio, seed=seed, split=split,
                        replica=replica, placement=placement, coupling=coupling)


def bank_conditions(bank, cfg):
    """Enumerate the pre-registered conditions of one bank."""
    mask = cfg["mask"]
    out = []
    if bank == "select":
        seed, replicas = mask["select"]["seed"], mask["select"]["replicas"]
        for subset in SUBSETS:
            for ratio in RATIOS:
                for replica in range(replicas):
                    out.append(dict(subset=subset, ratio=ratio, placement="random",
                                    coupling="independent", replica=replica, seed=seed))
    elif bank == "confirm":
        seed, replicas = mask["confirm"]["seed"], mask["confirm"]["replicas"]
        for subset in SUBSETS:
            for ratio in RATIOS:
                for replica in range(replicas):
                    out.append(dict(subset=subset, ratio=ratio, placement="random",
                                    coupling="independent", replica=replica, seed=seed))
    elif bank == "position":
        seed = mask["select"]["seed"]
        for subset in ((0,), (1,), (2,)):
            for ratio in RATIOS:
                for placement in POSITION_PLACEMENTS:
                    out.append(dict(subset=subset, ratio=ratio, placement=placement,
                                    coupling="independent", replica=0, seed=seed))
    elif bank == "coupling":
        seed = mask["select"]["seed"]
        replicas = mask["coupling"]["replicas"]
        for subset in COUPLING_SUBSETS:
            for ratio in RATIOS:
                for replica in range(replicas):
                    out.append(dict(subset=subset, ratio=ratio, placement="random",
                                    coupling="synchronized", replica=replica, seed=seed))
    else:
        raise ValueError(f"Unknown bank: {bank}")
    for spec in out:
        spec["bank"] = bank
        spec["condition_id"] = condition_id(bank, spec["subset"], spec["ratio"],
                                            spec["placement"], spec["coupling"], spec["replica"])
    return out


def build_bank(bank, domain, observed, ids, cfg):
    """Return packed visibility, condition ids, per-sample SHAs and interval records."""
    specs = bank_conditions(bank, cfg)
    n, t = domain.shape
    packed = np.empty((len(specs), n, 3, (t + 7) // 8), dtype=np.uint8)
    cond_sha, sample_sha, records = [], [], []
    for i, spec in enumerate(specs):
        vis, recs = _make(domain, observed, ids, spec["subset"], spec["ratio"],
                          seed=spec["seed"], split="valid", replica=spec["replica"],
                          placement=spec["placement"], coupling=spec["coupling"])
        if np.any(vis & ~observed):
            raise ValueError("Synthetic masking restored or invented an observation")
        packed[i] = pack_masks(vis)
        cond_sha.append(condition_sha(vis))
        sample_sha.append([sample_mask_sha(vis[j]) for j in range(n)])
        for rec in recs:
            rec["condition_id"] = spec["condition_id"]
            rec["bank"] = bank
        records.extend(recs)
    return {
        "packed": packed,
        "condition_ids": np.asarray([s["condition_id"] for s in specs]),
        "condition_sha256": np.asarray(cond_sha),
        "sample_sha256": np.asarray(sample_sha),
        "specs": json.dumps([{k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in s.items()} for s in specs], ensure_ascii=False),
        "records": records,
        "T": int(t),
        "N": int(n),
        "bank": bank,
        "bank_sha256": bank_sha(packed, [s["condition_id"] for s in specs]),
    }


def clean_bank(observed):
    """The un-masked condition: visibility equals the native observation state."""
    n, t = observed.shape[0], observed.shape[2]
    packed = pack_masks(observed[None])
    return {
        "packed": packed,
        "condition_ids": np.asarray(["clean:clean"]),
        "condition_sha256": np.asarray([condition_sha(observed)]),
        "sample_sha256": np.asarray([[sample_mask_sha(observed[j]) for j in range(n)]]),
        "specs": json.dumps([{"bank": "clean", "subset": [], "ratio": 0.0,
                              "placement": "none", "coupling": "none", "replica": 0,
                              "condition_id": "clean:clean"}], ensure_ascii=False),
        "records": [],
        "T": int(t),
        "N": int(n),
        "bank": "clean",
        "bank_sha256": bank_sha(packed, ["clean:clean"]),
    }


def save_bank(run_dir, bank_data, ids):
    path = Path(run_dir) / "masks" / f"{bank_data['bank']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=bank_data["packed"],
        condition_ids=bank_data["condition_ids"],
        condition_sha256=bank_data["condition_sha256"],
        sample_sha256=bank_data["sample_sha256"],
        sample_ids=np.asarray(ids),
        specs=np.asarray(bank_data["specs"]),
        T=np.asarray(bank_data["T"]),
        N=np.asarray(bank_data["N"]),
        bank=np.asarray(bank_data["bank"]),
        bank_sha256=np.asarray(bank_data["bank_sha256"]),
    )
    return sha256_file(path)


def load_bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "packed": z["packed"],
            "condition_ids": [str(x) for x in z["condition_ids"]],
            "condition_sha256": [str(x) for x in z["condition_sha256"]],
            "sample_sha256": z["sample_sha256"],
            "sample_ids": [str(x) for x in z["sample_ids"]],
            "specs": json.loads(str(z["specs"])),
            "T": int(z["T"]),
            "N": int(z["N"]),
            "bank": str(z["bank"]),
            "bank_sha256": str(z["bank_sha256"]),
        }


def visibility_of(bank_data, index):
    return unpack_masks(bank_data["packed"][index], bank_data["T"])