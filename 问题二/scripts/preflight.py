#!/usr/bin/env python3
"""Read-only source audit; writes only to a new directory under 问题二/."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import sys
import numpy as np
from protocol_core import interface_masks, make_missing, mask_hash

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def shapes(item, n):
    for field, shape in {
        "text_bert": (n, 3, 50), "audio": (n, 50, 74), "vision": (n, 50, 35)
    }.items():
        if np.asarray(item[field]).shape != shape:
            raise ValueError(f"{field}: expected {shape}, got {np.asarray(item[field]).shape}")


def audit(args):
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    sources = {"aligned_50.pkl": sha(args.aligned)}
    checks = []
    # Original competition pickle only. Test object is deserialized with the
    # container, but its labels/metrics never enter the operations below.
    with args.aligned.open("rb") as f:
        data = pickle.load(f)
    identities, splits = {}, {}
    for split in ("train", "valid", "test"):
        item = data[split]
        n = protocol["data"][split + "_count"]
        shapes(item, n)
        ids = list(map(str, item["id"]))
        if len(ids) != n or len(set(ids)) != n or any("$_$" not in s for s in ids):
            raise ValueError(f"Invalid identities: {split}")
        identities[split] = ids
        groups = {s.split("$_$", 1)[0] for s in ids}
        splits[split] = {"count": n, "video_groups": len(groups)}
        if split == "test":
            splits[split]["scope"] = "shape_and_identity_only_no_labels_or_metrics"
            continue
        tb, a, v = (np.asarray(item[k]) for k in ("text_bert", "audio", "vision"))
        domain, obs = interface_masks(tb, a, v)
        cls = np.asarray(item["classification_labels"])
        reg = np.asarray(item["regression_labels"])
        if cls.shape != (n,) or reg.shape != (n,):
            raise ValueError("Unexpected label shapes")
        if not np.isfinite(reg).all() or not np.isin(cls, (0, 1, 2)).all() or (np.abs(reg) > 3).any():
            raise ValueError(f"Invalid labels: {split}")
        if not np.array_equal(cls, np.where(reg < 0, 0, np.where(reg > 0, 2, 1))):
            raise ValueError(f"Label polarity/intensity mismatch: {split}")
        mask, records = make_missing(domain, obs, ids, (0, 1, 2), 0.3,
                                     split=split, seed=20260925)
        splits[split].update({
            "class_counts": {str(k): int((cls == k).sum()) for k in (0, 1, 2)},
            "semantic_lengths": {"min": int(domain.sum(1).min()), "max": int(domain.sum(1).max())},
            "original_empty_modalities": (~obs.any(-1)).sum(0).tolist(),
            "out_of_domain_nonzero_audio_rows": int((np.any(a != 0, -1) & ~domain).sum()),
            "out_of_domain_nonzero_vision_rows": int((np.any(v != 0, -1) & ~domain).sum()),
            "mask_smoke_status": dict(Counter(r["status"] for r in records)),
            "mask_smoke_sha256": mask_hash(mask),
            "original_observation_sha256": mask_hash(obs)
        })
    for s1, s2 in (("train", "valid"), ("train", "test"), ("valid", "test")):
        ids1, ids2 = set(identities[s1]), set(identities[s2])
        groups1, groups2 = ({s.split("$_$", 1)[0] for s in ids} for ids in (ids1, ids2))
        if ids1 & ids2 or groups1 & groups2:
            raise ValueError(f"Split overlap: {s1}/{s2}")
        checks.append(f"no_id_or_video_overlap_{s1}_{s2}")
    files = sorted(args.attachment3_dir.glob("*.pkl"))
    expected = {f"附件3_{i:02d}.pkl" for i in range(1, 31)}
    if {p.name for p in files} != expected:
        raise ValueError("Attachment3 filenames must be exactly 附件3_01.pkl ... 附件3_30.pkl")
    for path in files:
        sources["attachment3/" + path.name] = sha(path)
        with path.open("rb") as f:
            item = pickle.load(f)["test"]
        if set(item) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"Unexpected Attachment3 fields: {path.name}; revise contract explicitly")
        shapes(item, 1)
        interface_masks(item["text_bert"], item["audio"], item["vision"])
    checks += ["attachment3_interface_30", "train_valid_masks_labels_finite"]
    resources = json.loads((ROOT / "resources/manifest.json").read_text())
    for item in resources["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed model resource: {item['path']}")
    vocab = (ROOT / "resources/bert_shared/vocab.txt").read_text().splitlines()
    if len(vocab) != 30522 or [vocab[i] for i in (0, 100, 101, 102, 103)] != [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ]:
        raise ValueError("Unexpected BERT vocabulary")
    checks += ["frozen_bert_files_and_special_ids"]
    packages = {}
    for name in ("numpy", "torch", "transformers", "safetensors", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "q2-preflight-1", "passed": True, "scope": "input_and_protocol_only",
        "model_training_status": "not_run", "checks": checks, "splits": splits,
        "attachment3": {"count": len(files), "ids": [p.stem for p in files],
                        "scope": "schema_only_not_used_to_choose_training_distribution",
                        "has_precomputed_text": False},
        "source_sha256": sources, "protocol_sha256": sha(ROOT / "configs/protocol.json"),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages},
        "training_dependencies_ready": all(packages[n] for n in ("numpy", "torch", "transformers", "safetensors")),
        "note": "Preflight passed does not certify completed training, performance or final submission."
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--attachment3-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        ap.error("output-dir must be a new child directory under 问题二")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args)
    except Exception as e:
        result = {"schema": "q2-preflight-1", "passed": False, "error": f"{type(e).__name__}: {e}"}
    (output / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "report": str(output / "preflight.json")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
