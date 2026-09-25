#!/usr/bin/env python3
"""Standalone frozen inference: rebuild the packaged model and predict one archive.

Only ``numpy``, ``torch``, ``transformers`` and this directory are required.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

LABELS = ("Negative", "Neutral", "Positive")


def apply_normalizer(x, visible, mean, std):
    values = (np.asarray(x, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return np.where(np.asarray(visible, dtype=bool)[..., None], values, 0).astype(np.float32)


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def publish_c2(prob, raw, epsilon=1e-4):
    cls = np.asarray(prob, dtype=np.float64).argmax(1).astype(np.int64)
    r = np.clip(np.asarray(raw, dtype=np.float64), -3.0, 3.0)
    out = np.zeros_like(r)
    pos, neg = cls == 2, cls == 0
    out[pos] = np.maximum(r[pos], epsilon)
    out[neg] = np.minimum(r[neg], -epsilon)
    return cls, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    pkg = args.package.resolve()
    sys.path.insert(0, str(pkg / "src"))
    import models as M
    cfg = json.loads((pkg / "model" / "config.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if cfg["architecture"] == "training_prior":
        model = M.build_model("training_prior", class_counts=cfg["class_counts"],
                              class_intensity_mean=cfg["class_intensity_mean"])
    else:
        model = M.build_model(cfg["architecture"], **cfg.get("model_kwargs", {}))
        state = torch.load(pkg / "model" / "state_dict.pt", map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("text_encoder.")]
        unexpected = [k for k in unexpected if not k.startswith("text_encoder.")]
        if missing or unexpected:
            raise SystemExit("state_dict mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    model.to(device).eval()
    z = np.load(args.input, allow_pickle=False)
    norm = np.load(pkg / "data" / "normalizer.npz", allow_pickle=False)
    observed = z["observed"].astype(bool)
    tensors = {
        "text_bert": torch.from_numpy(z["text_bert"].astype(np.int64)).to(device),
        "audio": torch.from_numpy(apply_normalizer(z["audio"], observed[:, 1],
                                                   norm["audio_mean"], norm["audio_std"])).to(device),
        "vision": torch.from_numpy(apply_normalizer(z["vision"], observed[:, 2],
                                                    norm["vision_mean"], norm["vision_std"])).to(device),
        "domain": torch.from_numpy(z["domain"].astype(bool)).to(device),
    }
    visible = torch.from_numpy(observed).to(device)
    logits, raw = [], []
    with torch.no_grad():
        for start in range(0, len(observed), 32):
            stop = min(start + 32, len(observed))
            out = model(tensors["text_bert"][start:stop], tensors["audio"][start:stop],
                        tensors["vision"][start:stop], tensors["domain"][start:stop],
                        visible[start:stop])
            logits.append(out["logits"].detach().float().cpu().numpy())
            raw.append(out["raw_intensity"].detach().float().cpu().numpy())
    logits = np.concatenate(logits)
    raw = np.concatenate(raw)
    prob = softmax(logits)
    cls, intensity = publish_c2(prob, raw, cfg["publication"]["nonneutral_epsilon"])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "logits.npz", logits=logits, raw=raw, prob=prob)
    ids = [str(x) for x in z["id"]]
    files = [str(x) for x in z["source_file"]]
    with (args.out / "q2_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cfg["publication"]["csv_columns"])
        for i, sid in enumerate(ids):
            writer.writerow([sid, files[i], LABELS[int(cls[i])], "%.6f" % intensity[i]])
    print(json.dumps({"rows": len(ids),
                      "ids_in_order": ids == ["附件3_%02d" % i for i in range(1, 31)]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
