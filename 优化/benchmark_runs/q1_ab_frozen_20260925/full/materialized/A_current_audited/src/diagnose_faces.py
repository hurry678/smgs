"""Evaluate bounded same-model multiscale fallback on a real zero-detection clip."""
import json
import time
import numpy as np
from modalities import vision_features
from prepare import ROOT, dump


def main():
    c = json.loads((ROOT / "configs/q1.json").read_text())["vision"]
    c.update({"crop_fallback": True, "crop_fraction": .6})
    sid = "-iRBcNs9oI8$_$3"
    row = next(r for r in json.loads((ROOT / "resources/manifest.json").read_text()) if r["sample_id"] == sid)
    before = np.load(ROOT / "intermediate/samples" / sid / "vision.npz")
    t0 = time.perf_counter()
    after = vision_features(row, c)
    report = {"sample_id": sid, "before_face_valid_rate": float(before["valid"].all(1).mean()),
              "after_face_valid_rate": float(after["valid"].all(1).mean()),
              "frames_recovered_by_crop": int(after["crop_fallback_used"].sum()),
              "sampled_frames": len(after["time"]), "elapsed_s": time.perf_counter()-t0,
              "same_model_thresholds": True, "fixed_crop_fraction": .6,
              "crop_count": 5, "label_use": False}
    dump(ROOT / "reports/face_retry.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
