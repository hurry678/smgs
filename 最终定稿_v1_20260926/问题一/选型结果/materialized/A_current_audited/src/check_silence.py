"""Verify digital-silence diagnosis with an independent bundled FFmpeg decoder."""
import json
import av
import numpy as np
from prepare import ROOT, dump


def main():
    results = []
    for clip in ("1", "2"):
        source = ROOT / "resources/input/-mJ2ud6oKI8" / (clip + ".mp4")
        frames = []
        with av.open(str(source)) as container:
            for frame in container.decode(audio=0):
                frames.append(frame.to_ndarray())
        data = np.concatenate(frames, axis=1)
        results.append({"sample_id": "-mJ2ud6oKI8$_$" + clip, "shape": list(data.shape),
                        "sample_rate": frame.sample_rate, "decoded_frames": len(frames),
                        "peak_abs": float(np.max(np.abs(data))), "nonzero_samples": int(np.count_nonzero(data)),
                        "all_zero": bool(np.all(data == 0))})
    report = {"pyav_version": av.__version__, "library_versions": av.library_versions,
              "scope": "Independently decode both original stereo AAC channels without downmixing",
              "results": results}
    dump(ROOT / "reports/silence_check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
