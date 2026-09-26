"""Export actual representative frames and a prioritised review queue."""
import json
from prepare import ROOT


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.parse_math"] = False
    import matplotlib.pyplot as plt
    import cv2
    items = [json.loads(p.read_text()) for p in sorted((ROOT / "outputs/details").glob("*.json"))]
    # Validation owns the review policy and queue; frame export must not overwrite decisions.
    queue = json.loads((ROOT / "reports/review_queue.json").read_text())
    # All zero-face clips and the two digitally silent clips, one actual midpoint frame each.
    selected = [r for r in items if r["face_valid_rate"] == 0 or r.get("digitally_silent_audio")]
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.2*rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, selected):
        cap = cv2.VideoCapture(str(ROOT / row["source_file"]))
        index = row["frame_count"] // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if ok:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        probe = json.loads((ROOT / "intermediate/probes" / (row["sample_id"] + ".json")).read_text())
        pts = probe["frames"][index]["best_effort_timestamp_time"]
        ax.set_title(f"{row['sample_id']}\nframe {index}, PTS={pts} s\n"
                     f"face {row['face_valid_rate']:.0%} | silent={row.get('digitally_silent_audio', False)}",
                     fontsize=9)
    fig.suptitle("Actual source frames: zero-face and silent-audio exceptions", fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/figures/exception_frames.png", dpi=140)
    plt.close(fig)
    print(f"Review queue: {len(queue)}; exception-frame montage: {len(selected)}")


if __name__ == "__main__":
    main()
