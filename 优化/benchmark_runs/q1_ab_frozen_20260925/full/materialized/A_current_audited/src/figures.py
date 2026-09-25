"""Make standalone figures from actual Q1 waveforms, features, and source frames."""
import csv
import json
import textwrap
import numpy as np
from prepare import ROOT


def main():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.parse_math"] = False
    import matplotlib.pyplot as plt
    import cv2
    import soundfile as sf
    pilots = json.loads((ROOT / "configs/pilot.json").read_text())
    examples = list(dict.fromkeys(pilots + ["-THoVjtIkeU$_$2"]))
    out = ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    for sid in examples:
        meta = json.loads((ROOT / "outputs/details" / (sid + ".json")).read_text())
        cache = ROOT / "intermediate/samples" / sid
        y, sr = sf.read(ROOT / "intermediate/audio" / (sid + ".wav"))
        with np.load(cache / "audio.npz") as au, np.load(cache / "vision.npz") as vi:
            fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True, constrained_layout=True,
                                     gridspec_kw={"height_ratios": [1.3, 1, 1, 1, 1]})
            fig.suptitle(
                f"Q1 temporal alignment | {sid}\n"
                "MFA-primary automatic protocol; stable-ts/VAD consistency evidence",
                fontsize=13)
            t = np.arange(len(y)) / sr + meta["audio_start_s"]
            axes[0].plot(t[::16], y[::16], linewidth=.45, color="#45556c")
            axes[0].set_ylabel("Waveform")
            colors = plt.get_cmap("tab10")
            for p in meta["phrases"]:
                j, a, b = p["phrase_index"], p["start_s"], p["end_s"]
                for ax in axes[:4]:
                    ax.axvspan(a, b, color=colors(j % 10), alpha=.08)
                    ax.axvline(a, linewidth=.5, color=colors(j % 10))
                axes[4].broken_barh([(a, b-a)], (0.2, .6), facecolors=colors(j % 10), alpha=.65)
                axes[4].text((a+b)/2, .5, f"P{j}", ha="center", va="center", fontsize=9)
            axes[1].plot(au["intervals"].mean(1), au["values"][:, 13], color="#267d8c", linewidth=.8)
            axes[1].set_ylabel("log RMS")
            voiced = au["f0_valid"]
            axes[2].scatter(au["f0_intervals"].mean(1)[voiced], au["f0"][voiced], s=3, color="#a85d23")
            axes[2].set_ylabel("F0 (Hz)")
            face = vi["valid"][:, 2]
            mouth = np.where(face, vi["values"][:, 2], np.nan)
            axes[3].plot(vi["time"], mouth, marker=".", linewidth=.8, color="#8054a2")
            if not face.any():
                axes[3].text(.5, .5, "No valid face observation (mask=False)",
                             ha="center", va="center", transform=axes[3].transAxes, color="#666666")
            axes[3].set_ylabel("Mouth opening\n(normalized)")
            axes[4].set_ylabel("Phrases")
            axes[4].set_yticks([])
            axes[4].set_ylim(0, 1)
            axes[4].set_xlabel("Actual clip time (s)")
            axes[4].set_xlim(0, meta["duration_s"])
            for ax in axes:
                ax.grid(axis="x", color="#e3e7ed", linewidth=.4)
                ax.spines[["top", "right"]].set_visible(False)
            fig.savefig(out / (sid + "_alignment.svg"))
            fig.savefig(out / (sid + "_alignment.png"), dpi=150)
            plt.close(fig)
            # One real, in-phrase sampled frame per phrase; no generated imagery.
            count = len(meta["phrases"])
            cols, rows = min(3, count), (count + 2) // 3
            fig, axarr = plt.subplots(rows, cols, figsize=(4.4*cols, 3.3*rows), squeeze=False)
            cap = cv2.VideoCapture(str(ROOT / meta["source_file"]))
            for ax in axarr.flat:
                ax.axis("off")
            for ax, p in zip(axarr.flat, meta["phrases"]):
                indices = p["sampled_frame_positions"]
                if indices:
                    face_indices = [k for k in indices if vi["valid"][k].all()]
                    chosen = (face_indices or indices)[len(face_indices or indices)//2]
                    idx = int(vi["frame_index"][chosen])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if ok:
                        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    timestamp = float(vi["time"][chosen])
                    title = f"P{p['phrase_index']} | [{p['start_s']:.2f}, {p['end_s']:.2f}) s\nFrame {idx}, PTS={timestamp:.3f} s"
                else:
                    title = f"P{p['phrase_index']}: no sampled frame in interval"
                ax.set_title(title, fontsize=10)
                ax.text(.5, -.04, "\n".join(textwrap.wrap(p["text"], 52)), ha="center", va="top",
                        fontsize=9, transform=ax.transAxes)
            cap.release()
            fig.suptitle(f"Original frames and transcript | {sid}", fontsize=13)
            fig.subplots_adjust(hspace=.65, top=.9, bottom=.12)
            fig.savefig(out / (sid + "_frames.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)
    summary = list(csv.DictReader((ROOT / "outputs/q1_summary.csv").open(encoding="utf-8-sig")))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    axes[0].hist([float(r["duration_s"]) for r in summary], bins=12, color="#3c7689")
    axes[0].set_xlabel("Measured duration (s)")
    axes[1].hist([int(r["phrases"]) for r in summary], bins=range(1, 18), color="#7f6599")
    axes[1].set_xlabel("Phrases per clip")
    axes[2].hist([float(r["frame_face_valid_rate"]) for r in summary], bins=np.linspace(0, 1, 11), color="#b58950")
    axes[2].set_xlabel("Valid face / sampled frames")
    for ax in axes:
        ax.set_ylabel("Clips")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Q1 extraction audit | {len(summary)} real clips", fontsize=13)
    fig.savefig(out / "coverage_overview.svg")
    fig.savefig(out / "coverage_overview.png", dpi=150)
    plt.close(fig)
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    boxes = [
        (0.25, 3.1, 1.7, 1.1, "Input audit\n100 videos + text", "#dcebf2"),
        (2.35, 3.1, 1.7, 1.1, "Real timeline\nffprobe PTS", "#e9e4f1"),
        (4.45, 3.1, 1.7, 1.1, "MFA primary\nalignment", "#f5e7cd"),
        (6.55, 3.1, 1.7, 1.1, "Native features\n128 + 16 + 6", "#dcebdc"),
        (8.65, 3.1, 1.7, 1.1, "Overlap-weighted\nphrase aggregation", "#dcebf2"),
        (10.75, 3.1, 1.9, 1.1, "Validate, report\nand package", "#e9e4f1"),
        (4.45, 0.65, 1.7, 1.1, "Missing speech\nclip context + mask", "#f1dddd"),
        (6.55, 0.65, 1.7, 1.1, "Face miss\nfixed crop retry", "#f1dddd"),
    ]
    for x, y0, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                    facecolor=color, edgecolor="#44515d", linewidth=1.2))
        ax.text(x+w/2, y0+h/2, label, ha="center", va="center", fontsize=10)
    for x in (1.95, 4.05, 6.15, 8.25, 10.35):
        ax.annotate("", xy=(x+.4, 3.65), xytext=(x, 3.65),
                    arrowprops={"arrowstyle": "->", "color": "#44515d", "lw": 1.5})
    ax.annotate("", xy=(5.3, 1.75), xytext=(5.3, 3.1),
                arrowprops={"arrowstyle": "->", "color": "#9b3f3f", "lw": 1.4})
    ax.annotate("", xy=(7.4, 1.75), xytext=(7.4, 3.1),
                arrowprops={"arrowstyle": "->", "color": "#9b3f3f", "lw": 1.4})
    ax.annotate("", xy=(9.6, 3.1), xytext=(6.15, 1.2),
                arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1.2,
                            "connectionstyle": "arc3,rad=-0.16"})
    ax.annotate("", xy=(8.65, 3.35), xytext=(8.25, 1.2),
                arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1.2,
                            "connectionstyle": "arc3,rad=0.16"})
    ax.text(6.5, 4.85, "Q1 reproducible extraction and missing-modality protocol",
            ha="center", fontsize=15, fontweight="bold", color="#24313b")
    fig.savefig(out / "solution_roadmap.svg", bbox_inches="tight")
    fig.savefig(out / "solution_roadmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures to {out.relative_to(ROOT)}")
