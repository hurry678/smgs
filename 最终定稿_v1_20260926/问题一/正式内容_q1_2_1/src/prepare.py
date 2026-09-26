"""Copy immutable Q1 resources, audit exact pairing, and probe real timestamps."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def probe(path, frames=False):
    cmd = ["ffprobe", "-v", "error", "-of", "json"]
    if frames:
        cmd += ["-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time,pts_time,pkt_duration_time"]
    else:
        cmd += ["-show_format", "-show_streams"]
    result = subprocess.run(cmd + [str(path)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main(source_dir=None):
    import openpyxl
    for d in ("configs", "resources/input", "resources/models", "resources/reference",
              "intermediate/probes", "annotations", "outputs/features", "outputs/figures",
              "reports", "logs", "submission", ".cache"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    search_root = Path(source_dir).expanduser() if source_dir else PROJECT / "E题数据/附件1-数据集原始多模态样本"
    matches = list(search_root.rglob("label-100.xlsx"))
    if len(matches) != 1:
        raise ValueError("Expected one label-100.xlsx under --input-dir")
    source = matches[0].parent
    target = ROOT / "resources/input"
    originals = sorted(source.rglob("*.mp4")) + [source / "label-100.xlsx"]
    hashes = []
    for path in originals:
        dest = target / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
        a, b = sha256(path), sha256(dest)
        if a != b:
            raise ValueError(f"Input copy mismatch: {path.name}")
        hashes.append({"path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size, "sha256": b})
    dump(ROOT / "resources/input_manifest.json", hashes)
    for path in (PROJECT / "ROADMAP.md", PROJECT / "roadmap_review/data_audit.json"):
        if path.exists():
            shutil.copy2(path, ROOT / "resources/reference" / path.name)
    docs = list(PROJECT.glob("*.docx"))
    if docs:
        doc = docs[0]
        shutil.copy2(doc, ROOT / "resources/reference" / doc.name)
        xml = ET.fromstring(zipfile.ZipFile(doc).read("word/document.xml"))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        text = "\n".join("".join(n.text or "" for n in p.iter(ns + "t")) for p in xml.iter(ns + "p"))
        (ROOT / "resources/reference/题目原文.txt").write_text(text, encoding="utf-8")
    rows = list(openpyxl.load_workbook(target / "label-100.xlsx", read_only=True, data_only=True)
                .active.iter_rows(values_only=True))
    records = [dict(zip(rows[0], r)) for r in rows[1:] if any(x is not None for x in r)]
    train_ids = json.loads((ROOT / "resources/reference/data_audit.json").read_text())["q1"][
        "usable_train_ids_for_mapping_development"]
    manifest = []
    for row in records:
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        path = target / video_id / (clip_id + ".mp4")
        meta = probe(path)
        streams = {s["codec_type"]: s for s in meta["streams"]}
        if not {"audio", "video"} <= streams.keys():
            raise ValueError(f"Missing stream: {path}")
        sid = f"{video_id}$_${clip_id}"
        origin = float(meta["format"].get("start_time", 0))
        duration = float(meta["format"]["duration"])
        frame_data = probe(path, frames=True)["frames"]
        pts = [float(f["best_effort_timestamp_time"]) for f in frame_data]
        if not pts or any(b <= a for a, b in zip(pts, pts[1:])):
            raise ValueError(f"Non-increasing or absent PTS: {sid}")
        dump(ROOT / "intermediate/probes" / (sid + ".json"),
             {"metadata": meta, "frames": frame_data, "origin_s": origin})
        manifest.append({
            "sample_id": sid, "video_id": video_id, "clip_id": clip_id,
            "source_file": str(path.relative_to(ROOT)), "text": row["text"],
            "duration_s": duration, "origin_s": origin,
            "audio_start_s": float(streams["audio"].get("start_time", origin)) - origin,
            "video_start_s": float(streams["video"].get("start_time", origin)) - origin,
            "frame_count": len(pts), "width": streams["video"]["width"],
            "height": streams["video"]["height"],
            "train_overlap": sid in train_ids,
        })
    ids = [r["sample_id"] for r in manifest]
    actual = {f"{p.parent.name}$_${p.stem}" for p in target.rglob("*.mp4")}
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != actual:
        raise ValueError("100-item coverage/pairing failed")
    dump(ROOT / "resources/manifest.json", manifest)
    eligible = sorted((r for r in manifest if r["train_overlap"]), key=lambda r: r["duration_s"])
    pilot = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
    dump(ROOT / "configs/pilot.json", [r["sample_id"] for r in pilot])
    summary = {
        "samples": len(manifest), "video_bytes": sum(h["bytes"] for h in hashes if h["path"].endswith(".mp4")),
        "source_copies_hash_verified": len(hashes), "train_overlap": len(eligible),
        "total_duration_s": sum(r["duration_s"] for r in manifest),
        "duration_min_s": min(r["duration_s"] for r in manifest),
        "duration_max_s": max(r["duration_s"] for r in manifest),
        "pilots": [{"sample_id": r["sample_id"], "duration_s": r["duration_s"]} for r in pilot],
        "labels_policy": "Original spreadsheet copied unchanged; labels excluded from extraction manifest.",
    }
    dump(ROOT / "reports/input_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
