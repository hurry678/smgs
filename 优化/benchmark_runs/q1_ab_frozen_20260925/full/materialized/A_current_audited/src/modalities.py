"""Frozen text encoding and timestamped, interpretable audio/face measurements."""
import json
import subprocess
import numpy as np
from prepare import ROOT, dump


def decode_audio(row, config, source_sha256):
    import soundfile as sf
    path = ROOT / "intermediate/audio" / (row["sample_id"] + ".wav")
    metadata_path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {"source_sha256": source_sha256, "sample_rate": config["sample_rate"],
                "channels": 1, "codec": "pcm_s16le"}
    actual = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if not path.exists() or actual != expected:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / row["source_file"]),
                        "-map", "0:a:0", "-ac", "1", "-ar", str(config["sample_rate"]),
                        "-c:a", "pcm_s16le", str(path)], check=True)
        dump(metadata_path, expected)
    y, sr = sf.read(path, dtype="float32")
    if sr != config["sample_rate"] or y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Invalid decoded waveform")
    return y, path


def audio_features(y, offset, c):
    import librosa
    sr, hop, win = c["sample_rate"], c["hop_length"], c["frame_length"]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=c["n_fft"], hop_length=hop, win_length=win,
        window=c["window"], center=c["center"], power=c["power"],
        n_mels=c["n_mels"], fmin=c["fmin"], fmax=c["fmax"],
        htk=c["mel_htk"], norm=c["mel_norm"])
    db = librosa.power_to_db(mel, ref=c["db_ref"], amin=c["db_amin"], top_db=c["top_db"])
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=c["n_mfcc"], dct_type=c["dct_type"],
                               norm=c["mfcc_norm"], lifter=c["lifter"]).T
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=False).T
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop,
                                           center=False, threshold=0.0, zero_pos=True).T
    values = np.concatenate([mfcc, np.log(np.maximum(rms, c["rms_epsilon"])), zcr], axis=1)
    starts = np.arange(len(values)) * hop / sr + offset
    f0, voiced, prob = librosa.pyin(
        y, fmin=c["pyin_fmin"], fmax=c["pyin_fmax"], sr=sr,
        frame_length=c["pyin_frame_length"], hop_length=hop,
        center=c["pyin_center"], fill_na=np.nan,
        n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2,
        resolution=0.1, max_transition_rate=35.92, switch_prob=0.01,
        no_trough_prob=0.01)
    fstarts = np.arange(len(f0)) * hop / sr + offset
    valid_f0 = np.isfinite(f0) & voiced
    return {
        "values": values.astype(np.float32),
        "intervals": np.column_stack([starts, starts + win / sr]),
        "f0": np.where(valid_f0, f0, 0).astype(np.float32),
        "f0_valid": valid_f0,
        "f0_intervals": np.column_stack([fstarts, fstarts + c["pyin_frame_length"] / sr]),
        "voiced_probability": prob.astype(np.float32),
        "pitch_boundary_hits": np.array(int((valid_f0 & ((f0 < c["pyin_fmin"] * 1.01) |
                                           (f0 > c["pyin_fmax"] / 1.01))).sum())),
    }


class TextEncoder:
    def __init__(self, config):
        import torch
        from transformers import BertModel, BertTokenizerFast
        self.torch = torch
        self.c = config
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(ROOT / config["model_dir"]), do_lower_case=True, local_files_only=True)
        self.model = BertModel.from_pretrained(str(ROOT / config["model_dir"]), local_files_only=True)
        self.model.eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != 128:
            raise ValueError("Unexpected BERT dimension")

    def encode(self, text):
        t = self.tokenizer(
            text, truncation=True, max_length=self.c["max_length"], stride=self.c["stride"],
            return_overflowing_tokens=True, return_offsets_mapping=True,
            return_special_tokens_mask=True, padding=False)
        positions, ids, embeddings, seen = [], [], [], set()
        with self.torch.inference_mode():
            for w in range(len(t["input_ids"])):
                kwargs = {k: self.torch.tensor([t[k][w]]) for k in (
                    "input_ids", "attention_mask", "token_type_ids")}
                h = self.model(**kwargs).last_hidden_state[0].cpu().numpy()
                for i, (a, b) in enumerate(t["offset_mapping"][w]):
                    token = int(t["input_ids"][w][i])
                    key = (a, b, token)
                    if t["special_tokens_mask"][w][i] or a == b or key in seen:
                        continue
                    seen.add(key)
                    positions.append([a, b])
                    ids.append(token)
                    embeddings.append(h[i])
        order = np.argsort([p[0] for p in positions])
        if not len(order):
            raise ValueError("No semantic text tokens")
        return {"offsets": np.asarray(positions, dtype=np.int32)[order],
                "ids": np.asarray(ids, dtype=np.int32)[order],
                "embeddings": np.asarray(embeddings, dtype=np.float32)[order],
                "windows": np.array(len(t["input_ids"]))}


def face_geometry(landmarks, width, height, c):
    p = np.array([[v.x * width, v.y * height] for v in landmarks])
    ix = c["indices"]
    dist = lambda a, b: float(np.linalg.norm(p[a] - p[b]))
    def eye(indices):
        a, b, cc, d, e, f = indices
        return (dist(b, f) + dist(cc, e)) / (2 * max(dist(a, d), 1e-6))
    scale = max(dist(*ix["scale"]), 1e-6)
    upper, lower, left, right = ix["mouth"]
    # Projection onto the face's in-plane vertical axis reduces camera-roll sensitivity.
    horizontal = (p[ix["scale"][1]] - p[ix["scale"][0]]) / scale
    vertical = np.array([-horizontal[1], horizontal[0]])
    if vertical[1] < 0:
        vertical *= -1
    nose = p[ix["nose"]]
    feature = np.array([
        eye(ix["left_eye"]), eye(ix["right_eye"]),
        dist(upper, lower) / scale, dist(left, right) / scale,
        np.dot(p[left] - nose, vertical) / scale,
        np.dot(p[right] - nose, vertical) / scale,
    ], dtype=np.float32)
    valid = (
        np.isfinite(feature).all()
        and scale >= c["minimum_interocular_px"]
        and np.all(feature[:2] <= c["maximum_eye_aspect_ratio"])
        and feature[2] <= c["maximum_mouth_open_ratio"]
        and feature[3] <= c["maximum_mouth_width_ratio"]
        and np.all(np.abs(feature[4:]) <= c["maximum_mouth_corner_height_ratio"])
    )
    return feature, bool(valid), scale


def vision_features(row, c):
    from dataclasses import replace
    from types import SimpleNamespace
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    data = json.loads((ROOT / "intermediate/probes" / (row["sample_id"] + ".json")).read_text())
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in data["frames"]])
    times = pts - row["origin_s"]
    chosen, next_time = [], times[0]
    for i, t in enumerate(times):
        if t + 1e-9 >= next_time:
            chosen.append(i)
            next_time = t + c["sample_period_s"]
    chosen_set = set(chosen)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(ROOT / c["model_path"]),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_faces=c["num_faces"],
        min_face_detection_confidence=c["min_face_detection_confidence"],
        min_face_presence_confidence=c["min_face_presence_confidence"],
        min_tracking_confidence=c["min_tracking_confidence"],
        output_face_blendshapes=False, output_facial_transformation_matrixes=False)
    values, raw_values, masks, counts, ambiguous, boxes = [], [], [], [], [], []
    crop_used = []
    geometry_rejected = []
    face_scales = []
    previous = None
    cap = cv2.VideoCapture(str(ROOT / row["source_file"]))
    frame_index = 0
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector, \
                vision.FaceLandmarker.create_from_options(
                    replace(options, running_mode=vision.RunningMode.IMAGE)) as crop_detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index in chosen_set:
                    h, w = frame.shape[:2]
                    if w > 640:
                        frame = cv2.resize(frame, (640, round(h * 640 / w)))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, round((times[frame_index] - times[0]) * 1000))
                    faces = result.face_landmarks
                    used_crop = False
                    if not faces and c.get("crop_fallback", False):
                        rh, rw = rgb.shape[:2]
                        fraction = c["crop_fraction"]
                        ch, cw = round(rh * fraction), round(rw * fraction)
                        proposals = []
                        for fx, fy in ((.5, .5), (0, 0), (1, 0), (0, 1), (1, 1)):
                            left, top = round((rw-cw)*fx), round((rh-ch)*fy)
                            cropped = np.ascontiguousarray(rgb[top:top+ch, left:left+cw])
                            result_crop = crop_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped))
                            for face in result_crop.face_landmarks:
                                mapped = [SimpleNamespace(x=(p.x*cw+left)/rw, y=(p.y*ch+top)/rh) for p in face]
                                center = np.mean([[p.x, p.y] for p in mapped], axis=0)
                                if not any(np.linalg.norm(center - prev) < .05 for prev, _ in proposals):
                                    proposals.append((center, mapped))
                        faces = [f for _, f in proposals]
                        used_crop = bool(faces)
                    crop_used.append(used_crop)
                    counts.append(len(faces))
                    if faces:
                        bounds = np.array([[min(p.x for p in f), min(p.y for p in f),
                                            max(p.x for p in f), max(p.y for p in f)] for f in faces])
                        centers = (bounds[:, :2] + bounds[:, 2:]) / 2
                        area = np.prod(bounds[:, 2:] - bounds[:, :2], axis=1)
                        order = (np.argsort(-area) if previous is None else
                                 np.argsort(np.linalg.norm(centers - previous, axis=1)))
                        which = int(order[0])
                        feature = np.zeros(6, np.float32)
                        geometry_valid, scale = False, 0.0
                        for candidate in order:
                            candidate_feature, candidate_valid, candidate_scale = face_geometry(
                                faces[int(candidate)], rgb.shape[1], rgb.shape[0], c)
                            if candidate_valid:
                                which = int(candidate)
                                feature, geometry_valid, scale = (
                                    candidate_feature, candidate_valid, candidate_scale)
                                break
                            if candidate == order[0]:
                                feature, scale = candidate_feature, candidate_scale
                        jump = previous is not None and np.linalg.norm(centers[which] - previous) > c[
                            "max_center_jump_normalized"]
                        previous = centers[which]
                        raw_values.append(feature)
                        values.append(feature if geometry_valid else np.zeros(6, np.float32))
                        masks.append(np.full(6, geometry_valid))
                        geometry_rejected.append(not geometry_valid)
                        face_scales.append(scale)
                        ambiguous.append(bool(len(faces) > 1 or jump or not geometry_valid))
                        boxes.append(bounds[which])
                    else:
                        values.append(np.zeros(6, np.float32))
                        raw_values.append(np.zeros(6, np.float32))
                        masks.append(np.zeros(6, bool))
                        geometry_rejected.append(False)
                        face_scales.append(0.0)
                        ambiguous.append(False)
                        boxes.append(np.zeros(4))
                frame_index += 1
    finally:
        cap.release()
    if frame_index != len(pts) or len(values) != len(chosen):
        raise ValueError(f"Decoded frame/PTS mismatch: {frame_index}/{len(pts)}")
    return {"values": np.asarray(values), "raw_values": np.asarray(raw_values),
            "valid": np.asarray(masks), "geometry_rejected": np.asarray(geometry_rejected),
            "face_scale_px": np.asarray(face_scales, dtype=np.float32),
            "crop_fallback_used": np.asarray(crop_used),
            "frame_index": np.asarray(chosen, dtype=np.int32), "pts": pts[chosen],
            "time": times[chosen], "faces": np.asarray(counts, dtype=np.int16),
            "ambiguous": np.asarray(ambiguous), "boxes": np.asarray(boxes, dtype=np.float32)}
