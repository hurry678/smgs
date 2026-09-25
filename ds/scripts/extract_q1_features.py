#!/usr/bin/env python3
"""Q1 reproducible multimodal feature extraction and word-level alignment.

Server-oriented implementation:
- BERT (`bert-base-uncased`) produces word-level 768-D contextual features.
- FFmpeg + NumPy/SciPy produce 16 kHz, 25 ms / 10 ms acoustic features.
- OpenCV Haar cascades produce masked 5 fps face-region visual features.
- A monotonic energy/pause constrained aligner maps transcript words to time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d, label
from scipy.signal import find_peaks

TEXT_DIM = 768
AUDIO_DIM = 40
VISION_DIM = 90
AUDIO_SR = 16000
AUDIO_NFFT = 400
AUDIO_HOP = 160
AUDIO_NMELS = 20
AUDIO_NMFCC = 13
VISION_FPS = 5.0
VISION_FACE_SIZE = 48
VISION_GRID = 6
ALIGNMENT_BACKEND = "energy_pause_dp_v1"

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of",
    "in", "on", "at", "for", "from", "with", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "have", "has",
    "had", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "not", "no", "so", "just", "very", "really",
}
NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "can't", "don't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't"}
POSITIVE_LEX = {"good", "great", "happy", "love", "like", "nice", "best", "better", "excellent", "wonderful", "amazing", "positive", "success", "succeed", "win", "winning", "beautiful", "fun", "enjoy", "glad", "proud", "hope", "hopeful"}
NEGATIVE_LEX = {"bad", "sad", "angry", "hate", "terrible", "worst", "worse", "awful", "negative", "fail", "failed", "failure", "pain", "afraid", "fear", "problem", "problems", "sorry", "unhappy", "disappointed", "disappointing", "hurt", "loss", "lose", "losing"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def tool_version(executable: str) -> str:
    result = run_command([executable, "-version"])
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_sample_key(sample_id: str) -> str:
    return sample_id.replace("$_$", "__").replace("$", "_")


def stable_hash(token: str, salt: str = "") -> tuple[int, int]:
    raw = hashlib.blake2b((salt + token).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(raw, "little", signed=False)
    return value, 1 if (value & 1) else -1


def add_hashed(vector: np.ndarray, token: str, dim: int, offset: int, weight: float, salt: str) -> None:
    if not token:
        return
    value, sign = stable_hash(token, salt)
    vector[offset + (value % dim)] += sign * weight


def char_ngrams(token: str, minimum: int = 3, maximum: int = 5) -> list[str]:
    padded = f"^{token}$"
    grams: list[str] = []
    for n in range(minimum, maximum + 1):
        if len(padded) >= n:
            grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def tokenize_with_spans(text: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token).lower()
        if not cleaned:
            continue
        trailing = text[match.end():match.end() + 1] if match.end() < len(text) else ""
        tokens.append({
            "token": token,
            "clean": cleaned,
            "start_char": match.start(),
            "end_char": match.end(),
            "trailing_punct": trailing in {".", "!", "?", ",", ";", ":"},
        })
    return tokens


def hash_text_features(tokens: list[dict[str, Any]], dim: int = TEXT_DIM) -> np.ndarray:
    result = np.zeros((len(tokens), dim), dtype=np.float32)
    if not tokens:
        return result
    for i, item in enumerate(tokens):
        token = item["clean"]
        vector = result[i]
        add_hashed(vector, token, 64, 0, 1.0, "word")
        for gram in char_ngrams(token):
            add_hashed(vector, gram, 96, 64, 0.35, "char")
        if i > 0:
            add_hashed(vector, tokens[i - 1]["clean"], 48, 160, 0.30, "prev")
        if i + 1 < len(tokens):
            add_hashed(vector, tokens[i + 1]["clean"], 48, 208, 0.30, "next")
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            vector /= norm
        original = item["token"]
        letters = [c for c in original if c.isalpha()]
        vowel_count = sum(c.lower() in "aeiou" for c in letters)
        upper_count = sum(c.isupper() for c in original)
        digit_count = sum(c.isdigit() for c in original)
        length = max(1, len(original))
        position = i / max(1, len(tokens) - 1)
        tail = np.zeros(16, dtype=np.float32)
        tail[0] = math.log1p(len(original)) / 5.0
        tail[1] = vowel_count / length
        tail[2] = digit_count / length
        tail[3] = upper_count / length
        tail[4] = float("'" in original or "’" in original)
        tail[5] = float(token in STOPWORDS)
        tail[6] = position
        tail[7] = 1.0 - position
        tail[8] = float(i > 0)
        tail[9] = float(i + 1 < len(tokens))
        tail[10] = float(token in NEGATIONS)
        tail[11] = float(token in POSITIVE_LEX)
        tail[12] = float(token in NEGATIVE_LEX)
        tail[13] = float(item["trailing_punct"])
        tail[14] = min(1.0, max(0.0, (len(original) - 4.0) / 8.0))
        tail[15] = float(original[:1].isupper()) if original else 0.0
        vector[256:272] = tail
    return result


def load_bert_backend(model_name: str, device_name: str) -> tuple[Callable[[str, list[dict[str, Any]]], tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval().to(device)

    def extract(text: str, tokens: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        if not tokens:
            return np.zeros((0, TEXT_DIM), dtype=np.float32), {"truncated": False, "fallback_words": 0}
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        input_length = int(encoded["input_ids"].shape[1])
        if input_length > 512:
            raise ValueError(f"BERT sequence length {input_length} exceeds 512; long-text chunking is required")
        offsets = encoded.pop("offset_mapping")[0].cpu().numpy()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
        result = np.zeros((len(tokens), TEXT_DIM), dtype=np.float32)
        fallback_words = 0
        for i, item in enumerate(tokens):
            left, right = int(item["start_char"]), int(item["end_char"])
            selected = [
                j for j, (a, b) in enumerate(offsets)
                if b > a and a < right and b > left and not (a == 0 and b == 0)
            ]
            if selected:
                result[i] = hidden[selected].mean(axis=0)
                norm = float(np.linalg.norm(result[i]))
                if norm > 1e-8:
                    result[i] /= norm
            else:
                fallback = hash_text_features([item], dim=TEXT_DIM)[0]
                result[i] = fallback
                fallback_words += 1
        return result, {"truncated": False, "fallback_words": fallback_words, "input_tokens": input_length}

    return extract, {
        "backend": "bert",
        "model_name": model_name,
        "device": str(device),
        "feature_dim": TEXT_DIM,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
    }


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 50.0, fmax: float = 7600.0) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points))
    bank = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        left_slope = (freqs - left) / max(center - left, 1e-8)
        right_slope = (right - freqs) / max(right - center, 1e-8)
        bank[m - 1] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-8)
    return bank


MEL_BANK = mel_filterbank(AUDIO_SR, AUDIO_NFFT, AUDIO_NMELS)


def decode_audio(video_path: Path, sr: int = AUDIO_SR) -> tuple[np.ndarray, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="q1_audio_") as temp_dir:
        output = Path(temp_dir) / "audio.wav"
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(output),
        ]
        result = run_command(command)
        if result.returncode != 0 or not output.exists():
            raise RuntimeError(result.stderr.strip() or "ffmpeg audio decode failed")
        read_sr, data = wavfile.read(output)
        if read_sr != sr:
            raise RuntimeError(f"unexpected audio sample rate: {read_sr} != {sr}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = max(abs(info.min), abs(info.max))
            waveform = data.astype(np.float32) / float(scale)
        else:
            waveform = data.astype(np.float32)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        return waveform, {"sample_rate": sr, "samples": int(len(waveform)), "duration_sec": len(waveform) / sr}


def frame_audio(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) < AUDIO_NFFT:
        waveform = np.pad(waveform, (0, AUDIO_NFFT - len(waveform)))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, AUDIO_NFFT)[::AUDIO_HOP]
    window = np.hanning(AUDIO_NFFT).astype(np.float32)
    return (frames * window[None, :]).astype(np.float32)


def autocorrelation_pitch(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = frames.shape[1]
    fft_size = 1 << int(math.ceil(math.log2(max(2 * n, 2))))
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size, axis=1)[:, :n]
    autocorr = autocorr / np.maximum(autocorr[:, :1], 1e-8)
    min_lag = max(1, int(AUDIO_SR / 400.0))
    max_lag = min(n - 1, int(AUDIO_SR / 70.0))
    region = autocorr[:, min_lag:max_lag + 1]
    peak_lags = np.argmax(region, axis=1) + min_lag
    peak_values = region[np.arange(region.shape[0]), peak_lags - min_lag]
    voiced_probability = np.clip((peak_values - 0.20) / 0.50, 0.0, 1.0).astype(np.float32)
    f0 = AUDIO_SR / np.maximum(peak_lags, 1)
    f0_norm = np.log(np.clip(f0, 70.0, 400.0) / 70.0) / np.log(400.0 / 70.0)
    f0_norm = np.where(voiced_probability >= 0.35, f0_norm, np.nan)
    return voiced_probability, f0_norm.astype(np.float32)


def extract_audio_features(video_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    waveform, metadata = decode_audio(video_path)
    frames = frame_audio(waveform)
    if len(frames) == 0:
        return (np.zeros((0, AUDIO_DIM), dtype=np.float32), np.zeros((0, AUDIO_DIM), dtype=bool), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool), metadata)
    spectrum = np.fft.rfft(frames, n=AUDIO_NFFT, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = magnitude ** 2
    eps = 1e-10
    freqs = np.fft.rfftfreq(AUDIO_NFFT, 1.0 / AUDIO_SR).astype(np.float32)
    sum_mag = np.maximum(magnitude.sum(axis=1), eps)
    centroid = (magnitude * freqs[None, :]).sum(axis=1) / sum_mag
    bandwidth = np.sqrt((magnitude * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / sum_mag)
    cumulative = np.cumsum(power, axis=1)
    rolloff_index = np.argmax(cumulative >= 0.85 * np.maximum(cumulative[:, -1:], eps), axis=1)
    rolloff = freqs[rolloff_index]
    flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / np.maximum(np.mean(power, axis=1), eps)
    spectral_flux = np.zeros(len(frames), dtype=np.float32)
    if len(frames) > 1:
        spectral_flux[1:] = np.sqrt(np.mean((magnitude[1:] - magnitude[:-1]) ** 2, axis=1))
    energy_db = 10.0 * np.log10(np.mean(power, axis=1) + eps)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    voiced_probability, f0_norm = autocorrelation_pitch(frames)
    mel_energy = np.maximum(power @ MEL_BANK.T, eps)
    log_mel = np.log(mel_energy)
    dct = np.sqrt(2.0 / AUDIO_NMELS) * np.cos(
        np.pi / AUDIO_NMELS * (np.arange(AUDIO_NMFCC)[:, None] + 0.5) * (np.arange(AUDIO_NMELS)[None, :] + 0.5)
    )
    dct[0] *= 1.0 / math.sqrt(2.0)
    mfcc = log_mel @ dct.T
    delta = np.gradient(mfcc, axis=0) if len(mfcc) > 1 else np.zeros_like(mfcc)
    features = np.column_stack([
        energy_db,
        zcr,
        centroid / (AUDIO_SR / 2.0),
        bandwidth / (AUDIO_SR / 2.0),
        rolloff / (AUDIO_SR / 2.0),
        np.log10(flatness + eps) / 5.0,
        spectral_flux / np.maximum(np.max(spectral_flux), eps),
        voiced_probability,
        f0_norm,
        mfcc[:, :AUDIO_NMFCC],
        delta[:, :10],
        log_mel[:, :8],
    ]).astype(np.float32)
    if features.shape[1] != AUDIO_DIM:
        raise RuntimeError(f"audio feature dimension mismatch: {features.shape[1]} != {AUDIO_DIM}")
    valid = np.isfinite(features)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    times = ((np.arange(len(features)) * AUDIO_HOP) + AUDIO_NFFT / 2.0) / AUDIO_SR
    metadata.update({"frame_count": int(len(features)), "frame_window_sec": AUDIO_NFFT / AUDIO_SR, "frame_hop_sec": AUDIO_HOP / AUDIO_SR, "feature_dim": AUDIO_DIM})
    return features, valid, times.astype(np.float32), voiced_probability >= 0.35, metadata


def smooth_contiguous_valid(features: np.ndarray, valid: np.ndarray, sigma: float = 0.75) -> tuple[np.ndarray, dict[str, Any]]:
    """Weakly smooth each contiguous all-dimension-valid run without crossing masks."""
    if features.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if valid.ndim == 1 and valid.shape[0] == features.shape[0]:
        valid_matrix = np.repeat(valid[:, None], features.shape[1], axis=1)
        frame_valid = valid.astype(bool)
    elif valid.shape == features.shape:
        valid_matrix = valid.astype(bool)
        frame_valid = np.all(valid_matrix, axis=1)
    else:
        raise ValueError("valid must be [T,D] matching features or [T] frame-level mask")
    output = np.nan_to_num(features.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    if len(features) == 0 or sigma <= 0:
        return output, {"enabled": bool(sigma > 0), "sigma": float(sigma), "smoothed_frames": 0, "runs": 0}
    row_valid = frame_valid
    components, count = label(row_valid)
    smoothed_frames = 0
    runs = 0
    for component_id in range(1, int(count) + 1):
        indices = np.flatnonzero(components == component_id)
        if len(indices) < 3:
            continue
        segment = output[indices]
        output[indices] = gaussian_filter1d(segment, sigma=float(sigma), axis=0, mode="nearest")
        smoothed_frames += int(len(indices))
        runs += 1
    return output.astype(np.float32), {
        "enabled": True,
        "method": "gaussian_filter1d",
        "sigma": float(sigma),
        "boundary_rule": "within_contiguous_all_dim_valid_runs_only",
        "smoothed_frames": smoothed_frames,
        "runs": runs,
    }


def _require_cv2():
    import cv2
    return cv2


def _block_mean_std(image: np.ndarray, grid: int = 3) -> np.ndarray:
    height, width = image.shape
    values: list[np.ndarray] = []
    for row in range(grid):
        for col in range(grid):
            y0 = int(round(row * height / grid))
            y1 = int(round((row + 1) * height / grid))
            x0 = int(round(col * width / grid))
            x1 = int(round((col + 1) * width / grid))
            block = image[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
            values.append(np.asarray([float(block.mean()), float(block.std())], dtype=np.float32))
    return np.concatenate(values).astype(np.float32)


def _manual_hog(image: np.ndarray, cells: int = 2, bins: int = 9) -> np.ndarray:
    cv2 = _require_cv2()
    resized = cv2.resize(image, (VISION_FACE_SIZE, VISION_FACE_SIZE), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bin_index = np.minimum((angle / np.pi * bins).astype(np.int32), bins - 1)
    output = np.zeros((cells, cells, bins), dtype=np.float32)
    cell_size = VISION_FACE_SIZE // cells
    for row in range(cells):
        for col in range(cells):
            ys, ye = row * cell_size, (row + 1) * cell_size
            xs, xe = col * cell_size, (col + 1) * cell_size
            selected_bins = bin_index[ys:ye, xs:xe].ravel()
            selected_mag = magnitude[ys:ye, xs:xe].ravel()
            output[row, col] = np.bincount(selected_bins, weights=selected_mag, minlength=bins)[:bins]
            norm = float(np.linalg.norm(output[row, col]))
            if norm > 1e-8:
                output[row, col] /= norm
    return output.reshape(-1).astype(np.float32)


def _visual_feature_vector(frame: np.ndarray, face_box: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    cv2 = _require_cv2()
    x, y, width, height = face_box
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y + height, x:x + width]
    if roi.size == 0:
        raise ValueError("empty face ROI")
    roi = cv2.resize(roi, (VISION_FACE_SIZE, VISION_FACE_SIZE), interpolation=cv2.INTER_AREA)
    roi = cv2.equalizeHist(roi)

    x_norm = x / max(frame_w, 1)
    y_norm = y / max(frame_h, 1)
    w_norm = width / max(frame_w, 1)
    h_norm = height / max(frame_h, 1)
    cx_norm = (x + width / 2.0) / max(frame_w, 1)
    cy_norm = (y + height / 2.0) / max(frame_h, 1)
    area_norm = (width * height) / max(frame_w * frame_h, 1)
    aspect = width / max(height, 1)
    geometry = np.asarray([x_norm, y_norm, w_norm, h_norm, cx_norm, cy_norm, area_norm, math.log(max(aspect, 1e-6))], dtype=np.float32)

    moments = cv2.moments(roi)
    hu = cv2.HuMoments(moments).reshape(-1).astype(np.float32)
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    if hu.shape[0] != 7:
        raise RuntimeError(f"unexpected Hu moment dimension: {hu.shape[0]}")

    blocks = _block_mean_std(roi, grid=3)
    histogram = cv2.calcHist([roi], [0], None, [16], [0, 256]).reshape(-1).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1e-8)
    hog = _manual_hog(roi, cells=2, bins=9)
    laplacian = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(laplacian)
    lap_mean = float(abs_lap.mean()) / 255.0
    lap_std = float(abs_lap.std()) / 255.0
    lap_max = float(abs_lap.max()) / 255.0
    lap_active = float(np.mean(abs_lap > 20.0))
    lap_entropy = float(-np.sum((histogram + 1e-8) * np.log(histogram + 1e-8)) / math.log(16.0))
    texture = np.asarray([lap_mean, lap_std, lap_max, lap_active, lap_entropy], dtype=np.float32)

    vector = np.concatenate([geometry, hu, blocks, histogram, hog, texture]).astype(np.float32)
    if vector.shape[0] != VISION_DIM:
        raise RuntimeError(f"visual feature dimension mismatch: {vector.shape[0]} != {VISION_DIM}")
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    details = {
        "face_box_xywh": [int(x), int(y), int(width), int(height)],
        "face_area_ratio": float(area_norm),
        "detector_score_proxy": 1.0,
    }
    return vector, details


def extract_visual_features(video_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(native_fps) or native_fps <= 0:
        native_fps = 30.0
    sample_step = max(1, int(round(native_fps / VISION_FPS)))
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        capture.release()
        raise RuntimeError(f"failed to load Haar cascade: {cascade_path}")

    features: list[np.ndarray] = []
    valid: list[bool] = []
    times: list[float] = []
    boxes: list[list[int]] = []
    frame_index = 0
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames += 1
        if frame_index % sample_step != 0:
            frame_index += 1
            continue
        timestamp = frame_index / native_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        detected = detector.detectMultiScale(
            equalized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(detected) > 0:
            box = max((tuple(int(v) for v in item) for item in detected), key=lambda item: item[2] * item[3])
            vector, _ = _visual_feature_vector(frame, box)
            features.append(vector)
            valid.append(True)
            boxes.append([int(v) for v in box])
        else:
            features.append(np.zeros(VISION_DIM, dtype=np.float32))
            valid.append(False)
            boxes.append([])
        times.append(timestamp)
        frame_index += 1
    capture.release()

    feature_array = np.asarray(features, dtype=np.float32).reshape(-1, VISION_DIM)
    valid_array = np.asarray(valid, dtype=bool)
    time_array = np.asarray(times, dtype=np.float32)
    metadata = {
        "backend": "opencv_haar_face_region",
        "native_fps": native_fps,
        "target_fps": VISION_FPS,
        "sample_step_frames": sample_step,
        "decoded_frames": decoded_frames,
        "sampled_frames": int(len(feature_array)),
        "detected_frames": int(valid_array.sum()),
        "detection_rate": float(valid_array.mean()) if len(valid_array) else 0.0,
        "cascade": str(cascade_path),
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "min_size": [30, 30],
        "face_roi_size": [VISION_FACE_SIZE, VISION_FACE_SIZE],
        "feature_dim": VISION_DIM,
        "face_boxes": boxes,
        "feature_blocks": {
            "geometry": 8,
            "hu_moments": 7,
            "roi_3x3_mean_std": 18,
            "gray_hist_16": 16,
            "hog_2x2x9": 36,
            "laplacian_texture": 5,
        },
    }
    return feature_array, valid_array, time_array, metadata

def _robust_unit_scale(values: np.ndarray, low: float = 10.0, high: float = 90.0) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0:
        return values
    left, right = np.percentile(values, [low, high])
    if right - left <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - left) / (right - left), 0.0, 1.0).astype(np.float32)


def _aggregate_groups(values: np.ndarray, valid: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
    output = np.zeros(int(math.ceil(len(values) / group_size)), dtype=np.float32)
    valid_output = np.zeros_like(output, dtype=bool)
    for group_index, start in enumerate(range(0, len(values), group_size)):
        stop = min(len(values), start + group_size)
        selected = valid[start:stop]
        if np.any(selected):
            output[group_index] = float(values[start:stop][selected].mean())
            valid_output[group_index] = True
    return output, valid_output


def estimate_speech_activity(
    audio_features: np.ndarray,
    audio_valid: np.ndarray,
    voiced_flag: np.ndarray,
    working_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate a speech score at frame level and at a compact alignment rate."""
    frame_valid = np.all(audio_valid, axis=1) if len(audio_valid) else np.zeros((0,), dtype=bool)
    if len(audio_features) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        empty_bool = np.zeros((0,), dtype=bool)
        return empty, empty_bool, empty, empty_bool, {"working_hz": working_hz, "group_size": 1}

    energy_score = _robust_unit_scale(audio_features[:, 0])
    flux_score = _robust_unit_scale(audio_features[:, 6])
    voiced_probability = np.clip(audio_features[:, 7], 0.0, 1.0)
    voiced_score = np.maximum(voiced_probability, voiced_flag.astype(np.float32))
    score = (0.58 * energy_score + 0.27 * voiced_score + 0.15 * flux_score).astype(np.float32)
    score[~frame_valid] = 0.0

    if np.any(frame_valid):
        activity_threshold = max(0.24, float(np.percentile(score[frame_valid], 35.0)))
    else:
        activity_threshold = 0.24
    active_raw = (score >= activity_threshold) | voiced_flag | (energy_score >= 0.45)
    if np.any(active_raw):
        active = binary_closing(active_raw, structure=np.ones(3, dtype=bool))
        active = binary_opening(active, structure=np.ones(3, dtype=bool))
        if not np.any(active):
            active = active_raw
    else:
        active = active_raw
    activity = active.astype(np.float32)
    combined_score = np.clip(0.78 * score + 0.22 * activity, 0.0, 1.0).astype(np.float32)

    # Fill only isolated invalid gaps in the score for aggregation, retaining the original mask.
    score_clean = combined_score.copy()
    valid_components, component_count = label(frame_valid)
    for component_id in range(1, int(component_count) + 1):
        indices = np.flatnonzero(valid_components == component_id)
        if len(indices) >= 3:
            score_clean[indices] = gaussian_filter1d(score_clean[indices], sigma=0.75, mode="nearest")
    score_clean[~frame_valid] = 0.0

    frame_hop_sec = AUDIO_HOP / AUDIO_SR
    group_size = max(1, int(round((1.0 / working_hz) / frame_hop_sec)))
    score_working, valid_working = _aggregate_groups(score_clean, frame_valid, group_size)
    activity_working, _ = _aggregate_groups(activity, frame_valid, group_size)
    metadata = {
        "working_hz": working_hz,
        "group_size": group_size,
        "energy_percentiles": [10, 90],
        "active_threshold": float(activity_threshold),
        "closing_frames": 3,
        "opening_frames": 3,
        "score_weights": {"energy": 0.58, "voiced": 0.27, "spectral_flux": 0.15},
    }
    return score_clean, frame_valid, score_working, valid_working, metadata | {"activity_working": activity_working}


def align_words_to_audio(
    tokens: list[dict[str, Any]],
    audio_features: np.ndarray,
    audio_valid: np.ndarray,
    audio_times: np.ndarray,
    voiced_flag: np.ndarray,
    audio_duration_sec: float,
    working_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Monotonic non-overlapping word alignment using energy/pause-constrained DP."""
    word_count = len(tokens)
    if word_count == 0:
        empty_float = np.zeros((0,), dtype=np.float32)
        empty_bool = np.zeros((0,), dtype=bool)
        return empty_float, empty_float, empty_float, empty_bool, {
            "backend": ALIGNMENT_BACKEND,
            "word_count": 0,
            "assigned_word_count": 0,
            "coverage": 1.0,
            "fallback_used": False,
        }

    frame_count = len(audio_features)
    if frame_count == 0:
        edges = np.linspace(0.0, max(audio_duration_sec, 0.0), word_count + 1, dtype=np.float32)
        starts, ends = edges[:-1], edges[1:]
        confidence = np.zeros(word_count, dtype=np.float32)
        fallback = np.ones(word_count, dtype=bool)
        return starts, ends, confidence, fallback, {
            "backend": ALIGNMENT_BACKEND,
            "fallback": "uniform_no_audio_frames",
            "word_count": word_count,
            "assigned_word_count": word_count,
            "coverage": 1.0,
            "mean_confidence": 0.0,
            "fallback_used": True,
        }

    score_frame, valid_frame, score_work, valid_work, activity_meta = estimate_speech_activity(
        audio_features, audio_valid, voiced_flag, working_hz=working_hz
    )
    group_size = int(activity_meta["group_size"])
    frame_hop_sec = AUDIO_HOP / AUDIO_SR
    working_count = len(score_work)

    if working_count < word_count:
        edges = np.linspace(0.0, max(audio_duration_sec, 0.0), word_count + 1, dtype=np.float32)
        starts, ends = edges[:-1], edges[1:]
        confidence = np.zeros(word_count, dtype=np.float32)
        fallback = np.ones(word_count, dtype=bool)
        return starts, ends, confidence, fallback, {
            "backend": ALIGNMENT_BACKEND,
            "fallback": "uniform_too_few_audio_frames",
            "working_frames": int(working_count),
            "word_count": word_count,
            "assigned_word_count": word_count,
            "coverage": 1.0,
            "mean_confidence": 0.0,
            "fallback_used": True,
        }

    score_work = np.nan_to_num(score_work, nan=0.0, posinf=0.0, neginf=0.0)
    activity_work = np.asarray(activity_meta["activity_working"], dtype=np.float32)
    activity_prefix = np.concatenate([[0.0], np.cumsum(activity_work, dtype=np.float64)])

    weights = np.asarray([
        1.0 + 0.52 * len(item["clean"]) + (0.30 if item.get("trailing_punct") else 0.0)
        for item in tokens
    ], dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    target_boundaries = cumulative / cumulative[-1] * working_count
    average_word_frames = working_count / max(word_count, 1)
    radius = max(3, int(round(0.72 * average_word_frames)) + 3)

    candidates: list[np.ndarray] = []
    lead = max(2, int(round(0.12 * working_count)))
    start_candidates = np.arange(0, min(lead, working_count - word_count) + 1)
    if len(start_candidates) == 0:
        start_candidates = np.asarray([0])
    candidates.append(np.unique(start_candidates).astype(np.int32))

    transitions = np.flatnonzero(np.diff(activity_work.astype(np.int8)) != 0) + 1
    minima, _ = find_peaks(-score_work, distance=2)
    for boundary_index in range(1, word_count):
        target = float(target_boundaries[boundary_index])
        left = max(boundary_index, int(math.floor(target - radius)))
        right = min(working_count - (word_count - boundary_index), int(math.ceil(target + radius)))
        if right < left:
            left = right = int(np.clip(round(target), boundary_index, working_count - (word_count - boundary_index)))
        values = list(range(left, right + 1))
        extra = transitions[(transitions >= left) & (transitions <= right)]
        extra_minima = minima[(minima >= left) & (minima <= right)]
        values.extend(int(v) for v in extra)
        values.extend(int(v) for v in extra_minima)
        candidates.append(np.unique(np.asarray(values, dtype=np.int32)))

    tail = max(2, int(round(0.12 * working_count)))
    end_candidates = np.arange(max(word_count - 1, working_count - tail), working_count + 1)
    end_candidates = end_candidates[end_candidates >= word_count - 1]
    if len(end_candidates) == 0:
        end_candidates = np.asarray([working_count])
    candidates.append(np.unique(end_candidates).astype(np.int32))

    negative_inf = -1e18
    dp = np.full(len(candidates[0]), negative_inf, dtype=np.float64)
    back: list[np.ndarray] = []
    for index, boundary in enumerate(candidates[0]):
        pause_reward = 0.55 * (1.0 - float(score_work[boundary])) if boundary < working_count else 0.0
        dp[index] = pause_reward

    for word_index in range(1, word_count + 1):
        current = np.full(len(candidates[word_index]), negative_inf, dtype=np.float64)
        previous_indices = np.full(len(candidates[word_index]), -1, dtype=np.int32)
        expected_length = max(1.0, float(weights[word_index - 1] / weights.sum() * working_count))
        for current_index, right in enumerate(candidates[word_index]):
            best_value = negative_inf
            best_previous = -1
            for previous_index, left in enumerate(candidates[word_index - 1]):
                if left >= right or dp[previous_index] <= negative_inf / 2:
                    continue
                actual_length = int(right - left)
                active_fraction = float((activity_prefix[right] - activity_prefix[left]) / max(actual_length, 1))
                log_ratio = math.log((actual_length + 0.5) / (expected_length + 0.5))
                duration_penalty = -2.0 * log_ratio * log_ratio
                short_penalty = -2.5 if actual_length < 1 else 0.0
                pause_reward = 0.75 * (1.0 - float(score_work[right])) if right < working_count else 0.0
                value = (
                    dp[previous_index]
                    + 3.0 * active_fraction
                    + duration_penalty
                    + short_penalty
                    + pause_reward
                )
                if value > best_value:
                    best_value = value
                    best_previous = previous_index
            current[current_index] = best_value
            previous_indices[current_index] = best_previous
        dp = current
        back.append(previous_indices)

    final_index = int(np.argmax(dp))
    boundary_indices = [final_index]
    for reverse_step in range(word_count - 1, -1, -1):
        previous = int(back[reverse_step][boundary_indices[-1]])
        if previous < 0:
            boundary_indices.append(0)
        else:
            boundary_indices.append(previous)
    boundary_indices.reverse()
    boundary_working = np.asarray([candidates[i][boundary_indices[i]] for i in range(word_count + 1)], dtype=np.int32)

    boundary_original = np.rint(boundary_working.astype(np.float64) * group_size).astype(np.int32)
    boundary_original[0] = max(0, min(int(boundary_original[0]), frame_count))
    boundary_original[-1] = max(0, min(int(boundary_original[-1]), frame_count))
    for index in range(1, len(boundary_original)):
        boundary_original[index] = max(boundary_original[index], boundary_original[index - 1] + 1)
    if boundary_original[-1] > frame_count:
        boundary_original = np.linspace(0, frame_count, word_count + 1).round().astype(np.int32)

    starts = boundary_original[:-1].astype(np.float64) * frame_hop_sec
    ends = boundary_original[1:].astype(np.float64) * frame_hop_sec
    if frame_count > 0 and len(audio_times) > 0:
        decoded_end = float(max(audio_duration_sec, audio_times[-1] + frame_hop_sec / 2.0))
    else:
        decoded_end = float(audio_duration_sec)
    ends = np.minimum(ends, decoded_end).astype(np.float32)
    starts = np.minimum(starts, ends).astype(np.float32)

    confidence = np.zeros(word_count, dtype=np.float32)
    fallback = np.zeros(word_count, dtype=bool)
    for word_index in range(word_count):
        left, right = int(boundary_original[word_index]), int(boundary_original[word_index + 1])
        left = max(0, min(left, frame_count))
        right = max(left + 1, min(right, frame_count))
        valid_segment = valid_frame[left:right]
        score_segment = score_frame[left:right]
        if np.any(valid_segment):
            active_fraction = float(score_segment[valid_segment].mean())
        else:
            active_fraction = 0.0
        expected_length = max(1.0, float(weights[word_index] / weights.sum() * frame_count))
        actual_length = max(1, right - left)
        duration_fit = math.exp(-abs(math.log((actual_length + 0.5) / (expected_length + 0.5))))
        pause_quality = 0.5 * (
            (1.0 - float(score_work[min(left // group_size, working_count - 1)]))
            + (1.0 - float(score_work[min(max((right - 1) // group_size, 0), working_count - 1)]))
        )
        confidence[word_index] = float(np.clip(0.58 * active_fraction + 0.22 * pause_quality + 0.20 * duration_fit, 0.0, 1.0))
        fallback[word_index] = bool(active_fraction < 0.12 or confidence[word_index] < 0.22)

    metadata = {
        "backend": ALIGNMENT_BACKEND,
        "description": "Monotonic non-overlapping energy/pause-constrained DP; deterministic baseline, not a neural forced aligner.",
        "working_hz": working_hz,
        "working_frames": int(working_count),
        "group_size_audio_frames": group_size,
        "word_count": int(word_count),
        "assigned_word_count": int(word_count),
        "coverage": 1.0,
        "mean_confidence": float(confidence.mean()) if len(confidence) else 0.0,
        "min_confidence": float(confidence.min()) if len(confidence) else 0.0,
        "low_confidence_words": int(fallback.sum()),
        "fallback_used": bool(np.any(fallback)),
        "speech_activity": activity_meta,
        "boundary_rule": "strictly_monotonic_non_overlapping",
        "weights_rule": "1 + 0.52*clean_token_length + 0.30*trailing_punctuation",
    }
    return starts.astype(np.float32), ends.astype(np.float32), confidence, fallback, metadata


def _interval_weighted_mean(
    features: np.ndarray,
    valid: np.ndarray,
    edges_sec: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, bool, float]:
    dimension = features.shape[1] if features.ndim == 2 else 0
    if len(features) == 0 or end_sec <= start_sec:
        return np.zeros(dimension, dtype=np.float32), False, 0.0
    left = int(np.searchsorted(edges_sec, start_sec, side="right") - 1)
    right = int(np.searchsorted(edges_sec, end_sec, side="left"))
    left = max(0, min(left, len(features) - 1))
    right = max(left + 1, min(right, len(features)))
    weights = np.zeros(right - left, dtype=np.float64)
    for local_index, frame_index in enumerate(range(left, right)):
        overlap = min(end_sec, float(edges_sec[frame_index + 1])) - max(start_sec, float(edges_sec[frame_index]))
        weights[local_index] = max(0.0, overlap)
    selected_valid = np.all(valid[left:right], axis=1) & (weights > 0)
    if not np.any(selected_valid):
        return np.zeros(dimension, dtype=np.float32), False, 0.0
    selected_weights = weights[selected_valid]
    selected_features = features[left:right][selected_valid]
    observed = float(selected_weights.sum() / max(weights.sum(), 1e-12))
    mean = np.average(selected_features, axis=0, weights=selected_weights).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    return mean, True, observed


def aggregate_audio_to_words(
    audio_features: np.ndarray,
    audio_valid: np.ndarray,
    word_start_sec: np.ndarray,
    word_end_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(audio_features)
    word_count = len(word_start_sec)
    output = np.zeros((word_count, AUDIO_DIM), dtype=np.float32)
    observed = np.zeros(word_count, dtype=bool)
    ratios = np.zeros(word_count, dtype=np.float32)
    if frame_count == 0:
        return output, observed, ratios
    edges = np.arange(frame_count + 1, dtype=np.float64) * (AUDIO_HOP / AUDIO_SR)
    for word_index in range(word_count):
        output[word_index], observed[word_index], ratios[word_index] = _interval_weighted_mean(
            audio_features, audio_valid, edges, float(word_start_sec[word_index]), float(word_end_sec[word_index])
        )
    return output, observed, ratios


def aggregate_visual_to_words(
    visual_features: np.ndarray,
    visual_valid: np.ndarray,
    visual_times: np.ndarray,
    word_start_sec: np.ndarray,
    word_end_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    word_count = len(word_start_sec)
    output = np.zeros((word_count, VISION_DIM), dtype=np.float32)
    observed = np.zeros(word_count, dtype=bool)
    ratios = np.zeros(word_count, dtype=np.float32)
    for word_index in range(word_count):
        selected = (
            (visual_times >= float(word_start_sec[word_index]))
            & (visual_times < float(word_end_sec[word_index]))
            & visual_valid
        )
        sampled_count = int(np.sum((visual_times >= float(word_start_sec[word_index])) & (visual_times < float(word_end_sec[word_index]))))
        if sampled_count > 0:
            ratios[word_index] = float(np.sum(selected) / sampled_count)
        if np.any(selected):
            output[word_index] = visual_features[selected].mean(axis=0).astype(np.float32)
            observed[word_index] = True
    return output, observed, ratios

def load_precomputed_text(feature_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key = safe_sample_key(sample_id)
    npz_path = feature_dir / "features" / f"{key}.npz"
    json_path = feature_dir / "metadata" / f"{key}.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"precomputed text features not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    features = np.asarray(data["text_features"], dtype=np.float32)
    valid = np.asarray(data["text_valid"], dtype=bool)
    metadata = {"backend": "precomputed", "path": str(npz_path)}
    if json_path.exists():
        metadata.update(json.loads(json_path.read_text(encoding="utf-8")))
    return features, valid, metadata


def save_text_only_outputs(
    sample_id: str,
    text: str,
    tokens: list[dict[str, Any]],
    features: np.ndarray,
    valid: np.ndarray,
    backend_metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    key = safe_sample_key(sample_id)
    feature_dir = output_dir / "features"
    metadata_dir = output_dir / "metadata"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    npz_path = feature_dir / f"{key}.npz"
    json_path = metadata_dir / f"{key}.json"
    np.savez_compressed(
        npz_path,
        text_features=features.astype(np.float32),
        text_valid=valid.astype(bool),
        word_tokens=np.asarray([item["token"] for item in tokens], dtype="<U256"),
    )
    payload = {
        "sample_id": sample_id,
        "raw_text": text,
        "word_tokens": [item["token"] for item in tokens],
        "backend_metadata": backend_metadata,
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else 0,
        "word_count": int(len(tokens)),
    }
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, json_path


def _feature_finite_counts(*arrays: np.ndarray) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for array in arrays:
        if not np.issubdtype(array.dtype, np.number):
            continue
        nan_count += int(np.isnan(array).sum())
        inf_count += int(np.isinf(array).sum())
    return nan_count, inf_count


def extract_word_level_sample(
    text: str,
    video_path: Path,
    text_extractor: Callable[[str, list[dict[str, Any]]], tuple[np.ndarray, dict[str, Any]]] | None,
    smooth_sigma: float = 0.75,
    text_features_override: np.ndarray | None = None,
    text_valid_override: np.ndarray | None = None,
    text_metadata_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = tokenize_with_spans(text)
    if text_features_override is None:
        if text_extractor is None:
            raise ValueError("text_extractor is required when no precomputed text features are supplied")
        text_features, text_metadata = text_extractor(text, tokens)
    else:
        text_features = np.asarray(text_features_override, dtype=np.float32)
        text_metadata = text_metadata_override or {"backend": "precomputed", "fallback_words": 0}
    if text_valid_override is None:
        text_valid = np.ones((len(tokens), TEXT_DIM), dtype=bool)
    else:
        text_valid = np.asarray(text_valid_override, dtype=bool)
    if text_features.shape != (len(tokens), TEXT_DIM):
        raise RuntimeError(f"text feature shape mismatch: {text_features.shape}")

    audio_features_raw, audio_valid, audio_times, voiced_flag, audio_metadata = extract_audio_features(video_path)
    audio_features, audio_smoothing = smooth_contiguous_valid(audio_features_raw, audio_valid, sigma=smooth_sigma)
    vision_features_raw, vision_valid, vision_times, vision_metadata = extract_visual_features(video_path)
    vision_features, vision_smoothing = smooth_contiguous_valid(vision_features_raw, vision_valid, sigma=smooth_sigma)

    audio_duration = float(audio_metadata.get("duration_sec", 0.0))
    word_start, word_end, word_confidence, word_fallback, alignment_metadata = align_words_to_audio(
        tokens,
        audio_features,
        audio_valid,
        audio_times,
        voiced_flag,
        audio_duration,
    )
    word_audio_features, word_audio_valid, word_audio_observed_ratio = aggregate_audio_to_words(
        audio_features, audio_valid, word_start, word_end
    )
    word_vision_features, word_vision_valid, word_vision_observed_ratio = aggregate_visual_to_words(
        vision_features, vision_valid, vision_times, word_start, word_end
    )

    numeric_arrays = (
        text_features, text_valid.astype(np.float32), audio_features, audio_valid.astype(np.float32),
        audio_times, vision_features, vision_valid.astype(np.float32), vision_times,
        word_audio_features, word_audio_valid.astype(np.float32), word_audio_observed_ratio,
        word_vision_features, word_vision_valid.astype(np.float32), word_vision_observed_ratio,
        word_start, word_end, word_confidence, word_fallback.astype(np.float32),
    )
    nan_count, inf_count = _feature_finite_counts(*numeric_arrays)
    if nan_count or inf_count:
        raise RuntimeError(f"non-finite values after extraction: nan={nan_count}, inf={inf_count}")

    if len(word_start) and (
        not np.all(word_end > word_start)
        or not np.all(np.diff(word_start) >= 0)
        or not np.all(np.diff(word_end) >= 0)
        or float(word_start.min()) < -1e-6
        or float(word_end.max()) > audio_duration + 1e-3
    ):
        raise RuntimeError("word alignment interval audit failed")

    return {
        "tokens": tokens,
        "text_features": text_features.astype(np.float32),
        "text_valid": text_valid,
        "text_metadata": text_metadata,
        "audio_features": audio_features,
        "audio_valid": audio_valid,
        "audio_times": audio_times,
        "audio_metadata": audio_metadata,
        "audio_smoothing": audio_smoothing,
        "voiced_flag": voiced_flag,
        "vision_features": vision_features,
        "vision_valid": vision_valid,
        "vision_times": vision_times,
        "vision_metadata": vision_metadata,
        "vision_smoothing": vision_smoothing,
        "word_start_sec": word_start,
        "word_end_sec": word_end,
        "word_confidence": word_confidence,
        "word_fallback": word_fallback,
        "alignment_metadata": alignment_metadata,
        "word_audio_features": word_audio_features,
        "word_audio_valid": word_audio_valid,
        "word_audio_observed_ratio": word_audio_observed_ratio,
        "word_vision_features": word_vision_features,
        "word_vision_valid": word_vision_valid,
        "word_vision_observed_ratio": word_vision_observed_ratio,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_sample_outputs(
    sample: dict[str, Any],
    row: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    feature_dir = output_dir / "features"
    metadata_dir = output_dir / "metadata"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    key = safe_sample_key(str(row["sample_id"]))
    npz_path = feature_dir / f"{key}.npz"
    json_path = metadata_dir / f"{key}.json"

    word_tokens = np.asarray([item["token"] for item in sample["tokens"]], dtype="<U256")
    np.savez_compressed(
        npz_path,
        text_features=sample["text_features"],
        text_valid=sample["text_valid"],
        word_tokens=word_tokens,
        audio_features=sample["audio_features"],
        audio_valid=sample["audio_valid"],
        audio_times=sample["audio_times"],
        vision_features=sample["vision_features"],
        vision_valid=sample["vision_valid"],
        vision_times=sample["vision_times"],
        word_audio_features=sample["word_audio_features"],
        word_audio_valid=sample["word_audio_valid"],
        word_audio_observed_ratio=sample["word_audio_observed_ratio"],
        word_vision_features=sample["word_vision_features"],
        word_vision_valid=sample["word_vision_valid"],
        word_vision_observed_ratio=sample["word_vision_observed_ratio"],
        word_start_sec=sample["word_start_sec"],
        word_end_sec=sample["word_end_sec"],
        word_confidence=sample["word_confidence"],
        word_fallback=sample["word_fallback"],
    )

    metadata = {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "clip_id": row["clip_id"],
        "raw_text": row.get("text", ""),
        "source_video_path": row["resolved_video_path"],
        "source_sha256": row["source_sha256"],
        "word_tokens": [item["token"] for item in sample["tokens"]],
        "text": sample["text_metadata"],
        "audio": sample["audio_metadata"],
        "audio_smoothing": sample["audio_smoothing"],
        "vision": sample["vision_metadata"],
        "vision_smoothing": sample["vision_smoothing"],
        "alignment": sample["alignment_metadata"],
        "dimensions": {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM},
        "padding_rule": "No cross-sample padding in NPZ; variable-length arrays are retained and all invalid entries are masked.",
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, ensure_ascii=False, indent=2)
    return npz_path, json_path


def resolve_video_path(row: dict[str, str], raw_root: Path) -> Path:
    original = Path(row.get("source_video_path", ""))
    if original.is_file():
        return original.resolve()
    video_id = str(row.get("video_id", "")).strip()
    clip_id = str(row.get("clip_id", "")).strip()
    candidates = [
        raw_root / video_id / f"{clip_id}.mp4",
        raw_root / "MOSEI数据集部分原始视频-100条" / video_id / f"{clip_id}.mp4",
        raw_root / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条" / video_id / f"{clip_id}.mp4",
        raw_root / f"{clip_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(raw_root.rglob(f"{clip_id}.mp4")) if clip_id else []
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"video not found for sample {row.get('sample_id')}: {candidates}")
    raise RuntimeError(f"ambiguous video match for sample {row.get('sample_id')}: {matches[:5]}")


def _tool_versions() -> dict[str, str]:
    import platform
    from importlib import metadata

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": tool_version("ffmpeg"),
    }
    for package in ["numpy", "scipy", "opencv-python", "Pillow", "torch", "transformers"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def write_feature_dictionary(output_dir: Path, args: argparse.Namespace) -> None:
    content = f"""# 问题一特征字典与处理规则

## 总体接口

- 样本单位：附件1中的每个原始视频；样本级 NPZ 保留变长序列，不使用跨样本 padding。
- 词级公共索引：英文转写经 `[A-Za-z0-9]+(?:['’\\-][A-Za-z0-9]+)*` 分词；每个词都有 `word_start_sec`、`word_end_sec`、`word_confidence` 和 `word_fallback`。
- 所有模态缺失均由 `*_valid` 或 `word_*_valid` 显式标记，零向量不等于有效观测。
- 弱平滑只在连续且全维有效的帧段内执行，不跨 padding、缺失边界或视频边界。

## 文本模态

- 后端：`{args.text_backend}`；BERT 模型：`{args.model_name}`。
- 维度：{TEXT_DIM}。
- 输出：`text_features`、`text_valid`、`word_tokens`。
- 处理：`bert-base-uncased` 子词 offset 映射到词级均值；无子词映射时使用确定性哈希回退。
- 未做文本改写、标签训练或跨模态监督。

## 语音模态

- 采样率：{AUDIO_SR} Hz；窗长：{AUDIO_NFFT} 点（25 ms）；步长：{AUDIO_HOP} 点（10 ms）。
- 维度：{AUDIO_DIM}；Mel 滤波器：{AUDIO_NMELS}；MFCC：{AUDIO_NMFCC}。
- 组成：能量、ZCR、频谱质心、带宽、85% 滚降、平坦度、频谱通量、有声概率、归一化基频、13 维 MFCC、10 维 MFCC 差分、8 维 log-Mel。
- 弱平滑：连续全维有效帧内 `gaussian_filter1d(sigma={args.smooth_sigma})`；mask 不改变。
- 词级聚合：按时间重叠加权平均；无可观测帧时 `word_audio_valid=0`。

## 视觉模态

- 后端：OpenCV Haar `haarcascade_frontalface_default.xml`。
- 采样率：{VISION_FPS} fps；人脸 ROI：{VISION_FACE_SIZE}×{VISION_FACE_SIZE}。
- 维度：{VISION_DIM}。
- 组成：人脸框几何 8 维、Hu 矩 7 维、ROI 3×3 均值/标准差 18 维、灰度直方图 16 维、HOG 2×2×9=36 维、Laplacian 纹理 5 维。
- 检测失败：零向量且 `vision_valid=0`；不推断为中性表情。
- 弱平滑：仅连续检测成功帧内 `gaussian_filter1d(sigma={args.smooth_sigma})`；mask 不改变。
- 词级聚合：词区间内有效采样帧均值；无有效帧时 `word_vision_valid=0`。

## 词级对齐

- 后端：`{ALIGNMENT_BACKEND}`，20 fps 工作分辨率。
- 方法：语音能量、有声概率、频谱通量估计活动与停顿；词长权重、单调边界、非重叠约束和停顿奖励组成动态规划。
- 边界规则：严格单调、非重叠；词区间限制在有效音频时长内。
- 置信度：活动占比、边界停顿质量和时长适配度的加权组合。
- 边界说明：这是确定性可复现的基线，不宣称等价于神经强制对齐器。

## NPZ 字段

- `text_features`：`[W, {TEXT_DIM}]`
- `text_valid`：`[W, {TEXT_DIM}]`
- `word_tokens`：`[W]`
- `audio_features`：`[T, {AUDIO_DIM}]`，`audio_times` 为帧中心秒数
- `audio_valid`：`[T, {AUDIO_DIM}]`
- `vision_features`：`[V, {VISION_DIM}]`，`vision_times` 为采样时间秒数
- `vision_valid`：`[V]`
- `word_audio_features`：`[W, {AUDIO_DIM}]`，`word_audio_valid`：`[W]`
- `word_vision_features`：`[W, {VISION_DIM}]`，`word_vision_valid`：`[W]`
- `word_start_sec`、`word_end_sec`、`word_confidence`、`word_fallback`：`[W]`
"""
    (output_dir / "feature_dictionary.md").write_text(content, encoding="utf-8")


def _draw_typical_alignment(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    candidates = [row for row in rows if row.get("status") == "ok" and int(row.get("word_count", 0)) > 0]
    if not candidates:
        return
    candidates.sort(key=lambda item: int(item.get("word_count", 0)))
    row = candidates[len(candidates) // 2]
    key = safe_sample_key(str(row["sample_id"]))
    npz_path = output_dir / "features" / f"{key}.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path, allow_pickle=False)
    times = data["audio_times"]
    energy = data["audio_features"][:, 0] if len(data["audio_features"]) else np.zeros((0,), dtype=np.float32)
    if len(energy):
        low, high = np.percentile(energy, [5, 95])
        energy_plot = np.clip((energy - low) / max(high - low, 1e-8), 0.0, 1.0)
    else:
        energy_plot = energy
    starts = data["word_start_sec"]
    ends = data["word_end_sec"]
    confidence = data["word_confidence"]
    tokens = data["word_tokens"].astype(str)
    duration = float(max(float(times[-1]) if len(times) else 0.0, float(ends[-1]) if len(ends) else 1.0))

    width, height = 1800, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left_margin, right_margin = 90, 60
    plot_width = width - left_margin - right_margin
    plot_top, plot_bottom = 100, 360
    draw.text((left_margin, 30), f"Typical word-level alignment: {row['sample_id']}", fill="black")
    draw.line((left_margin, plot_bottom, width - right_margin, plot_bottom), fill="black", width=2)
    if len(energy_plot) > 1:
        points = []
        for index in range(len(energy_plot)):
            x = left_margin + plot_width * float(times[index]) / max(duration, 1e-8)
            y = plot_bottom - (plot_bottom - plot_top) * float(energy_plot[index])
            points.append((x, y))
        draw.line(points, fill=(35, 100, 210), width=2)
    draw.text((left_margin, plot_bottom + 12), "0 s", fill="black")
    draw.text((width - right_margin - 45, plot_bottom + 12), f"{duration:.1f} s", fill="black")
    draw.text((left_margin, plot_top - 26), "speech energy (normalized)", fill=(35, 100, 210))

    max_words = min(len(tokens), 36)
    row_height = 14
    word_top = 430
    for index in range(max_words):
        x0 = left_margin + plot_width * float(starts[index]) / max(duration, 1e-8)
        x1 = left_margin + plot_width * float(ends[index]) / max(duration, 1e-8)
        y = word_top + index * row_height
        color = (70, 150, 90) if float(confidence[index]) >= 0.45 else (220, 150, 50)
        draw.rectangle((x0, y, max(x1, x0 + 2), y + 11), fill=color)
        label = str(tokens[index])
        if len(label) > 18:
            label = label[:15] + "..."
        draw.text((x1 + 4, y - 2), label, fill="black")
    if len(tokens) > max_words:
        draw.text((left_margin, word_top + max_words * row_height + 8), f"... {len(tokens) - max_words} more words", fill="black")
    image.save(output_dir / "typical_alignment_example.png")

    with (output_dir / "typical_alignment_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "word_index", "token", "start_sec", "end_sec", "confidence", "fallback"])
        for index in range(len(tokens)):
            writer.writerow([row["sample_id"], index, tokens[index], float(starts[index]), float(ends[index]), float(confidence[index]), bool(data["word_fallback"][index])])


def _write_summary_and_audit(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    total_rows: int,
    tool_versions: dict[str, str],
) -> None:
    with (output_dir / "q1_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["sample_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    ok_rows = [row for row in manifest_rows if row.get("status") == "ok"]
    error_rows = [row for row in manifest_rows if row.get("status") != "ok"]
    audit = {
        "total_input_samples": int(total_rows),
        "processed_samples": int(len(manifest_rows)),
        "ok_samples": int(len(ok_rows)),
        "error_samples": int(len(error_rows)),
        "coverage": float(len(manifest_rows) / total_rows) if total_rows else 0.0,
        "all_samples_covered": bool(len(manifest_rows) == total_rows and not error_rows),
        "all_word_intervals_valid": bool(all(row.get("alignment_interval_valid", False) for row in ok_rows)),
        "all_arrays_finite": bool(all(int(row.get("nan_count", 0)) == 0 and int(row.get("inf_count", 0)) == 0 for row in ok_rows)),
        "audio_observed_word_ratio_mean": float(np.mean([float(row.get("audio_observed_word_ratio", 0.0)) for row in ok_rows])) if ok_rows else 0.0,
        "vision_observed_word_ratio_mean": float(np.mean([float(row.get("vision_observed_word_ratio", 0.0)) for row in ok_rows])) if ok_rows else 0.0,
        "alignment_confidence_mean": float(np.mean([float(row.get("alignment_mean_confidence", 0.0)) for row in ok_rows])) if ok_rows else 0.0,
        "smoothing": {"sigma": 0.75, "boundary_rule": "within_contiguous_valid_runs_only"},
        "tool_versions": tool_versions,
        "generated_at": now_iso(),
    }
    (output_dir / "quality_audit.json").write_text(json.dumps(_json_safe(audit), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1 multimodal feature extraction and word-level alignment")
    parser.add_argument("--csv", type=Path, required=True, help="qualified_samples.csv")
    parser.add_argument("--raw-root", type=Path, required=True, help="root containing attachment 1 videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-backend", choices=["bert", "hash", "precomputed"], default="bert")
    parser.add_argument("--text-feature-dir", type=Path, default=None, help="precomputed BERT text feature directory")
    parser.add_argument("--text-only", action="store_true", help="only extract and save text features; do not decode audio/video")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smooth-sigma", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    selected_rows = all_rows
    if args.sample_id:
        wanted = set(args.sample_id)
        selected_rows = [row for row in selected_rows if row.get("sample_id") in wanted]
    if args.limit and args.limit > 0:
        selected_rows = selected_rows[:args.limit]

    versions = _tool_versions()
    if args.text_only or args.text_backend == "bert":
        text_extractor, text_backend_metadata = load_bert_backend(args.model_name, args.device)
    elif args.text_backend == "precomputed":
        if args.text_feature_dir is None:
            raise ValueError("--text-feature-dir is required with --text-backend precomputed")
        text_extractor = None
        text_backend_metadata = {"backend": "precomputed", "path": str(args.text_feature_dir), "feature_dim": TEXT_DIM}
    else:
        text_extractor = lambda text, tokens: (hash_text_features(tokens), {"backend": "hash", "fallback_words": 0})
        text_backend_metadata = {"backend": "hash", "model_name": None, "device": "cpu", "feature_dim": TEXT_DIM}

    if args.text_only:
        text_manifest_rows: list[dict[str, Any]] = []
        for index, row in enumerate(selected_rows, start=1):
            sample_id = row.get("sample_id", "")
            try:
                tokens = tokenize_with_spans(row.get("text", ""))
                features, text_metadata = text_extractor(row.get("text", ""), tokens)
                valid = np.ones((len(tokens), TEXT_DIM), dtype=bool)
                npz_path, json_path = save_text_only_outputs(
                    sample_id, row.get("text", ""), tokens, features, valid, text_metadata, args.output_dir
                )
                text_manifest_rows.append({
                    "sample_id": sample_id, "video_id": row.get("video_id", ""), "clip_id": row.get("clip_id", ""),
                    "word_count": len(tokens), "feature_dim": TEXT_DIM, "output_npz": str(npz_path),
                    "metadata_json": str(json_path), "status": "ok", "error": "",
                })
            except Exception as exc:
                text_manifest_rows.append({
                    "sample_id": sample_id, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "word_count": 0, "feature_dim": TEXT_DIM, "output_npz": "", "metadata_json": "",
                })
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(selected_rows)):
                print(f"[text {index}/{len(selected_rows)}] {sample_id} {text_manifest_rows[-1]['status']}", flush=True)
        fields = ["sample_id", "video_id", "clip_id", "word_count", "feature_dim", "output_npz", "metadata_json", "status", "error"]
        with (args.output_dir / "text_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(text_manifest_rows)
        ok_count = sum(1 for row in text_manifest_rows if row["status"] == "ok")
        print(json.dumps({"text_only": True, "selected": len(selected_rows), "ok": ok_count, "errors": len(selected_rows) - ok_count, "output_dir": str(args.output_dir.resolve()), "backend": text_backend_metadata}, ensure_ascii=False, indent=2))
        return 0 if ok_count == len(selected_rows) else 1

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    log_path = args.output_dir / "q1_processing_log.jsonl"
    alignment_path = args.output_dir / "q1_alignment.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    alignment_handle = alignment_path.open("w", encoding="utf-8")
    started_at = now_iso()
    try:
        for index, row in enumerate(selected_rows, start=1):
            sample_id = row.get("sample_id", "")
            key = safe_sample_key(sample_id)
            base_manifest = {
                "sample_id": sample_id,
                "video_id": row.get("video_id", ""),
                "clip_id": row.get("clip_id", ""),
                "text_chars": int(parse_float(row.get("text_chars"), 0.0)),
                "label": row.get("label", ""),
                "annotation": row.get("annotation", ""),
                "source_video_path": row.get("source_video_path", ""),
                "status": "ok",
                "error": "",
            }
            try:
                video_path = resolve_video_path(row, args.raw_root)
                source_hash = sha256_file(video_path)
                override_features = override_valid = override_metadata = None
                if args.text_feature_dir is not None:
                    override_features, override_valid, override_metadata = load_precomputed_text(
                        args.text_feature_dir, sample_id
                    )
                sample = extract_word_level_sample(
                    row.get("text", ""),
                    video_path,
                    text_extractor,
                    smooth_sigma=max(0.0, float(args.smooth_sigma)),
                    text_features_override=override_features,
                    text_valid_override=override_valid,
                    text_metadata_override=override_metadata,
                )
                audio_duration = float(sample["audio_metadata"].get("duration_sec", 0.0))
                interval_valid = bool(
                    len(sample["word_start_sec"]) == 0
                    or (
                        np.all(sample["word_end_sec"] > sample["word_start_sec"])
                        and np.all(np.diff(sample["word_start_sec"]) >= -1e-6)
                        and np.all(np.diff(sample["word_end_sec"]) >= -1e-6)
                        and float(sample["word_end_sec"].max()) <= audio_duration + 1e-3
                    )
                )
                npz_path, metadata_path = save_sample_outputs(
                    sample,
                    {**base_manifest, "resolved_video_path": str(video_path), "source_sha256": source_hash},
                    args.output_dir,
                )
                audio_word_valid = sample["word_audio_valid"]
                vision_word_valid = sample["word_vision_valid"]
                row_manifest = {
                    **base_manifest,
                    "resolved_video_path": str(video_path),
                    "source_sha256": source_hash,
                    "output_npz": str(npz_path),
                    "metadata_json": str(metadata_path),
                    "word_count": int(len(sample["tokens"])),
                    "audio_frames": int(len(sample["audio_features"])),
                    "audio_valid_frames": int(np.all(sample["audio_valid"], axis=1).sum()),
                    "vision_samples": int(len(sample["vision_features"])),
                    "vision_valid_samples": int(sample["vision_valid"].sum()),
                    "audio_observed_word_ratio": float(np.mean(audio_word_valid)) if len(audio_word_valid) else 0.0,
                    "vision_observed_word_ratio": float(np.mean(vision_word_valid)) if len(vision_word_valid) else 0.0,
                    "alignment_mean_confidence": float(sample["alignment_metadata"].get("mean_confidence", 0.0)),
                    "alignment_low_confidence_words": int(sample["alignment_metadata"].get("low_confidence_words", 0)),
                    "text_fallback_words": int(sample["text_metadata"].get("fallback_words", 0)),
                    "alignment_interval_valid": interval_valid,
                    "nan_count": 0,
                    "inf_count": 0,
                    "status": "ok",
                    "error": "",
                }
                manifest_rows.append(row_manifest)
                summary_rows.append({
                    "sample_id": sample_id,
                    "video_id": row.get("video_id", ""),
                    "clip_id": row.get("clip_id", ""),
                    "word_count": int(len(sample["tokens"])),
                    "text_dim": TEXT_DIM,
                    "audio_frames": int(len(sample["audio_features"])),
                    "audio_valid_frames": int(np.all(sample["audio_valid"], axis=1).sum()),
                    "vision_samples": int(len(sample["vision_features"])),
                    "vision_valid_samples": int(sample["vision_valid"].sum()),
                    "audio_observed_word_ratio": float(np.mean(audio_word_valid)) if len(audio_word_valid) else 0.0,
                    "vision_observed_word_ratio": float(np.mean(vision_word_valid)) if len(vision_word_valid) else 0.0,
                    "alignment_mean_confidence": float(sample["alignment_metadata"].get("mean_confidence", 0.0)),
                    "alignment_low_confidence_words": int(sample["alignment_metadata"].get("low_confidence_words", 0)),
                    "nan_count": 0,
                    "inf_count": 0,
                    "status": "ok",
                })
                alignment_record = {
                    "sample_id": sample_id,
                    "source_video_path": str(video_path),
                    "source_sha256": source_hash,
                    "backend": sample["alignment_metadata"].get("backend", ALIGNMENT_BACKEND),
                    "audio_duration_sec": audio_duration,
                    "word_count": int(len(sample["tokens"])),
                    "coverage": float(sample["alignment_metadata"].get("coverage", 0.0)),
                    "mean_confidence": float(sample["alignment_metadata"].get("mean_confidence", 0.0)),
                    "low_confidence_words": int(sample["alignment_metadata"].get("low_confidence_words", 0)),
                    "words": [
                        {
                            "index": word_index,
                            "token": sample["tokens"][word_index]["token"],
                            "start_sec": float(sample["word_start_sec"][word_index]),
                            "end_sec": float(sample["word_end_sec"][word_index]),
                            "confidence": float(sample["word_confidence"][word_index]),
                            "fallback": bool(sample["word_fallback"][word_index]),
                        }
                        for word_index in range(len(sample["tokens"]))
                    ],
                }
                alignment_handle.write(json.dumps(_json_safe(alignment_record), ensure_ascii=False) + "\n")
                alignment_handle.flush()
                log_handle.write(json.dumps(_json_safe({
                    "event": "sample_complete",
                    "sample_id": sample_id,
                    "index": index,
                    "total": len(selected_rows),
                    "status": "ok",
                    "output_npz": str(npz_path),
                    "timestamp": now_iso(),
                }), ensure_ascii=False) + "\n")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                row_manifest = {
                    **base_manifest,
                    "status": "error",
                    "error": error,
                    "word_count": 0,
                    "audio_frames": 0,
                    "vision_samples": 0,
                    "alignment_interval_valid": False,
                    "nan_count": 0,
                    "inf_count": 0,
                }
                manifest_rows.append(row_manifest)
                summary_rows.append({"sample_id": sample_id, "status": "error", "error": error})
                log_handle.write(json.dumps(_json_safe({
                    "event": "sample_error",
                    "sample_id": sample_id,
                    "index": index,
                    "total": len(selected_rows),
                    "status": "error",
                    "error": error,
                    "traceback": traceback.format_exc(),
                    "timestamp": now_iso(),
                }), ensure_ascii=False) + "\n")
            log_handle.flush()
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(selected_rows)):
                print(f"[{index}/{len(selected_rows)}] {sample_id} {row_manifest['status']}", flush=True)
    finally:
        log_handle.close()
        alignment_handle.close()

    manifest_fields = [
        "sample_id", "video_id", "clip_id", "text_chars", "label", "annotation",
        "source_video_path", "resolved_video_path", "source_sha256", "output_npz", "metadata_json",
        "word_count", "audio_frames", "audio_valid_frames", "vision_samples", "vision_valid_samples",
        "audio_observed_word_ratio", "vision_observed_word_ratio", "alignment_mean_confidence",
        "alignment_low_confidence_words", "text_fallback_words", "alignment_interval_valid",
        "nan_count", "inf_count", "status", "error",
    ]
    with (args.output_dir / "q1_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_feature_dictionary(args.output_dir, args)
    _draw_typical_alignment(args.output_dir, manifest_rows)
    _write_summary_and_audit(args.output_dir, manifest_rows, summary_rows, len(all_rows), versions)

    ok_count = sum(1 for row in manifest_rows if row.get("status") == "ok")
    print(json.dumps({
        "selected": len(selected_rows),
        "total_input": len(all_rows),
        "ok": ok_count,
        "errors": len(selected_rows) - ok_count,
        "output_dir": str(args.output_dir.resolve()),
        "text_backend": text_backend_metadata,
        "tools": versions,
    }, ensure_ascii=False, indent=2))
    return 0 if ok_count == len(selected_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())