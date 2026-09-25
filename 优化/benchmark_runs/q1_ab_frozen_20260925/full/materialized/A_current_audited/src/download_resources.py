"""Download official, fixed model assets; record source URLs and SHA-256."""
import json
import subprocess
from prepare import ROOT, dump, sha256

BERT_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
MFA_MODELS = (
    (
        "resources/models/mfa/english_mfa_acoustic_v3.1.0.zip",
        "https://github.com/MontrealCorpusTools/mfa-models/releases/download/"
        "acoustic-english_mfa-v3.1.0/english_mfa.zip",
        "Montreal Forced Aligner English MFA acoustic model",
        "3.1.0",
        "primary word and phone forced alignment",
    ),
    (
        "resources/models/mfa/english_mfa_dictionary_v3.1.0.dict",
        "https://github.com/MontrealCorpusTools/mfa-models/releases/download/"
        "dictionary-english_mfa-v3.1.0/english_mfa.dict",
        "Montreal Forced Aligner English MFA dictionary",
        "3.1.0",
        "orthography-to-pronunciation mapping for primary alignment",
    ),
    (
        "resources/models/mfa/english_us_mfa_g2p_v3.0.0.zip",
        "https://github.com/MontrealCorpusTools/mfa-models/releases/download/"
        "g2p-english_us_mfa-v3.0.0/english_us_mfa.zip",
        "Montreal Forced Aligner English US G2P model",
        "3.0.0",
        "deterministic pronunciations for out-of-vocabulary transcript words",
    ),
)


def expected_hashes(root=None):
    return json.loads(((root or ROOT) / "resources/resource_lock.json").read_text())["sha256"]


def fetch(url, path):
    relative = str(path.relative_to(ROOT))
    expected = expected_hashes()[relative]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            subprocess.run(["curl", "--fail", "-L", "--retry", "1", "--connect-timeout", "15",
                            "--max-time", "300", "-o", str(tmp), url], check=True)
            if sha256(tmp) != expected:
                raise ValueError(f"Downloaded resource SHA-256 mismatch: {relative}")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Existing resource SHA-256 mismatch: {relative}; resource lock was not changed")
    return {"path": relative, "url": url,
            "bytes": path.stat().st_size, "sha256": actual,
            "expected_sha256": expected, "verified_against": "resources/resource_lock.json"}


def main():
    records = []
    for name in ("config.json", "vocab.txt", "model.safetensors", "README.md"):
        url = f"https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/{BERT_REVISION}/{name}"
        item = fetch(url, ROOT / "resources/models/bert" / name)
        item.update({"model": "google/bert_uncased_L-2_H-128_A-2", "revision": BERT_REVISION,
                     "license": "Apache-2.0 (model card)", "use": "frozen 128-d text encoding"})
        records.append(item)
    for path, url, model, revision, use in MFA_MODELS:
        item = fetch(url, ROOT / path)
        item.update({
            "model": model, "revision": revision, "license": "CC BY 4.0",
            "use": use,
        })
        records.append(item)
    records.append(fetch(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        ROOT / "resources/models/face_landmarker.task"))
    records[-1].update({"model": "MediaPipe Face Landmarker float16 v1",
                        "use": "face landmarks only, no identity recognition",
                        "license_reference": "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"})
    for name, url in (
        ("stable-ts-README.md", "https://raw.githubusercontent.com/jianfch/stable-ts/main/README.md"),
        ("whisper-LICENSE", "https://raw.githubusercontent.com/openai/whisper/main/LICENSE"),
        ("mediapipe-LICENSE", "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/LICENSE"),
    ):
        records.append(fetch(url, ROOT / "resources/reference" / name))
    import whisper
    url = whisper._MODELS["tiny.en"]
    item = fetch(url, ROOT / "resources/models/whisper/tiny.en.pt")
    if item["sha256"] != url.split("/")[-2]:
        raise ValueError("Whisper official SHA-256 mismatch")
    item.update({"model": "OpenAI Whisper tiny.en", "license": "MIT",
                 "use": "given transcript alignment only", "official_hash_verified": True})
    records.append(item)
    dump(ROOT / "resources/model_sources.json", records)
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
