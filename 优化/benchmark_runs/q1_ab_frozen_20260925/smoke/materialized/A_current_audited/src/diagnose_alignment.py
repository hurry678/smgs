"""Bounded same-model retry for a reproducible alignment failure; no transcript edits."""
import json
import time
import numpy as np
from prepare import ROOT, dump
from alignment import map_words, phrase_groups


def main():
    import torch
    import soundfile as sf
    import stable_whisper
    c = json.loads((ROOT / "configs/q1.json").read_text())
    torch.set_num_threads(c["torch_threads"])
    rows = {r["sample_id"]: r for r in json.loads((ROOT / "resources/manifest.json").read_text())}
    sid = "-ri04Z7vwnc$_$2"
    row = rows[sid]
    y, sr = sf.read(ROOT / "intermediate/audio" / (sid + ".wav"), dtype="float32")
    assert sr == 16000 and np.any(y != 0)
    model = stable_whisper.load_model("tiny.en", device="cpu",
                                     download_root=str(ROOT / "resources/models/whisper"))
    t0 = time.perf_counter()
    options = {"language": "en", "verbose": None, "regroup": False, "remove_instant_words": False,
               "presplit": False, "max_word_dur": None, "word_dur_factor": None,
               "nonspeech_skip": None, "fast_mode": True, "suppress_silence": True, "stream": False}
    result = model.align(y, row["text"], **options)
    cache = ROOT / "intermediate/samples" / sid
    dump(cache / "alignment_retry.json", result.to_dict())
    try:
        words = map_words(row["text"], result.to_dict(), row["audio_start_s"], row["duration_s"])
        phrases = phrase_groups(words, row["text"], c["alignment"], c["alignment_quality"])
        status = {"text_preserved": True, "words": len(words), "phrases": len(phrases),
                  "instant_words": sum(w["instant"] for w in words),
                  "word_details": words, "phrase_details": phrases}
    except ValueError as exc:
        status = {"text_preserved": False, "error": str(exc)}
    status.update({"sample_id": sid, "options": options, "elapsed_s": time.perf_counter() - t0,
                   "reason": "Default duration-based repeated realignment inserted a duplicate 'it'. "
                             "Retry disables presplitting and word-duration retry constraints in the same model."})
    dump(ROOT / "reports/alignment_retry.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
