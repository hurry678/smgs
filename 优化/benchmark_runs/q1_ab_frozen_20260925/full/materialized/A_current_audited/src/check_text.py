"""Check fixed BERT vocabulary against provided train/valid token IDs, without fitting."""
import importlib
import pickle
import time
import numpy as np
from prepare import ROOT, dump, sha256


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = module.replace("numpy._core", "numpy.core")
        allowed = {("numpy", "asarray"), ("numpy", "ndarray"), ("numpy", "dtype"),
                   ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
                   ("numpy.core.numeric", "_frombuffer"), ("builtins", "set"), ("builtins", "slice"),
                   ("_codecs", "encode")}
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported pickle global {module}.{name}")
        return getattr(importlib.import_module(module), name)


def main(config):
    import torch
    from transformers import BertTokenizerFast, BertModel
    torch.set_num_threads(config["torch_threads"])
    model_dir = ROOT / config["text"]["model_dir"]
    tok = BertTokenizerFast.from_pretrained(str(model_dir), do_lower_case=True, local_files_only=True)
    path = ROOT.parent / "E题数据/附件2-数据集特征文件/aligned_50.pkl"
    with path.open("rb") as f:
        data = ArrayUnpickler(f).load()
    report = {"model": config["text"]["model_id"], "vocab_sha256": sha256(model_dir / "vocab.txt"),
              "vocab_size": len(tok), "special_ids": {
                  k: getattr(tok, k + "_token_id") for k in ("pad", "unk", "cls", "sep", "mask")},
              "scope": "train/valid text and text_bert only; no labels, model selection or test evaluation",
              "splits": {}}
    for split in ("train", "valid"):
        d = data[split]
        expected = np.asarray(d["text_bert"])
        text = list(map(str, d["raw_text"]))
        t0 = time.perf_counter()
        actual = tok(text, padding="max_length", truncation=True, max_length=50)
        recoded = np.stack([actual[k] for k in ("input_ids", "attention_mask", "token_type_ids")], axis=1)
        same = (recoded == expected).all(axis=(1, 2))
        ids = expected[:, 0]
        failures = []
        for i in np.flatnonzero(~same):
            failures.append({"sample_id": str(d["id"][i]), "original_text": text[i],
                             "different_positions": np.argwhere(recoded[i] != expected[i]).tolist(),
                             "expected_tokens": tok.convert_ids_to_tokens(expected[i, 0].astype(int).tolist()),
                             "recoded_tokens": tok.convert_ids_to_tokens(recoded[i, 0].tolist())})
        report["splits"][split] = {
            "samples": len(text), "exact_matches": int(same.sum()), "exact_match_rate": float(same.mean()),
            "all_values_finite_integer": bool(np.isfinite(expected).all() and (expected == np.rint(expected)).all()),
            "all_ids_in_vocab": bool(((ids >= 0) & (ids < len(tok))).all()),
            "encoding_seconds": time.perf_counter() - t0, "mismatches": failures}
    del data
    model = BertModel.from_pretrained(str(model_dir), local_files_only=True).eval()
    model.requires_grad_(False)
    sample = tok("A short reproducible encoder check.", return_tensors="pt")
    with torch.inference_mode():
        a = model(**sample).last_hidden_state
        b = model(**sample).last_hidden_state
    report["encoder"] = {"dimension": a.shape[-1], "parameters": sum(p.numel() for p in model.parameters()),
                         "finite": bool(torch.isfinite(a).all()), "repeat_max_abs_error": float((a-b).abs().max()),
                         "stored_weights_bytes": (model_dir / "model.safetensors").stat().st_size}
    report["compatible"] = all(s["exact_matches"] == s["samples"] and s["all_ids_in_vocab"]
                               and s["all_values_finite_integer"] for s in report["splits"].values())
    dump(ROOT / "reports/tokenizer_check.json", report)
    print({k: {kk: vv for kk, vv in v.items() if kk != "mismatches"} for k, v in report["splits"].items()})
    print(report["encoder"])
    if not report["compatible"]:
        raise ValueError("Unexplained token mismatches; inspect tokenizer_check.json")


if __name__ == "__main__":
    import json
    main(json.loads((ROOT / "configs/q1.json").read_text()))
