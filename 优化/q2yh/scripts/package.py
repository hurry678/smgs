#!/usr/bin/env python3
"""T04: build the self-contained submission package and verify it in a clean process.

The package holds only inference-time artefacts: the frozen candidate weights
(the shared encoder is never duplicated into a candidate file), exactly one copy
of the shared text encoder, the train-fitted normalizer, the publication rule,
the frozen Attachment-3 predictions and a standalone entry point that rebuilds
the model from the package alone.  ``package.json`` records the byte accounting
against the 50,000,000-byte combined Q1/Q2/Q3 limit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                       # noqa: E402
import engine as E                     # noqa: E402
from metrics import softmax            # noqa: E402
from protocol_core import publish_c2   # noqa: E402

INFER_SCRIPT = '''#!/usr/bin/env python3
"""Standalone frozen inference: rebuild the packaged model and predict one archive.

Only ``numpy``, ``torch``, ``transformers`` and this directory are required.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

LABELS = ("Negative", "Neutral", "Positive")


def apply_normalizer(x, visible, mean, std):
    values = (np.asarray(x, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return np.where(np.asarray(visible, dtype=bool)[..., None], values, 0).astype(np.float32)


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def publish_c2(prob, raw, epsilon=1e-4):
    cls = np.asarray(prob, dtype=np.float64).argmax(1).astype(np.int64)
    r = np.clip(np.asarray(raw, dtype=np.float64), -3.0, 3.0)
    out = np.zeros_like(r)
    pos, neg = cls == 2, cls == 0
    out[pos] = np.maximum(r[pos], epsilon)
    out[neg] = np.minimum(r[neg], -epsilon)
    return cls, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    pkg = args.package.resolve()
    sys.path.insert(0, str(pkg / "src"))
    import models as M
    cfg = json.loads((pkg / "model" / "config.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if cfg["architecture"] == "training_prior":
        model = M.build_model("training_prior", class_counts=cfg["class_counts"],
                              class_intensity_mean=cfg["class_intensity_mean"])
    else:
        model = M.build_model(cfg["architecture"], **cfg.get("model_kwargs", {}))
        state = torch.load(pkg / "model" / "state_dict.pt", map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("text_encoder.")]
        unexpected = [k for k in unexpected if not k.startswith("text_encoder.")]
        if missing or unexpected:
            raise SystemExit("state_dict mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    model.to(device).eval()
    z = np.load(args.input, allow_pickle=False)
    norm = np.load(pkg / "data" / "normalizer.npz", allow_pickle=False)
    observed = z["observed"].astype(bool)
    tensors = {
        "text_bert": torch.from_numpy(z["text_bert"].astype(np.int64)).to(device),
        "audio": torch.from_numpy(apply_normalizer(z["audio"], observed[:, 1],
                                                   norm["audio_mean"], norm["audio_std"])).to(device),
        "vision": torch.from_numpy(apply_normalizer(z["vision"], observed[:, 2],
                                                    norm["vision_mean"], norm["vision_std"])).to(device),
        "domain": torch.from_numpy(z["domain"].astype(bool)).to(device),
    }
    visible = torch.from_numpy(observed).to(device)
    logits, raw = [], []
    with torch.no_grad():
        for start in range(0, len(observed), 32):
            stop = min(start + 32, len(observed))
            out = model(tensors["text_bert"][start:stop], tensors["audio"][start:stop],
                        tensors["vision"][start:stop], tensors["domain"][start:stop],
                        visible[start:stop])
            logits.append(out["logits"].detach().float().cpu().numpy())
            raw.append(out["raw_intensity"].detach().float().cpu().numpy())
    logits = np.concatenate(logits)
    raw = np.concatenate(raw)
    prob = softmax(logits)
    cls, intensity = publish_c2(prob, raw, cfg["publication"]["nonneutral_epsilon"])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "logits.npz", logits=logits, raw=raw, prob=prob)
    ids = [str(x) for x in z["id"]]
    files = [str(x) for x in z["source_file"]]
    with (args.out / "q2_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cfg["publication"]["csv_columns"])
        for i, sid in enumerate(ids):
            writer.writerow([sid, files[i], LABELS[int(cls[i])], "%.6f" % intensity[i]])
    print(json.dumps({"rows": len(ids),
                      "ids_in_order": ids == ["附件3_%02d" % i for i in range(1, 31)]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha256_file(path):
    return D.sha256_file(path)


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_config(run_dir, manifest):
    config = {"schema": "q2-package-config-1", "architecture": manifest["architecture"],
              "model_kwargs": dict(manifest["config"].get("model_kwargs", {})),
              "publication": manifest["publication"],
              "input_protocol": manifest["input_protocol"],
              "text_encoder": {"dir": "resources/bert_shared",
                               "model_id": manifest["text_encoder"]["model_id"],
                               "revision": manifest["text_encoder"]["revision"],
                               "frozen": True, "always_eval": True,
                               "mask_before_encoding": True},
              "trained_run_id": manifest["deployed_run_id"],
              "deployment_seed": manifest["deployment_seed"],
              "checkpoint_sha256": manifest["checkpoint"]["sha256"]}
    if manifest["architecture"] == "training_prior":
        train = D.load_npz(run_dir / "data" / "train.npz")
        cls = train["classification_labels"].astype(int)
        reg = train["regression_labels"].astype(float)
        config["class_counts"] = [int((cls == k).sum()) for k in range(3)]
        config["class_intensity_mean"] = [float(reg[cls == k].mean()) if (cls == k).any() else 0.0
                                          for k in range(3)]
    return config


def export_state_dict(run_dir, manifest, out_path):
    checkpoint = run_dir / manifest["checkpoint"]["path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    leaked = [k for k in state if k.startswith("text_encoder.")]
    if leaked:
        raise SystemExit(f"checkpoint duplicates the frozen encoder: {leaked[:3]}")
    if not state:
        raise SystemExit("empty state_dict; refusing to package a SHA-only deliverable")
    torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, out_path)
    return len(state)


def frozen_logits(run_dir, manifest, input_path, device):
    from models import build_model
    config = build_config(run_dir, manifest)
    if manifest["architecture"] == "training_prior":
        model = build_model("training_prior", class_counts=config["class_counts"],
                            class_intensity_mean=config["class_intensity_mean"])
    else:
        model = build_model(manifest["architecture"], **config["model_kwargs"])
        E.load_checkpoint(run_dir / manifest["checkpoint"]["path"], model)
    model.to(device).eval()
    a3 = D.load_npz(input_path)
    normalizer = D.load_normalizer(run_dir / "data" / "normalizer.npz")
    tensors = {
        "text_bert": torch.from_numpy(a3["text_bert"].astype(np.int64)).to(device),
        "audio": torch.from_numpy(D.apply_normalizer(
            a3["audio"], a3["observed"][:, 1], normalizer["audio"]["mean"],
            normalizer["audio"]["std"])).to(device),
        "vision": torch.from_numpy(D.apply_normalizer(
            a3["vision"], a3["observed"][:, 2], normalizer["vision"]["mean"],
            normalizer["vision"]["std"])).to(device),
        "domain": torch.from_numpy(a3["domain"].astype(bool)).to(device),
        "observed": torch.from_numpy(a3["observed"].astype(bool)).to(device),
    }
    logits, raw = E.forward_condition(model, tensors, tensors["observed"], batch_size=32)
    return logits, raw, softmax(logits), publish_c2(softmax(logits), raw,
                                                    manifest["publication"]["nonneutral_epsilon"])


def clean_process_check(pkg, input_path, expected_logits, expected_raw, expected_csv, device):
    """Re-run the packaged model in a fresh interpreter and compare against the freeze."""
    tmp = pkg / "_clean_process"
    if tmp.exists():
        shutil.rmtree(tmp)
    env = dict(os.environ)
    env.update({"PYTHONNOUSERSITE": "1", "USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                "USE_FLAX": "0", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1"})
    cmd = [sys.executable, str(pkg / "infer.py"), "--package", str(pkg),
           "--input", str(input_path), "--out", str(tmp), "--device", device]
    proc = subprocess.run(cmd, cwd=str(pkg), env=env, capture_output=True, text=True)
    report = {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip()[-2000:],
              "stderr": proc.stderr.strip()[-2000:]}
    if proc.returncode != 0:
        raise SystemExit(f"clean-process inference failed: {report}")
    got = np.load(tmp / "logits.npz", allow_pickle=False)
    report["logits_max_abs_diff"] = float(np.abs(got["logits"] - expected_logits).max())
    report["raw_max_abs_diff"] = float(np.abs(got["raw"] - expected_raw).max())
    report["logits_ok"] = report["logits_max_abs_diff"] <= 1e-5
    report["raw_ok"] = report["raw_max_abs_diff"] <= 1e-5
    rows_expected = list(csv.DictReader(Path(expected_csv).open(encoding="utf-8-sig")))
    rows_got = list(csv.DictReader((tmp / "q2_predictions.csv").open(encoding="utf-8-sig")))
    report["rows_match"] = len(rows_expected) == len(rows_got) == 30
    report["polarity_match"] = all(a["polarity"] == b["polarity"]
                                   and a["sample_id"] == b["sample_id"]
                                   for a, b in zip(rows_expected, rows_got))
    report["intensity_max_readback_diff"] = max(
        (abs(float(a["intensity"]) - float(b["intensity"])) for a, b in zip(rows_expected, rows_got)),
        default=float("inf"))
    report["csv_ok"] = report["intensity_max_readback_diff"] <= 5e-7
    report["passed"] = bool(report["logits_ok"] and report["raw_ok"] and report["rows_match"]
                            and report["polarity_match"] and report["csv_ok"])
    shutil.rmtree(tmp)
    if not report["passed"]:
        raise SystemExit(f"clean-process verification failed: {json.dumps(report, ensure_ascii=False)}")
    return report


def zip_package(pkg, out_zip):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(pkg.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(pkg.parent).as_posix())
    return out_zip.stat().st_size


def write_run_manifest(run_dir):
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST_SHA256.txt":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}")
    target = run_dir / "MANIFEST_SHA256.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, len(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--q1-zip", type=Path, default=None,
                    help="existing Q1 archive, counted against the 50,000,000-byte limit")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-run-manifest", action="store_true")
    args = ap.parse_args()
    freeze = args.freeze.resolve()
    run_dir = freeze.parent
    manifest = load_manifest(freeze)
    pkg = (args.out or (run_dir / "package")).resolve()
    if pkg.exists():
        raise SystemExit(f"refusing to overwrite an existing package: {pkg}")
    submission = run_dir / "submission"
    csv_path = submission / "q2_predictions.csv"
    if not csv_path.exists():
        raise SystemExit("run predict_attachment3.py first: submission/q2_predictions.csv is missing")
    input_path = run_dir / "data" / "attachment3.npz"
    (pkg / "model").mkdir(parents=True)
    (pkg / "data").mkdir(parents=True)
    (pkg / "src").mkdir(parents=True)
    (pkg / "submission").mkdir(parents=True)
    (pkg / "resources" / "bert_shared").mkdir(parents=True)

    n_tensors = export_state_dict(run_dir, manifest, pkg / "model" / "state_dict.pt")
    (pkg / "model" / "config.json").write_text(
        json.dumps(build_config(run_dir, manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    shutil.copy2(ROOT / "src" / "models.py", pkg / "src" / "models.py")
    # models.py imports text_encoder as a top-level module; the file lives in scripts/,
    # not src/, so copy it from its real location into the package's src/ directory.
    shutil.copy2(ROOT / "scripts" / "text_encoder.py", pkg / "src" / "text_encoder.py")
    bert_dir = ROOT / "resources" / "bert_shared"
    for item in sorted(bert_dir.iterdir()):
        if item.is_file():
            shutil.copy2(item, pkg / "resources" / "bert_shared" / item.name)
    shutil.copy2(run_dir / "data" / "normalizer.npz", pkg / "data" / "normalizer.npz")
    shutil.copy2(input_path, pkg / "data" / "attachment3_input.npz")
    for name in ("q2_predictions.csv", "q2_predictions.json", "validation.json"):
        if (submission / name).exists():
            shutil.copy2(submission / name, pkg / "submission" / name)
    (pkg / "infer.py").write_text(INFER_SCRIPT, encoding="utf-8")
    (pkg / "README.md").write_text(
        "# Q2 frozen inference package\n\n"
        "Rebuild the frozen model and reproduce the 30 Attachment-3 predictions:\n\n"
        "```bash\n"
        "python infer.py --input data/attachment3_input.npz --out /tmp/q2_out --device cuda:0\n"
        "```\n\n"
        "Device note: the frozen predictions shipped in `submission/` were produced on CUDA\n"
        "(`--device cuda:1`).  Re-running on CUDA reproduces `q2_predictions.csv` bit-for-bit\n"
        "(clean-process check: logits/raw max abs diff 0.0).  Running on CPU reproduces the same\n"
        "30 rows and the same polarities, but the float32 logits drift by up to ~5e-4, so the\n"
        "CPU output is not byte-identical to the shipped CSV.\n\n"
        "Environment note: if a broken user-site TensorFlow/protobuf shadows the interpreter,\n"
        "run with `PYTHONNOUSERSITE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_FLAX=0`.\n\n"
        "The shared text encoder is stored once under `resources/bert_shared`; the candidate\n"
        "checkpoint holds fusion/head weights only.  See `model/config.json` and `package.json`.\n",
        encoding="utf-8")

    logits, raw, prob, (cls, intensity) = frozen_logits(run_dir, manifest, input_path,
                                                        torch.device(args.device))
    np.savez(pkg / "model" / "frozen_logits.npz", logits=logits, raw=raw, prob=prob)
    verification = {"skipped": bool(args.skip_verify)}
    if not args.skip_verify:
        verification = clean_process_check(pkg, pkg / "data" / "attachment3_input.npz",
                                           logits, raw, csv_path, args.device)

    zip_path = submission / "q2_package.zip"
    zip_bytes = zip_package(pkg, zip_path)
    q1_bytes = args.q1_zip.stat().st_size if (args.q1_zip and args.q1_zip.exists()) else None
    files = {}
    for path in sorted(pkg.rglob("*")):
        if path.is_file():
            files[path.relative_to(pkg).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    package_bytes = sum(item["bytes"] for item in files.values())
    report = {
        "schema": "q2-package-1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "freeze_manifest": str(freeze), "freeze_manifest_sha256": sha256_file(freeze),
        "deployed_run_id": manifest["deployed_run_id"], "architecture": manifest["architecture"],
        "state_dict_tensors": n_tensors,
        "package_dir": str(pkg), "package_bytes_uncompressed": package_bytes,
        "package_zip": str(zip_path), "package_zip_bytes": zip_bytes,
        "q1_zip": None if args.q1_zip is None else str(args.q1_zip), "q1_zip_bytes": q1_bytes,
        "combined_q1_q2_bytes": None if q1_bytes is None else zip_bytes + q1_bytes,
        "combined_limit_bytes": 50000000,
        "combined_within_limit": None if q1_bytes is None else (zip_bytes + q1_bytes) <= 50000000,
        "shared_encoder_copies": 1,
        "files": files, "clean_process_verification": verification,
        "publication_policy": manifest["publication"]["policy"],
    }
    (pkg / "package.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    run_manifest = None
    if not args.skip_run_manifest:
        target, count = write_run_manifest(run_dir)
        run_manifest = {"path": str(target), "entries": count}
    print(json.dumps({"package": str(pkg), "zip_bytes": zip_bytes,
                      "combined_q1_q2_bytes": report["combined_q1_q2_bytes"],
                      "clean_process_verified": verification.get("passed", False),
                      "run_manifest": run_manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
