"""Whitelist packaging with ZIP integrity, 100-sample round-trip, and byte budget checks."""
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from prepare import ROOT, dump, sha256
from pipeline import json_hash, provenance
from acceptance import validation_fingerprint, release_ready


def main(diagnostic=False, config=None):
    validation = json.loads((ROOT / "reports/validation.json").read_text())
    config = config or json.loads((ROOT / "configs/q1.json").read_text())
    current_provenance, _, input_hashes = provenance(config)
    if validation.get("provenance_base_sha256") != json_hash(current_provenance):
        raise ValueError("Validation is stale relative to current code, models, configuration or annotations")
    if validation.get("validation_fingerprint") != validation_fingerprint(ROOT, sha256):
        raise ValueError("Validation is stale: output, review, native evidence or release policy changed")
    for relative, expected_sha in input_hashes.items():
        source = ROOT / relative
        if source.exists() and sha256(source) != expected_sha:
            raise ValueError(f"Input changed after validation: {relative}")
    if not validation.get("structural_passed") or validation["checked_samples"] != 100:
        raise ValueError("Full 100-sample structural validation must pass before packaging")
    decisions = [json.loads(line) for line in (ROOT / "outputs/acceptance.jsonl").read_text().splitlines()]
    if decisions != validation["acceptance"]:
        raise ValueError("Acceptance file differs from validated decisions")
    strict_ready = release_ready(decisions, 100, validation["structural_passed"])
    if strict_ready != validation["strict_release_passed"]:
        raise ValueError("Validation/package release policy mismatch")
    acceptance = {r["sample_id"]: r for r in decisions}
    if not diagnostic and not strict_ready:
        raise ValueError("Strict release gates are open; use --diagnostic only for a clearly marked diagnostic archive")
    if not diagnostic:
        mfa = json.loads((ROOT / "intermediate/mfa/run.json").read_text())
        audit = json.loads((ROOT / "reports/boundary_audit.json").read_text())
        raw_hashes = audit["raw_output_sha256"]
        mapping = [json.loads(line) for line in (ROOT / "outputs/alignment.jsonl").read_text().splitlines()]
        expected_mfa_ids = {m["sample_id"] for m in mapping if m["audio_observed"]}
        if (mfa.get("execution_schema") != "mfa-execution-2"
                or {r["sample_id"] for r in mfa["samples"]} != expected_mfa_ids):
            raise ValueError("Full MFA execution evidence is missing")
        audited = {r["sample_id"]: r for r in audit["samples"]}
        if len(audit["samples"]) != 100 or set(audited) != set(acceptance):
            raise ValueError("Audio-only audit does not cover all 100 samples")
        for meta in mapping:
            sid = meta["sample_id"]
            if meta["automatic_boundary_evidence"]["mfa_batch_fingerprint"] != mfa["fingerprint"]:
                raise ValueError(f"Stale MFA execution evidence: {sid}")
            if audited[sid]["artifact_fingerprint"] != meta["artifact_fingerprint"]:
                raise ValueError(f"Stale audio-only audit: {sid}")
            if sha256(ROOT / "intermediate/boundary_audit" / (sid + ".json")) != raw_hashes[sid]:
                raise ValueError(f"Changed raw ASR audit: {sid}")
        for row in mfa["samples"]:
            if row["result"] == "aligned" and sha256(ROOT / row["output"]) != row["sha256"]:
                raise ValueError(f"Changed MFA raw output: {row['sample_id']}")
    files = [ROOT / name for name in ("README.md", "实施计划.md", "requirements.txt",
                                      "requirements.lock.txt", "run.py")]
    for directory, pattern in (("src", "*.py"), ("tests", "*.py"), ("configs", "*.json"),
                               ("outputs/features", "*.npz"), ("outputs/figures", "*"),
                               ("outputs/audit_audio", "*"), ("intermediate/boundary_audit", "*.json"),
                               ("resources/models/bert", "*")):
        files += sorted((ROOT / directory).glob(pattern))
    for name in (
        "reports/第一问解题报告.md", "reports/特征字典.md",
        "reports/自动边界协议与100条最终结果_2026-09-24.md",
        "annotations/boundary_review.csv", "annotations/manual_phrases.jsonl",
        "outputs/alignment.jsonl", "outputs/q1_summary.csv", "outputs/acceptance.jsonl",
        "resources/manifest.json", "resources/input_manifest.json", "resources/model_sources.json",
        "resources/resource_lock.json", "resources/audit_model_source.json",
        "resources/reference/data_audit.json", "resources/reference/ROADMAP.md",
        "resources/reference/题目原文.txt",
        "reports/input_audit.json", "reports/environment.json", "reports/tokenizer_check.json",
        "reports/validation.json", "reports/result_statistics.json",
        "reports/review_queue.json", "reports/face_retry.json",
        "reports/audio_missing_evidence.json",
        "reports/last_run.json", "reports/boundary_audit.json",
        "reports/边界独立核验与修正说明.md", "reports/典型样本与全量结果.md",
        "logs/tests_q1_2_1.log",
        "intermediate/mfa/run.json",
    ):
        path = ROOT / name
        if path.exists():
            files.append(path)
    files = sorted(set(p for p in files if p.is_file()))
    text_suffixes = {".md", ".py", ".json", ".jsonl", ".csv", ".txt", ".log"}
    user_dir = Path.home().name
    forbidden = ("/" + "Users" + "/", "\\" + "Users" + "\\", user_dir)
    anonymity_hits = []
    for path in files:
        if path.suffix.lower() in text_suffixes:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for marker in forbidden:
                if marker.lower() in text.lower():
                    anonymity_hits.append({"path": str(path.relative_to(ROOT)), "marker": marker})
    if anonymity_hits:
        raise ValueError(f"Anonymity scan failed: {anonymity_hits}")
    # This manifest fingerprints actual source, config, outputs and included weights once at packaging.
    manifest = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    dump(ROOT / "reports/delivery_manifest.json", manifest)
    archive = ROOT / "submission" / ("第一问诊断包.zip" if diagnostic else "第一问交付包.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files + [ROOT / "reports/delivery_manifest.json"]:
            z.write(p, str(p.relative_to(ROOT)))
    expected = {r["sample_id"] for r in json.loads((ROOT / "resources/manifest.json").read_text())}
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity failed")
        npzs = [n for n in z.namelist() if n.startswith("outputs/features/") and n.endswith(".npz")]
        checked = set()
        for name in npzs:
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as data:
                sid = str(data["sample_id"])
                checked.add(sid)
                import hashlib
                if hashlib.sha256(z.read(name)).hexdigest() != acceptance[sid]["feature_sha256"]:
                    raise ValueError(f"Archive feature differs from accepted artifact: {sid}")
                if data["features"].shape[1] != 150:
                    raise ValueError(f"Invalid feature width in {name}")
                if data["features"].shape != data["valid"].shape:
                    raise ValueError(f"Feature/mask mismatch in {name}")
                if not np.isfinite(data["features"]).all():
                    raise ValueError(f"Non-finite feature in {name}")
                if not diagnostic and not acceptance[sid]["release_eligible"]:
                    raise ValueError(f"Strict release rejects unaccepted artifact: {name}")
        if len(npzs) != 100 or checked != expected:
            raise ValueError("Archive feature coverage mismatch")
        mapping = [json.loads(line) for line in z.read("outputs/alignment.jsonl").decode().splitlines()]
        if {r["sample_id"] for r in mapping} != expected:
            raise ValueError("Archive mapping coverage mismatch")
    report = {"archive": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
              "decimal_mb": archive.stat().st_size / 1_000_000, "sha256": sha256(archive),
              "within_50000000_bytes": archive.stat().st_size <= 50_000_000,
              "files": len(files) + 1, "feature_records_read_back": len(checked),
              "zip_crc_passed": True, "mapping_coverage_passed": True,
              "anonymity_path_scan_passed": not anonymity_hits,
              "included_bert_weights": True, "includes_raw_video_or_environment": False,
              "includes_selected_source_audio_copies": True, "includes_asr_audit_weights": False,
              "mode": "diagnostic" if diagnostic else "strict_release",
              "strict_release_ready": strict_ready,
              "automatically_accepted_samples": sum(
                  d["aligned_accepted"] and not d["human_approved"] for d in decisions),
              "audio_missing_samples": sum(
                  d.get("audio_missing_evidence") is not None for d in decisions),
              "verified_source_exception_samples": sum(d["source_exception_verified"] for d in decisions),
              "scope": "Q1 package only; final Q1/Q2/Q3 combined archive must be measured separately."}
    dump(ROOT / "reports/package_check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["within_50000000_bytes"]:
        raise ValueError("Q1 package exceeds conservative whole-submission ceiling")
