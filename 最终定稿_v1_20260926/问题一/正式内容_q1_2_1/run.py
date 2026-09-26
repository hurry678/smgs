#!/usr/bin/env python3
"""第一问统一命令入口；所有路径相对于本目录。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for key, relative in {
    "HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch",
    "NUMBA_CACHE_DIR": ".cache/numba", "MPLCONFIGDIR": ".cache/matplotlib",
    "XDG_CACHE_HOME": ".cache", "TMPDIR": ".cache/tmp",
}.items():
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "download", "check-text", "align-mfa", "extract",
                                           "reaggregate", "validate", "figures", "package", "summarize",
                                           "audit-boundaries", "report"])
    parser.add_argument("--config", default="configs/q1.json")
    parser.add_argument("--input-dir", help="Folder containing original label-100.xlsx and video subfolders")
    parser.add_argument("--pilot", action="store_true", help="Only short/medium/long train-overlap clips")
    parser.add_argument("--sample", help="Exact sample ID (use --sample=ID for leading '-')")
    parser.add_argument("--force", action="store_true", help="Recompute selected sample stages")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Allow validation/package output while strict release gates remain open")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.command == "prepare":
        from prepare import main as action
        action(args.input_dir)
    elif args.command == "download":
        from download_resources import main as action
        action()
    elif args.command == "check-text":
        from check_text import main as action
        action(config)
    elif args.command == "align-mfa":
        from automatic_alignment import ensure_mfa_alignments
        from modalities import decode_audio
        from prepare import sha256
        rows = json.loads((ROOT / "resources/manifest.json").read_text())
        selected = json.loads((ROOT / "configs/pilot.json").read_text()) if args.pilot else None
        if args.sample:
            selected = [args.sample]
        if selected is not None:
            rows = [row for row in rows if row["sample_id"] in selected]
        for row in rows:
            decode_audio(row, config["audio"], sha256(ROOT / row["source_file"]))
        print(json.dumps(ensure_mfa_alignments(rows, config, force=args.force),
                         ensure_ascii=False, indent=2))
    elif args.command in ("extract", "reaggregate"):
        from pipeline import run
        selected = json.loads((ROOT / "configs/pilot.json").read_text()) if args.pilot else None
        if args.sample:
            selected = [args.sample]
        if args.command == "extract":
            run(config, selected, args.force)
        else:
            from reaggregate import main as action
            action(config, selected)
    elif args.command == "validate":
        from validate import main as action
        action(pilot=args.pilot, diagnostic=args.diagnostic, config=config)
    elif args.command == "figures":
        from figures import main as action
        action()
    elif args.command == "package":
        from package import main as action
        action(diagnostic=args.diagnostic, config=config)
    elif args.command == "summarize":
        from summarize_results import main as action
        action()
    elif args.command == "audit-boundaries":
        from audit_boundaries import main as action
        action(force=args.force)
    elif args.command == "report":
        from report_completion import main as action
        action()


if __name__ == "__main__":
    main()
