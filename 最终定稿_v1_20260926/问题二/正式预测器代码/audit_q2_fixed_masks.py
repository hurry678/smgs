#!/usr/bin/env python3
"""Verify that fixed-mask Q2 evaluations used identical missing conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payloads = {}
    for path in sorted(args.eval_root.glob("*/valid_valid_evaluation.json")):
        data = load(path)
        key = f"{data['kind']}_seed{data['seed']}"
        payloads[key] = data
    if not payloads:
        raise SystemExit("No evaluations found")

    seeds = {int(v.get("mask_seed", -1)) for v in payloads.values()}
    replicas = {int(v.get("replicas", -1)) for v in payloads.values()}
    mismatches = []
    reference_key = sorted(payloads)[0]
    reference = payloads[reference_key]

    ref_random = {
        (row["subset"], float(row["ratio"]), i): h
        for row in reference["conditions"]
        for i, h in enumerate(row["mask_sha256"])
    }
    ref_position = {
        (row["subset"], float(row["ratio"]), row["position"]): row["mask_sha256"]
        for row in reference["positions"]
    }

    for key, data in payloads.items():
        random_map = {
            (row["subset"], float(row["ratio"]), i): h
            for row in data["conditions"]
            for i, h in enumerate(row["mask_sha256"])
        }
        position_map = {
            (row["subset"], float(row["ratio"]), row["position"]): row["mask_sha256"]
            for row in data["positions"]
        }
        for condition, digest in random_map.items():
            if ref_random.get(condition) != digest:
                mismatches.append({"candidate": key, "condition_type": "random", "condition": list(condition)})
        for condition, digest in position_map.items():
            if ref_position.get(condition) != digest:
                mismatches.append({"candidate": key, "condition_type": "position", "condition": list(condition)})

    result = {
        "candidate_count": len(payloads),
        "reference_candidate": reference_key,
        "mask_seed_values": sorted(seeds),
        "replica_values": sorted(replicas),
        "random_condition_count": len(ref_random),
        "position_condition_count": len(ref_position),
        "mismatch_count": len(mismatches),
        "passed": len(seeds) == 1 and len(replicas) == 1 and not mismatches,
        "mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
