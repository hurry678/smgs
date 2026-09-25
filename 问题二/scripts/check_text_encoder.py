#!/usr/bin/env python3
"""Verify frozen dropout and pre-encoding mask invariance using bundled weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from protocol_core import interface_masks
from text_encoder import FrozenTextEncoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        ap.error("output must be a new file under 问题二")
    torch.manual_seed(42)
    tb = np.zeros((2, 3, 50), dtype=np.int64)
    tb[:, 0, :7] = [101, 2023, 2003, 1037, 2204, 3185, 102]
    tb[:, 1, :7] = 1
    domain, obs = interface_masks(tb, np.ones((2, 50, 74)), np.ones((2, 50, 35)))
    visible = obs[:, 0].copy()
    visible[0, 2:4] = False
    visible[1] = False
    changed = tb.copy()
    changed[:, 0][domain & ~visible] = 999
    changed[:, 2][domain & ~visible] = 1
    model = FrozenTextEncoder()
    model.train()
    a = model(torch.from_numpy(tb), torch.from_numpy(domain), torch.from_numpy(visible))
    b = model(torch.from_numpy(changed), torch.from_numpy(domain), torch.from_numpy(visible))
    repeat = model(torch.from_numpy(tb), torch.from_numpy(domain), torch.from_numpy(visible))
    difference = float((a-b).abs().max())
    checks = {
        "finite_output": bool(torch.isfinite(a).all()),
        "hidden_input_invariance": difference <= 1e-7,
        "repeat_determinism": bool(torch.equal(a, repeat)),
        "empty_text_zero_output": bool((a[1] == 0).all()),
        "frozen_gradients_disabled": not any(p.requires_grad for p in model.bert.parameters()),
        "encoder_stays_eval_after_train": not model.bert.training,
        "shape_2_50_128": list(a.shape) == [2, 50, 128]
    }
    result = {"schema": "q2-text-adapter-check-1", "passed": all(checks.values()),
              "checks": checks, "masked_input_max_abs_difference": difference,
              "scope": "interface_and_frozen_encoder_only_no_task_training"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
