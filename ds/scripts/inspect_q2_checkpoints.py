import json
from pathlib import Path
import torch
for name in ["B2_seed42", "B3_seed42", "B4_seed42", "B5_seed42", "M0_seed42"]:
    p = Path("artifacts/q2/final") / f"{name}.pt"
    c = torch.load(p, map_location="cpu")
    info = c.get("info", {})
    print(json.dumps({
        "checkpoint": name,
        "kind": c.get("kind"),
        "seed": c.get("seed"),
        "params_total": info.get("params_total"),
        "params_trainable": info.get("params_trainable"),
        "file_bytes": p.stat().st_size,
        "model_config": c.get("model_config"),
    }, ensure_ascii=False))
