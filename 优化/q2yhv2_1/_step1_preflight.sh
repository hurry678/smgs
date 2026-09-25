set -e
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
cd "/data2/hy/smgs/问题二/q2_ds_optimization_v2_1_20260925"
echo "==== unit tests ===="
$PY -m pytest -q tests/test_protocol.py 2>&1 | tail -20 || $PY tests/test_protocol.py 2>&1 | tail -20
echo "==== env ===="
$PY - <<'PY'
import numpy, sklearn, torch
print({"numpy": numpy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
       "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()})
PY
echo "==== preflight ===="
$PY scripts/run_q2hc.py preflight --legacy-root /data2/hy/cts/e_problem --work-dir /data2/hy/q2_runs/q2_ds_hc_v2_1_20260925
echo "==== resolved_inputs summary ===="
$PY - <<'PY'
import json
from pathlib import Path
p = Path("/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925/resolved_inputs.json")
d = json.loads(p.read_text())
print(json.dumps({k: v for k, v in d.items() if k not in ("baseline_checkpoints", "script_checks", "splits")}, ensure_ascii=False, indent=2))
print("script_checks:", json.dumps(d.get("script_checks"), ensure_ascii=False, indent=1))
print("n_checkpoints:", len(d.get("baseline_checkpoints", [])))
print("failures:", d.get("failures"))
PY