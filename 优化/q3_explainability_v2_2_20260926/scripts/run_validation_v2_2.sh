#!/usr/bin/env bash
set -euo pipefail

REPO=/data2/hy/smgs
EXT="$REPO/问题三/q3_explainability_v2_2_20260926"
LEGACY=/data2/hy/cts/e_problem
OLD=/data2/hy/q3_runs/q3_explainability_v2_20260925
RUN=/data2/hy/q3_runs/q3_explainability_v2_2_20260926
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
DEVICE=cuda:2

export PYTHONPATH="$OLD/python_deps"

CHECKPOINTS=(
  "$LEGACY/artifacts/q2/final/M0_seed42.pt"
  "$LEGACY/artifacts/q2/final/M0_seed43.pt"
  "$LEGACY/artifacts/q2/final/M0_seed44.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add/M0_seed45.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add/M0_seed46.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add/M0_seed47.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add2/M0_seed48.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add2/M0_seed49.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add2/M0_seed50.pt"
)

run_all() {
  for W in 3 5 10; do
    mkdir -p "$RUN/validation/w$W"
    echo "[$(date --iso-8601=seconds)] START validation w=$W"
    /usr/bin/time -v -o "$RUN/logs/validation_w$W.time" \
      "$PY" "$EXT/scripts/q3_explain_core_v2.py" \
      --mode validation \
      --data "$LEGACY/data/q2/q2_data.npz" \
      --offsets "$RUN/q2_valid_offsets.npz" \
      --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
      --checkpoints "${CHECKPOINTS[@]}" \
      --freeze-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
      --out-dir "$RUN/validation/w$W" \
      --split valid --n 728 \
      --window-size "$W" --stride "$W" \
      --point-n 128 --top-k 2 --batch-size 64 --seed 2026 \
      --device "$DEVICE" \
      2>&1 | tee "$RUN/logs/validation_w$W.log"
    echo "[$(date --iso-8601=seconds)] DONE validation w=$W"
  done
}

rc=0
run_all || rc=$?
echo "$rc" > "$RUN/logs/validation.exit_code"
date --iso-8601=seconds > "$RUN/logs/validation.finished_at"
exit "$rc"
