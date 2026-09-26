#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data2/hy/smgs
EXT="$REPO/问题三/q3_explainability_v2_2_20260926"
LEGACY=/data2/hy/cts/e_problem
OLD=/data2/hy/q3_runs/q3_explainability_v2_20260925
RUN=/data2/hy/q3_runs/q3_explainability_v2_2_20260926
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
DEVICE=cuda:2
: "${Q3_RUN_ID:?Set Q3_RUN_ID once and pass the same value to both runner scripts}"

export PYTHONPATH="$OLD/python_deps"
VALIDATION_ROOT="$RUN/validation_runs/$Q3_RUN_ID"
STATUS="$RUN/logs/validation_status_$Q3_RUN_ID.json"
mkdir -p "$RUN/logs" "$VALIDATION_ROOT"

finish() {
  rc=$?
  finished_at=$(date --iso-8601=seconds)
  tmp="$STATUS.tmp.$$"
  printf '{"run_id":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$Q3_RUN_ID" "$rc" "$finished_at" > "$tmp"
  mv "$tmp" "$STATUS"
  trap - EXIT
  exit "$rc"
}
trap finish EXIT

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
    mkdir -p "$VALIDATION_ROOT/w$W"
    echo "[$(date --iso-8601=seconds)] START validation w=$W"
    /usr/bin/time -v -o "$RUN/logs/validation_${Q3_RUN_ID}_w$W.time" \
      "$PY" "$EXT/scripts/q3_explain_core_v2.py" \
      --mode validation \
      --data "$LEGACY/data/q2/q2_data.npz" \
      --offsets "$RUN/q2_valid_offsets.npz" \
      --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
      --checkpoints "${CHECKPOINTS[@]}" \
      --freeze-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
      --out-dir "$VALIDATION_ROOT/w$W" \
      --split valid --n 728 \
      --window-size "$W" --stride "$W" \
      --point-n 128 --top-k 2 --batch-size 64 --seed 2026 \
      --device "$DEVICE" \
      2>&1 | tee "$RUN/logs/validation_${Q3_RUN_ID}_w$W.log"
    echo "[$(date --iso-8601=seconds)] DONE validation w=$W"
  done
}

run_all
