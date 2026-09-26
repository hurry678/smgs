#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/data2/hy/smgs_git_808b602
EXT="$REPO/问题三/q3_explainability_v2_2_20260926"
LEGACY=/data2/hy/cts/e_problem
OLD=/data2/hy/q3_runs/q3_explainability_v2_20260925
RUN=/data2/hy/q3_runs/q3_explainability_v2_2_20260926
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
DEVICE=cuda:2
RAW="/data2/hy/test_5/E题/E题数据/E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件"
MFA="$OLD/.mfa/bin/mfa"
MFA_DICT="$OLD/resources/mfa/english_mfa_dictionary_v3.1.0.dict"
MFA_ACOUSTIC="$OLD/resources/mfa/english_mfa_acoustic_v3.1.0.zip"
MFA_G2P="$OLD/resources/mfa/english_us_mfa_g2p_v3.0.0.zip"
export PYTHONPATH="$OLD/python_deps"

mkdir -p "$RUN/logs" "$RUN/attachment4_data" "$RUN/final_attachment4" "$RUN/submission"

finish() {
  rc=$?
  echo "$rc" > "$RUN/logs/postprocess.exit_code"
  if [ "$rc" -eq 0 ]; then
    echo "DONE postprocess $(date -Is)"
  else
    echo "FAILED postprocess rc=$rc $(date -Is)"
  fi
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

echo "WAIT validation $(date -Is)"
deadline=$(( $(date +%s) + 21600 ))
while [ ! -f "$RUN/logs/validation.exit_code" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "TIMEOUT waiting for validation.exit_code"
    exit 124
  fi
  sleep 20
done
validation_rc=$(tr -d '[:space:]' < "$RUN/logs/validation.exit_code")
if [ "$validation_rc" != "0" ]; then
  echo "validation failed rc=$validation_rc"
  exit 1
fi
for w in 3 5 10; do
  test -s "$RUN/validation/w$w/q3_validation_report.json"
done
echo "validation complete $(date -Is)"

echo "START freeze parameters $(date -Is)"
"$PY" "$EXT/scripts/q3_select_freeze_v2.py" \
  --report "3=$RUN/validation/w3/q3_validation_report.json" \
  --report "5=$RUN/validation/w5/q3_validation_report.json" \
  --report "10=$RUN/validation/w10/q3_validation_report.json" \
  --out "$RUN/freeze_parameters.json" \
  --expected-n 728 --fallback-window 5 \
  --point-scan-n 128 --top-k 2 --batch-size 64 --seed 2026 \
  --q2-freeze-manifest "$REPO/优化/q2yhv2_1/server_results_20260925/freeze_manifest.json" \
  --q2-version-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
  2>&1 | tee "$RUN/logs/freeze_parameters.log"
W=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_window_size"])' "$RUN/freeze_parameters.json")
case "$W" in 3|5|10) ;; *) echo "INVALID W=$W"; exit 1;; esac
SELECTED_REPORT="$RUN/validation/w$W/q3_validation_report.json"
echo "FROZEN W=$W $(date -Is)"

echo "START attachment4 preflight $(date -Is)"
"$PY" "$EXT/scripts/q3_preflight_v2.py" \
  --scope attachment4 \
  --protocol "$EXT/configs/protocol.json" \
  --repo-root "$REPO" \
  --legacy-root "$LEGACY" \
  --raw-dir "$RAW" \
  --mfa-executable "$MFA" \
  --dictionary "$MFA_DICT" \
  --acoustic-model "$MFA_ACOUSTIC" \
  --g2p-model "$MFA_G2P" \
  --out "$RUN/preflight_attachment4.json" \
  2>&1 | tee "$RUN/logs/preflight_attachment4.log"

cp "$OLD/attachment4_data/attachment4_aligned.npz" "$RUN/attachment4_data/"
cp "$OLD/attachment4_offsets.npz" "$RUN/"
cp "$OLD/alignment_audit.json" "$RUN/"
echo "7572f8bd6fdfb0e40ff0f7446894a6ace44d63a90a17c7ee2a6376d5b3b7567b  $RUN/attachment4_data/attachment4_aligned.npz" | sha256sum -c -
echo "36b8ec63d87e063c9598590e114d40e7ee4ea2ed45e9d4fd97f75be1d6630c5b  $RUN/attachment4_offsets.npz" | sha256sum -c -
echo "2002a77e72809677da74151534e3f8b44f51036b258952a06efd83a0048f886d  $RUN/alignment_audit.json" | sha256sum -c -
"$PY" - "$RUN/alignment_audit.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
assert d["status"] == "PASS", d.get("status")
assert int(d["mfa_aligned_samples"]) == 20, d.get("mfa_aligned_samples")
assert int(d["fallback_samples"]) == 0, d.get("fallback_samples")
print({"status": "PASS", "mfa_aligned_samples": 20, "fallback_samples": 0})
PY

echo "START attachment4 inference W=$W $(date -Is)"
/usr/bin/time -v "$PY" "$EXT/scripts/q3_explain_core_v2.py" \
  --mode attachment4 \
  --data "$LEGACY/data/q2/q2_data.npz" \
  --attachment4 "$RUN/attachment4_data/attachment4_aligned.npz" \
  --offsets "$RUN/attachment4_offsets.npz" \
  --audit "$RUN/alignment_audit.json" \
  --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --freeze-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
  --out-dir "$RUN/final_attachment4" \
  --window-size "$W" --stride "$W" \
  --point-n 20 --top-k 2 --batch-size 64 --seed 2026 \
  --device "$DEVICE" \
  2>&1 | tee "$RUN/logs/attachment4_inference.log"

echo "START package $(date -Is)"
"$PY" "$EXT/scripts/q3_build_outputs_v2.py" \
  --raw-json "$RUN/final_attachment4/q3_attachment4_raw.json" \
  --out-dir "$RUN/submission" \
  --attachment4 "$RUN/attachment4_data/attachment4_aligned.npz" \
  --model-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
  --explain-script "$EXT/scripts/q3_explain_core_v2.py" \
  --alignment-script "$EXT/scripts/q3_prepare_alignment_v2.py" \
  --math-script "$EXT/scripts/q3v2_math.py" \
  --data "$LEGACY/data/q2/q2_data.npz" \
  --offsets "$RUN/attachment4_offsets.npz" \
  --audit "$RUN/alignment_audit.json" \
  --validation-report "$SELECTED_REPORT" \
  --explanation-parameters "$RUN/freeze_parameters.json" \
  2>&1 | tee "$RUN/logs/package.log"

echo "START result comparison $(date -Is)"
"$PY" "$EXT/scripts/compare_q3v2_2_results.py" \
  --old-raw "$OLD/final_attachment4/q3_attachment4_raw.json" \
  --new-raw "$RUN/final_attachment4/q3_attachment4_raw.json" \
  --out "$RUN/result_comparison.json" \
  2>&1 | tee "$RUN/logs/result_comparison.log"

echo "START independent verification $(date -Is)"
"$PY" "$EXT/scripts/verify_q3v2.py" \
  --repo-root "$REPO" \
  --extension-root "$EXT" \
  --run-dir "$RUN" \
  --out "$RUN/verification.json" \
  2>&1 | tee "$RUN/logs/verify.log"

echo "START audit export $(date -Is)"
"$PY" "$EXT/scripts/export_q3v2_audit.py" \
  --extension-root "$EXT" \
  --run-dir "$RUN" \
  --export-dir "$RUN/audit_export" \
  2>&1 | tee "$RUN/logs/export_audit.log"
(
  cd "$RUN/audit_export"
  sha256sum -c SHA256SUMS.txt
)

echo "COMPLETE $(date -Is)"