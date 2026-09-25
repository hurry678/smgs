#!/bin/bash
# Re-run of the invalidated EMA arm (see failures.jsonl entry status=invalidated_rerun).
set -u
cd /data2/hy/cts/e_problem/q2yh
RUN=runs/q2run_20260925_srv
./scripts/run_python.sh scripts/rerun_arm.py \
  --run-dir "$RUN" \
  --stage losses \
  --run-id s5_B3_EMA_seed42 \
  --architecture bigru_content_gate_time_pool \
  --seed 42 \
  --device cuda:1 \
  --reason "invalid first run: engine computed EMA weights but validated and deployed the raw weights, so this arm was numerically identical to s4_B3_O3rho0.02_seed42; src/engine.py fixed and the arm re-run" \
  --config '{"loss":"L1","optimizer":"O3","regression_weight":1.0,"huber_beta":0.5,"rho":0.02,"lr":0.0005,"ema":true,"ema_decay":0.99}'
echo "=== rerun_ema exit $? $(date -Is) ==="
