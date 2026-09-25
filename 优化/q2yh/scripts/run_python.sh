#!/bin/bash
# Server-side launcher for q2-plan-1.0.
# Isolates user site-packages (broken TF/protobuf) and pins the exact interpreter.
set -euo pipefail
export PYTHONNOUSERSITE=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
exec /data2/hy/test_5/problem2-venv/bin/python "$@"
