#!/usr/bin/env bash
set -euo pipefail

# version_comparison
/data2/hy/anaconda3/envs/secgpt-vllm/bin/python /data2/hy/cts/e_problem/ds/q1_comparison_extension/scripts/compare_q1_versions.py --current-q1-dir /data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925/smoke/materialized/A_current_audited --git-result-dir /data2/hy/cts/e_problem/ds/artifacts/q1/server_final_100/final_100 --output-dir /data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925/smoke/version_comparison

# build_candidate_vectors
/data2/hy/anaconda3/envs/secgpt-vllm/bin/python /data2/hy/cts/e_problem/ds/q1_comparison_extension/scripts/build_q1_probe_vectors.py --current-q1-dir /data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925/smoke/materialized/A_current_audited --git-result-dir /data2/hy/cts/e_problem/ds/artifacts/q1/server_final_100/final_100 --labels-csv /data2/hy/cts/e_problem/ds/artifacts/q1/data_cleaning/qualified_samples.csv --output-dir /data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925/smoke/candidate_vectors

# nested_representation_probe
/data2/hy/anaconda3/envs/secgpt-vllm/bin/python /data2/hy/cts/e_problem/ds/q1_comparison_extension/scripts/evaluate_q1_candidate_vectors.py --candidate-dir /data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925/smoke/candidate_vectors --output-dir /data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925/smoke/representation_probe --seeds 20260924 --outer-folds 5 --inner-folds 3
