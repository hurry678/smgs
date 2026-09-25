#!/usr/bin/env bash
set -euo pipefail

# git_screening
/data2/hy/anaconda3/envs/secgpt-vllm/bin/python /data2/hy/cts/e_problem/scripts/compare_q1_methods.py --csv /data2/hy/cts/e_problem/artifacts/q1/data_cleaning/qualified_samples.csv --raw-root '/data2/hy/test_5/E题/E题数据/E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' --text-feature-dir /data2/hy/cts/e_problem/artifacts/q1/server_final_100/final_100 --cache-dir /data2/hy/cts/e_problem/benchmark_runs/q1_compare_20260925_full/full/cache --output-dir /data2/hy/cts/e_problem/benchmark_runs/q1_compare_20260925_full/full/git_screening --probe-repeats 10

# git_paired_statistics
/data2/hy/anaconda3/envs/secgpt-vllm/bin/python /data2/hy/cts/e_problem/scripts/analyze_q1_comparison.py --metrics /data2/hy/cts/e_problem/benchmark_runs/q1_compare_20260925_full/full/git_screening/comparison_metrics.csv --output-dir /data2/hy/cts/e_problem/benchmark_runs/q1_compare_20260925_full/full/git_screening
