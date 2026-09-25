# 问题一优化产出审核说明

本目录是问题一优化阶段的完整归档，供外部审核使用。归档时对应的远端仓库提交为 `4522c02069a67789b3e7e898dfe96817239195dc`（`origin/main`，提交信息“Enable frozen Q1 comparison without MFA or raw data”）。服务器工作目录为 `/data2/hy/cts/e_problem`。

## 1. 建议审核顺序

1. `README_服务器执行说明.md`：两轮服务器执行的完整说明，包括环境、命令、结果、回传和口径边界。
2. `benchmark_runs/q1_ab_frozen_20260925/full/result_analysis.md`：**本轮审核重点**，A/B/C 冻结对比正式报告。
3. `benchmark_runs/q1_ab_frozen_20260925/full/preflight.json`、`validation.json`、`failures.jsonl`：预检、验证与失败记录。
4. `benchmark_runs/q1_ab_frozen_20260925/full/representation_probe/`：主指标、逐样本预测和汇总 JSON。
5. `benchmark_runs/q1_ab_frozen_20260925/supplementary_pairwise/`：10,000 次按视频聚类 bootstrap 的附加配对统计。
6. `benchmark_runs/q1_compare_20260925_full/full/result_analysis.md`：第一轮 B（Git 冻结版）内部 23 候选筛选报告；其中 A/C 跨版本阶段当时未执行，不能当作 A/B/C 对比使用。
7. `q1_comparison_extension_v1_20260925/`：协议/工具包输入，随产出归档以便复核复现。

压缩包 `benchmark_runs/q1_full_results.tar.gz` 和 `benchmark_runs/q1_ab_frozen_20260925.tar.gz` 是打包副本；审核以对应解压目录为准。

## 2. 本轮核心结果（fused 视图，5 外层折 × 3 内层折 × 5 种子）

| 候选 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| A_current_audited | **0.5920 ± 0.0295** | **0.4971 ± 0.0299** | 0.6373 ± 0.0586 | 0.2460 ± 0.0617 |
| B_git_frozen | 0.5500 ± 0.0187 | 0.3966 ± 0.0120 | 0.5555 ± 0.0155 | 0.3193 ± 0.0718 |
| C_hybrid_capacity | 0.5860 ± 0.0219 | 0.4531 ± 0.0138 | **0.5441 ± 0.0252** | **0.3801 ± 0.0921** |

- 全量运行：exit 0，wall 48.64 s，CPU 609%，峰值 RSS 148.8 MB，纯 CPU。
- 预检 9/9 passed；结果验证 7/7 passed；`failures.jsonl` 为空。
- A/B 边界一致性（模型间一致性，不是人工真值）：A 端点分歧中位数 0.1000 s、p90 0.5472 s、≤0.25 s 比例 71.79%；B 分别为 0.3100 s、0.8300 s、40.84%。
- 附加配对统计：候选间差值的 95% 区间均跨 0；MAE 方向上 B、C 在 5/5 种子优于 A，效应量 `d_z ≈ -1.13 ~ -1.27`。因此不能宣称三者存在统计显著差异。

## 3. 审核时请特别注意

1. A/B 端点比较是**模型间一致性证据**，不是人工边界误差；A 包 `human_boundary_evaluation.status = not_required`，`human_reviewed_boundaries = 0`。
2. 报告模板 §4.1 的四项人工边界评估在本轮均为“不可得”，不得用模型间一致性替代。
3. B 的“掩码一致率”和“静音伪词边界数”在本轮为**未判定**，不得写成通过。
4. 附件3/4 的无标签样本不得用于调参或模型选择。
5. 第一轮运行 `q1_compare_20260925_full` 只完成了 B 内部组件筛选，不是 A/B/C 系统对比；A/B/C 对比只在第二轮 `q1_ab_frozen_20260925` 中完成。
6. `supplementary_pairwise/` 是附加分析，不属于冻结的 170 文件运行清单，不修改 `run_manifest_sha256.txt`。
7. 第二轮全量目录的冻结清单校验为 169 OK / 1 BAD；唯一 BAD 是设计内正式填充后的 `full/result_analysis.md`，不是文件损坏。
8. 本轮未安装新依赖，未使用 GPU；未修改候选 B 正式产物，未伪造 `boundary_audit.json` 或 `validation.json`。

## 4. 完整性校验

- `PUSH_MANIFEST_SHA256.txt` 记录了本目录除自身外全部文件的 SHA-256，格式兼容 `sha256sum -c`。
- 冻结运行内部仍保留原始 `run_manifest_sha256.txt`；推送归档没有重写它。
- 服务器复现命令见 `README_服务器执行说明.md` 第 9.7 节。
