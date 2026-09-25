# 论文图表目录（自动生成）

运行 `python paper/generate_paper_figures.py` 可重建本目录。

| 图片 | 建议图注 | 数据来源 | 写作用途 |
|---|---|---|---|
| `figures/q1_data_cleaning.svg` | 图1 附件1数据清洗覆盖与合格情况 | `artifacts/q1/data_cleaning/quality_report.json` | 说明100条样本全部通过文本、标签、音视频流、时长和完整解码检查。 |
| `figures/q1_alignment_methods.svg` | 图2 时序对齐方法对比 | `artifacts/q1/comparison/server_all_100_repeat10/comparison_metrics.csv` | 展示不同对齐方法的探针性能、时长适配与停顿质量权衡。 |
| `figures/q1_modality_components.svg` | 图3 文本、语音与视觉候选特征对比 | `artifacts/q1/comparison/server_all_100_repeat10/comparison_metrics.csv` | 支撑BERT、语音40维和视觉视图的选型讨论。 |
| `figures/q2_model_comparison.svg` | 图4 问题二候选模型族对比 | `artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json` | 说明掩码感知模型与备选融合结构的性能差异。 |
| `figures/q2_missing_robustness.svg` | 图5 问题二局部缺失条件下的鲁棒性 | `artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json` | 分析缺失类型组合对MAE与Macro-F1的影响；不能写成整模态永久删除。 |
| `figures/q2_seed_stability_m0.svg` | 图6 M0 随机种子稳定性 | `artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json` | 展示随机种子对鲁棒性指标的影响，支撑9模型集成决策。 |
| `figures/q2_seed_stability_b4.svg` | 图6 B4 随机种子稳定性 | `artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json` | 展示随机种子对鲁棒性指标的影响，支撑9模型集成决策。 |
| `figures/q2_prediction_distribution.svg` | 图7 附件3预测极性分布 | `submission/q2_predictions.csv` | 报告附件3推理结果；不得据此计算监督准确率。 |
| `figures/q3_shapley_modalities.svg` | 图8 模态级Shapley绝对作用份额 | `artifacts/q3/final_validation/q3_validation_report.json` | 量化文本、语音、视觉对类别与强度预测的平均绝对作用。 |
| `figures/q3_sensitivity.svg` | 图9 解释窗口大小敏感性 | `artifacts/q3/sensitivity_w3/、w5/、w10/` | 分析窗口尺度对证据集中度与控制组胜率的影响。 |
| `figures/q3_local_occlusion.svg` | 图10 局部遮挡忠实性对照 | `artifacts/q3/final_validation/q3_validation_report.json` | 说明连续窗口证据与对照组的差异，不宣称物理因果。 |
| `figures/q3_attachment4_distribution.svg` | 图11 附件4预测与主模态分布 | `submission/q3_predictions_explanations.csv` | 报告附件4推理输出；不得据此计算监督指标。 |
| `figures/q3_evidence_levels.svg` | 图12 附件4关键证据等级分布 | `submission/q3_predictions_explanations.csv` | 展示证据等级分布，并区分解释等级与预测正确性。 |

## 表格

所有 CSV 位于 `paper/tables/`，字段名和数值保留原始精度，可直接用于论文表格或进一步绘图。
