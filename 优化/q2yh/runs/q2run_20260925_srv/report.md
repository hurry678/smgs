# 问题二 运行报告 — q2run_20260925_srv

生成时间（UTC）：2026-09-25T07:19:12Z

本报告的全部数值均由本次运行落盘的 JSON/CSV 产物读取，未手工填写；
缺失的项一律写 `not_run`，不用其他来源的数字代替。

## 1. 结论摘要

- 部署模型：`bigru_content_gate_time_pool`（run `s5_B3_L2_seed42`，seed 42）
- 选择状态：`selected_three_seed`；select 库条件数 1
- 相对 B2 简单基线（同种子）的差值：R_MAE -0.0007、R_F1 0.0292、clean F1 0.0564、clean MAE -0.0076（负值表示优于基线）
- 附件3：30 行；CSV SHA256 6c7aa69d4b1b549c41fec06be43a99d6481dee21298b33bc6d3b34f39bf333f0
- 打包：17222868 字节（zip）；Q1+Q2 合计 None 字节，限值 50000000

## 2. 数据与协议

- 协议：`8cc7c0444993df4f1e7528afef98c5dabc457eda34ded7a3587087346bb10973`；文本编码器 `google/bert_uncased_L-2_H-128_A-2@30b0a37ccaaa32f332884b96992754e246e48c5f`（冻结、恒 eval、先遮挡再编码）
- mask 库：select=63, confirm=105, position=27, coupling=60, clean=1
- 归一化：仅用 train 中 O=1 行拟合（train），float64、std 下界 1e-5；valid/test/附件3 不重新 fit
- 缺失定义：D = 0<t<sep（不含 CLS/SEP）；padding≠缺失、UNK≠缺失；附件3/4 不参与训练与选型

## 3. 分阶段执行

| 阶段 | 完成/计划 | 候选数 | 备注 |
|---|---|---|---|
| smoke | 2/2 | 2 |  |
| p0 | 5/5 | 5 |  |
| screen | 4/4 | 4 |  |
| optimize | 9/9 | 9 |  |
| losses | 5/5 | 5 |  |
| multiseed | 6/6 | 6 |  |

## 4. 候选比较（select 库按条件平均 + clean 条件）

| run_id | architecture | seed | 条件数 | R_MAE | R_F1 | R_Accuracy | R_Pearson | 最差条件MAE | clean F1 | clean MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| s5_B3_L2_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.6566 | 0.5442 | 0.5702 | 0.4996 | 0.7192 | 0.5743 | 0.6359 |
| p0_B2_seed42 | three_modality_masked_mean_concat_mlp | 42 | 63 | 0.6573 | 0.515 | 0.5701 | 0.5079 | 0.6939 | 0.5179 | 0.6435 |
| s5_B3_L3_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.6582 | 0.5303 | 0.5835 | 0.5012 | 0.7212 | 0.5604 | 0.6378 |
| s4_B3_O3rho0.02_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.659 | 0.5309 | 0.5834 | 0.5005 | 0.722 | 0.5581 | 0.6385 |
| s4_B3_O3rho0.05_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.6594 | 0.5343 | 0.583 | 0.498 | 0.7214 | 0.5546 | 0.6397 |
| s5_B3_L1w2_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.6612 | 0.5206 | 0.5766 | 0.4965 | 0.7245 | 0.5401 | 0.6411 |
| s6_B2_baseline_seed44 | three_modality_masked_mean_concat_mlp | 44 | 63 | 0.6614 | 0.5244 | 0.5708 | 0.4937 | 0.6951 | 0.5356 | 0.6475 |
| s5_B3_EMA_seed42 | bigru_content_gate_time_pool | 42 | 126 | 0.6623 | 0.5169 | 0.5797 | 0.4982 | 0.7251 | 0.5232 | 0.6472 |
| s3_B1_seed42 | tinybert_masked_mean_linear_dual_head | 42 | 63 | 0.6628 | 0.502 | 0.575 | 0.503 | 0.701 | 0.508 | 0.6481 |
| s5_B3_L1w0.5_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.663 | 0.5183 | 0.577 | 0.499 | 0.7239 | 0.5412 | 0.6422 |
| s4_B1_O4_seed42 | tinybert_masked_mean_linear_dual_head | 42 | 63 | 0.663 | 0.4978 | 0.5694 | 0.5023 | 0.7037 | 0.502 | 0.6479 |
| s6_B3_loss_seed44 | bigru_content_gate_time_pool | 44 | 63 | 0.664 | 0.5384 | 0.5659 | 0.4928 | 0.7214 | 0.5598 | 0.645 |
| s3_B3_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.6647 | 0.5028 | 0.5741 | 0.4942 | 0.7207 | 0.5386 | 0.6442 |
| p0_M1_seed42 | bigru_binary_mask_gate_time_pool | 42 | 63 | 0.6651 | 0.5187 | 0.5755 | 0.4938 | 0.7149 | 0.5474 | 0.647 |
| p0_M1_cleanonly_seed42 | bigru_binary_mask_gate_time_pool | 42 | 63 | 0.6661 | 0.5321 | 0.5725 | 0.4895 | 0.7167 | 0.5689 | 0.6449 |
| s6_B2_baseline_seed43 | three_modality_masked_mean_concat_mlp | 43 | 63 | 0.6665 | 0.5129 | 0.5688 | 0.4824 | 0.7007 | 0.519 | 0.6517 |
| s6_B3_optimizer_seed44 | bigru_content_gate_time_pool | 44 | 63 | 0.6684 | 0.5214 | 0.5761 | 0.4949 | 0.7253 | 0.544 | 0.6503 |
| s4_B1_O2lr0.01_seed42 | tinybert_masked_mean_linear_dual_head | 42 | 63 | 0.669 | 0.4968 | 0.5767 | 0.5017 | 0.7063 | 0.5052 | 0.6553 |
| s4_B3_O2lr0.01_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.6697 | 0.5113 | 0.5774 | 0.4904 | 0.722 | 0.5281 | 0.6556 |
| s6_B3_loss_seed43 | bigru_content_gate_time_pool | 43 | 63 | 0.6711 | 0.5214 | 0.5585 | 0.4868 | 0.7301 | 0.5393 | 0.65 |
| s4_B1_O3rho0.02_seed42 | tinybert_masked_mean_linear_dual_head | 42 | 63 | 0.6725 | 0.4925 | 0.5759 | 0.4931 | 0.7078 | 0.4965 | 0.6609 |
| s3_M2_seed42 | bigru_domain_safe_gap_structure_gate | 42 | 63 | 0.6729 | 0.5053 | 0.5665 | 0.4751 | 0.7188 | 0.5485 | 0.6554 |
| s4_B1_O3rho0.05_seed42 | tinybert_masked_mean_linear_dual_head | 42 | 63 | 0.673 | 0.4826 | 0.5733 | 0.4915 | 0.7088 | 0.4822 | 0.6613 |
| p0_M1_CEMSE_seed42 | bigru_binary_mask_gate_time_pool | 42 | 63 | 0.6751 | 0.51 | 0.5691 | 0.4823 | 0.7215 | 0.5434 | 0.6601 |
| s6_B3_optimizer_seed43 | bigru_content_gate_time_pool | 43 | 63 | 0.6754 | 0.4946 | 0.5616 | 0.4844 | 0.732 | 0.515 | 0.6574 |
| s3_M3_seed42 | M2_plus_local_cross_time_attention | 42 | 63 | 0.6768 | 0.4853 | 0.5497 | 0.4752 | 0.7416 | 0.5087 | 0.6555 |
| s4_B3_O2lr0.03_seed42 | bigru_content_gate_time_pool | 42 | 63 | 0.678 | 0.4965 | 0.5756 | 0.4816 | 0.7267 | 0.5334 | 0.6616 |
| s4_B1_O2lr0.03_seed42 | tinybert_masked_mean_linear_dual_head | 42 | 63 | 0.6788 | 0.46 | 0.5681 | 0.4825 | 0.7146 | 0.4623 | 0.6681 |
| p0_B0_seed42 | training_prior | 42 | 63 | 0.7808 | 0.2114 | 0.4643 |  | 0.7808 | 0.2114 | 0.7808 |
| smoke_B2_seed42 | three_modality_masked_mean_concat_mlp | 42 | 63 | 1.0118 | 0.4077 | 0.4221 | 0.1293 | 1.0167 | 0.4269 | 1.0143 |
| smoke_M1_seed42 | bigru_binary_mask_gate_time_pool | 42 | 63 | 1.0218 | 0.1829 | 0.375 | 0.0748 | 1.0268 | 0.1818 | 1.0218 |

## 5. 机制消融（逐项单变量对照）

| 机制 | 对照 | 变体 | ΔR_MAE | ΔR_F1 | ΔcleanF1 | ΔcleanMAE | 状态 |
|---|---|---|---|---|---|---|---|
| missing_view | p0_M1_seed42 | p0_M1_cleanonly_seed42 | 0.001 | 0.0134 | 0.0215 | -0.0021 | completed |
| ce_mse_control | p0_M1_seed42 | p0_M1_CEMSE_seed42 | 0.01 | -0.0087 | -0.004 | 0.0131 | completed |
| mask_side_channel | p0_M1_seed42 | s3_B3_seed42 | -0.0004 | -0.0159 | -0.0088 | -0.0028 | completed |
| gap_structure | s3_B3_seed42 | s3_M2_seed42 | 0.0082 | 0.0025 | 0.0099 | 0.0112 | completed |
| cross_time_attention | s3_M2_seed42 | s3_M3_seed42 | 0.0039 | -0.02 | -0.0398 | 0.0001 | completed |
| mean_pool_vs_gru | p0_B2_seed42 | p0_M1_seed42 | 0.0078 | 0.0037 | 0.0295 | 0.0035 | completed |

## 6. 冻结后验证（描述性，不用于改模型）

- **confirm**：728 样本 × 105 条件；R_MAE 0.6575、R_F1 0.5443、最差条件 MAE 0.7224；配对 bootstrap 对比状态 completed
- **position**：728 样本 × 27 条件；R_MAE 0.6505、R_F1 0.5537、最差条件 MAE 0.7145；配对 bootstrap 对比状态 completed
- **coupling**：728 样本 × 60 条件；R_MAE 0.6618、R_F1 0.5402、最差条件 MAE 0.7195；配对 bootstrap 对比状态 completed
- **test**：727 样本 × 63 条件；R_MAE 0.7767、R_F1 0.5136、最差条件 MAE 0.8118；配对 bootstrap 对比状态 completed

## 7. 附件3 提交与打包

- 行数 30、ID 顺序正确 True、CSV/JSON 一致 True、符号规则 True
- 干净进程离线重载一致：True
- 图：17 个 SVG（confirm_f1_by_ratio.svg, confirm_mae_by_ratio.svg, confirm_type_ratio_heatmap.svg, confirm_worst_confusion_matrix.svg, position_f1_by_ratio.svg, position_mae_by_ratio.svg…）

## 8. 未执行项与限制

- 机制消融未执行：无
- 预算裁剪（not_run_budget）：无
- 训练失败记录：0 条
- P2 项目（M4/M5/O5、GradNorm、蒸馏、专家后融合）按协议仅在预算允许时开启；本次是否执行见 `execution_plan.json` 与 `failures.jsonl`。
- 旧官方 test 在本项目历史工作中已曝光，本报告中的 test 结果只是描述性审计，不构成首次盲测，也不用于选择模型。
- bootstrap 只度量固定模型、固定数据协议下按 video_id 抽样的不确定性，不覆盖全部训练数据重采样误差。

## 9. 实现事件记录

本次运行在正式训练前修复了以下实现缺陷（原始报错日志保存在 `logs/incidents/`）：

- `np.savez(path, **payload)` 与附件3载荷中的 `file` 键冲突 → 改为文件句柄并重命名该键为 `source_file`
- `RunData` 在 `normalizer` 赋值前调用 `_split` → 调整加载顺序
- `build_model` 把 `hidden` 传给不接受该参数的均值基线 → 按构造函数签名过滤 kwargs
- 训练循环在诊断梯度后重复 `loss.backward()` → 诊断移到主 backward 之前
- `TrainingPriorModel` 缺少 `clear_text_cache` → 调用处加 hasattr 守卫

以上修复均在冒烟阶段验证后重跑，未修改协议、数据划分或评价口径。
