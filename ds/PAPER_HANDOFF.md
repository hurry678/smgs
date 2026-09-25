# 论文写作与图表交接说明

## 1. 论文主线建议

本文可统一采用以下四层结构，避免把三问写成彼此割裂的实验：

1. **数据与统一表示层**：问题一完成 100 条样本质量审计、三模态特征提取、50 个位置的统一时序组织与填充记录。
2. **缺失鲁棒预测层**：问题二在统一表示上加入模态有效掩码，以双任务预测器输出极性、强度，并在局部连续缺失下评价。
3. **解释与核验层**：问题三以 Shapley 值量化模态作用，以连续窗口遮挡定位局部证据，并通过 LOO、随机连续遮挡和点散遮挡对照验证。
4. **工程与复现层**：冻结模型、参数和输出，区分验证集与附件推理，保证附件 3、附件 4 不参与调参。

## 2. 问题一可写入论文的结论

### 2.1 数据清洗

- 原始附件 1 共审计 `100` 条样本，合格 `100` 条，排除 `0` 条。
- 清洗规则覆盖文本非空、数值标签与类别一致、视频文件存在且非空、音视频流存在、时长有效和完整解码。
- 原始数据只审计，不复制、不修改、不删除。

权威文件：

- `artifacts/q1/data_cleaning/quality_report.json`
- `artifacts/q1/data_cleaning/all_samples_quality.csv`
- `artifacts/q1/data_cleaning/qualified_samples.csv`
- `artifacts/q1/data_cleaning/excluded_samples.csv`

### 2.2 最终时序组织与特征

正式特征包位于 `artifacts/q1/server_final_100/final_100/`，每条样本包含：

- `features/<sample_id>.npz`：对齐后的三模态序列特征；
- `metadata/<sample_id>.json`：长度、有效位置、填充、模态来源与映射记录；
- `q1_alignment.jsonl`：逐样本对齐记录；
- `q1_manifest.csv`：样本编号、模态文件与特征文件的一一对应表；
- `q1_processing_log.jsonl`：处理日志；
- `feature_dictionary.md`：特征定义；
- `quality_audit.json`：覆盖完整性与结构核验。

### 2.3 对比实验最终选型

以下以 `artifacts/q1/comparison/server_all_100_repeat10/selection_decision.md` 为准：

| 环节 | 最终选择 | 论文中的核心依据 |
|---|---|---|
| 文本 | BERT | 比 Hash 基线的 Macro-F1 高约 0.0627，Pearson 高约 0.2498，10/10 次胜出 |
| 语音 | 40 维完整特征 | F1 最高且保留 MFCC、差分和频谱信息；轻量方案没有显著全面优势 |
| 视觉归档 | 90 维完整特征 | 保证正式 NPZ 保留完整特征超集 |
| 下游视觉视图 | `no_hog_54` | F1 不降，Pearson 提升约 0.0548 |
| 视觉采样率 | 5 fps | 10 fps 没有显著提升，存储和计算成本约翻倍 |
| 时序对齐 | `full_dp_20` | 多重校正后无显著最优候选，但其停顿结构和时长适配更均衡 |
| 平滑 | `sigma=0.75` | `sigma=1.0` 的 F1 增益不显著，Pearson 下降 |

注意：对比实验中的线性探针是“方法选型探针”，不是问题一正式情感模型，也不能当作问题二预测结果。

## 3. 问题二可写入论文的结论

### 3.1 正式版本

- 版本：`M3-ensemble9 + C2`；
- 集成规模：9 个随机种子；
- 融合规则：softmax 概率均值、强度原始值均值；
- 发布策略：C2；
- 冻结协议：仅用验证集选择，官方测试只运行一次，附件 3 未参与调参。

权威文件：

- `artifacts/q2/v3_m3_ensemble9/version_manifest.json`
- `artifacts/q2/v3_m3_ensemble9/valid_valid_evaluation.json`
- `artifacts/q2/v3_m3_ensemble9/test/test_test_evaluation.json`
- `artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json`

### 3.2 核心指标

| 数据划分 | Accuracy | Macro-F1 | MAE | Pearson | n |
|---|---:|---:|---:|---:|---:|
| 验证集 | 0.620879 | 0.599707 | 0.579967 | 0.663537 | 728 |
| 官方 test | 0.635488 | 0.596337 | 0.643349 | 0.657517 | 727 |

附件 3 预测分布：Negative `6`、Neutral `9`、Positive `15`，共 `30` 条。

### 3.3 缺失鲁棒性写法

建议把“模态完全不存在”与“局部连续时段不可用”分开：

- 当前实验的 `missing ratio` 用于构造模态局部连续缺失掩码；
- `subset` 表示哪些模态保留可用；
- 结果应报告缺失类型、缺失比例、位置和固定随机重复下的变化，而不是写成整模态永久删除；
- 主结论应以 `R_MAE` 和 `Macro-F1` 同时评价，避免只报告分类指标。

可引用：

- `artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json`
- `artifacts/q2/robustness_eval_fixed5/`
- `paper/tables/q2_missing_robustness.csv`（由脚本生成）

## 4. 问题三可写入论文的结论

### 4.1 解释方法

问题三使用三种互补证据：

1. **模态级 Shapley 值**：精确枚举 8 个模态子集，量化文本、语音、视觉对类别和强度的作用；
2. **局部连续窗口遮挡**：在统一序列上以窗口为单位删除/替换信息，定位与预测相关的局部片段；
3. **对照与核验**：以 LOO、随机连续遮挡和点散遮挡作为对照，检查解释的忠实性和稳定性。

### 4.2 验证集结果

- 选择样本数：`128`；
- 类别与强度 Shapley 最大加和误差：`1.1921e-7`；
- Shapley 与 LOO 主模态一致率：`0.8203125`；
- 主模态计数：text `64`、audio `9`、vision `55`；
- 平均类别绝对份额：text `0.3659`、audio `0.2267`、vision `0.4074`；
- 平均强度绝对份额：text `0.3930`、audio `0.2503`、vision `0.3567`。

权威文件：`artifacts/q3/final_validation/q3_validation_report.json`。

### 4.3 附件 4 结果

- 样本数：`20`；
- 极性分布：Negative `7`、Neutral `3`、Positive `10`；
- 主模态分布：text `12`、audio `5`、vision `3`；
- 冻结核验错误：空列表；
- 附件 4 未参与调参；
- 文本证据为 tokenizer 精确字符偏移；音视频为比例近似映射。

权威文件：

- `submission/q3_predictions_explanations.csv`
- `submission/q3_evidence.jsonl`
- `submission/q3_validation_report.json`
- `submission/q3_freeze_manifest.json`

### 4.4 解释表述边界

- 可以说“模型判断主要依据某模态/某窗口”，不要说“某模态导致人类情感”；
- 可以说“连续窗口遮挡比对照更集中/更稳定”，不要说遮挡等于物理因果；
- 音视频证据只能写成“按视频时长比例映射得到的近似时段/帧区间”；
- `primary_modality`、Shapley 份额、遮挡证据等级是不同概念，必须在表中分列。

## 5. 推荐论文图表

运行 `python paper/generate_paper_figures.py` 后可直接使用：

| 图/表 | 建议位置 | 主要结论 |
|---|---|---|
| `q1_data_cleaning.svg` | 问题一数据预处理 | 100 条全部合格 |
| `q1_alignment_methods.svg` | 问题一模型对比 | 对齐方法在时长适配和探针指标间的权衡 |
| `q1_modality_components.svg` | 问题一特征对比 | BERT、语音 40 维、视觉候选的差异 |
| `q2_model_comparison.svg` | 问题二模型选择 | M0/B2-B5 的 R_MAE 与 Macro-F1 |
| `q2_missing_robustness.svg` | 问题二缺失分析 | 不同缺失模态组合的 MAE 增量 |
| `q2_seed_stability.svg` | 问题二稳定性 | M0 与 B4 的三随机种子波动 |
| `q2_prediction_distribution.svg` | 附件 3 推理 | 预测极性分布 |
| `q3_shapley_modalities.svg` | 问题三模态作用 | 类别/强度绝对份额 |
| `q3_sensitivity.svg` | 问题三窗口敏感性 | w3/w5/w10 下遮挡集中度 |
| `q3_attachment4_distribution.svg` | 附件 4 推理 | 极性与主模态分布 |
| `q3_evidence_levels.svg` | 问题三证据核验 | 高/中/低证据等级统计 |

## 6. 可复现实验说明模板

论文附录可按以下顺序写：

1. 说明原始附件版本、样本数和文件路径；
2. 给出清洗规则、工具版本和完整解码检查；
3. 说明三模态特征维度、时间尺度、50 位置对齐和填充规则；
4. 列出问题一 23 组对照及最终选择；
5. 说明问题二掩码、双任务损失、验证集冻结和集成规则；
6. 列出 Q2 正式版本 SHA-256 与测试只评价一次协议；
7. 说明问题三 Shapley 子集、窗口遮挡、控制组和配对统计；
8. 说明附件 3、附件 4 无标签，只做推理，不报告监督指标；
9. 引用 `MANIFEST_SHA256.txt` 作为交付包完整性校验。

## 7. 必须避免的表述

- 不能把附件 3、附件 4 的预测结果写成真实标签准确率；
- 不能把验证集的统计显著性写成全体真实分布的普遍显著性；
- 不能把音视频近似映射写成精确 token 时间戳或人工关键帧；
- 不能把视觉线性探针的弱结果写成“视觉模态无用”；
- 不能把解释模型中的 Shapley 值写成自然语言意义上的因果效应；
- 不能混用 Q1 对比探针指标和 Q2 正式预测指标。
