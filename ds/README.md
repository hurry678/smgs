# E题问题一至问题三论文交付包（ds）

本目录是“复杂场景下多模态情感识别的数学建模与算法设计”问题一、问题二、问题三的论文交付包。所有结果均来自本地或服务器实验的冻结产物，论文手可在**不安装第三方 Python 依赖**的情况下重建 SVG 图片和 CSV 表格。

## 1. 快速入口

| 用途 | 文件 |
|---|---|
| 论文写作总说明 | `PAPER_HANDOFF.md` |
| 问题一全过程 | `docs/问题一_全过程记录.md` |
| 问题二全过程 | `docs/问题二_全过程记录.md` |
| 问题三全过程 | `docs/问题三_全过程记录.md` |
| 题目原文提取 | `problem/E题_题目提取.txt` |
| Q2 正式提交结果 | `submission/q2_predictions.csv`、`submission/q2_predictions.json` |
| Q3 正式提交结果与解释 | `submission/q3_predictions_explanations.csv`、`submission/q3_evidence.jsonl` |
| Q3 冻结核验 | `submission/q3_validation_report.json`、`submission/q3_freeze_manifest.json` |
| 图片生成脚本 | `paper/generate_paper_figures.py` |
| 文件清单 | `file_inventory.csv` |
| 全包校验清单 | `MANIFEST_SHA256.txt` |

## 2. 目录结构

```text
ds/
├─ artifacts/q1/                 # Q1 清洗、100条最终特征、对齐/特征/平滑对比
├─ artifacts/q2/                 # Q2 候选模型、缺失鲁棒性、正式版本、附件3推理
├─ artifacts/q3/                 # Q3 忠实性验证、窗口敏感性、附件4解释
├─ docs/                         # 问题一至问题三全过程记录
├─ logs/q3/                      # Q3 运行日志
├─ paper/                        # 论文图片、表格与生成脚本
├─ problem/                      # 题目提取文本
├─ q1_comparison_extension/      # 问题一模型/优化算法系统比较扩展
├─ scripts/                      # 核心训练、评价、对齐与解释脚本
├─ submission/                   # 问题二、问题三正式提交文件
├─ ROADMAP.md                    # 原路线图
├─ README.md                     # 本说明
├─ PAPER_HANDOFF.md              # 论文写作交接
├─ file_inventory.csv            # 文件级清单
└─ MANIFEST_SHA256.txt           # SHA-256 校验清单
```

## 3. 权威版本

### 问题一

- 数据清洗：`artifacts/q1/data_cleaning/quality_report.json`
- 100 条最终特征包：`artifacts/q1/server_final_100/final_100/`
- 对齐、特征与平滑系统比较：`artifacts/q1/comparison/server_all_100_repeat10/`
- 推荐决策：`artifacts/q1/comparison/server_all_100_repeat10/selection_decision.md`
- 扩展对比：`q1_comparison_extension/`

### 问题二

- 正式版本：`artifacts/q2/v3_m3_ensemble9/version_manifest.json`
- 验证集结果：`artifacts/q2/v3_m3_ensemble9/valid_valid_evaluation.json`
- 官方测试结果：`artifacts/q2/v3_m3_ensemble9/test/test_test_evaluation.json`
- 缺失鲁棒性：`artifacts/q2/robustness_analysis_fixed5/q2_robustness_analysis.json`
- 附件 3 提交：`submission/q2_predictions.csv`

正式模型为 `M3-ensemble9 + C2`。附件 3 与附件 4 均无标签，未参与调参或反向选模。

### 问题三

- 验证集忠实性报告：`artifacts/q3/final_validation/q3_validation_report.json`
- 窗口敏感性：`artifacts/q3/sensitivity_w3/`、`sensitivity_w5/`、`sensitivity_w10/`
- 附件 4 推理与解释：`artifacts/q3/final_attachment4/`
- 正式提交：`submission/q3_predictions_explanations.csv`、`submission/q3_evidence.jsonl`
- 冻结核验：`submission/q3_validation_report.json`、`submission/q3_freeze_manifest.json`

## 4. 重建论文图表

在仓库任意目录执行：

```powershell
python C:\work\数模\smgs_repo\ds\paper\generate_paper_figures.py
```

脚本仅使用 Python 标准库，默认输出：

```text
ds/paper/figures/*.svg
ds/paper/tables/*.csv
ds/paper/FIGURE_CATALOG.md
```

如果只做 LaTeX/Word 排版，可直接引用 SVG；若目标模板只接受 PNG，可在 Word 或矢量软件中导出 PNG，不需要安装 Python 包。

## 5. 未纳入包中的内容

以下内容因体积、环境属性或版权原因未复制，但不影响论文手使用冻结结果和脚本：

- 原始 `E题数据/` 与 `E题数据.zip`：约 3.8 GB，属于原始附件，不应重复上传；
- 服务器虚拟环境与 Python 包；
- 服务器模型权重和训练检查点：通过冻结 SHA-256 清单追踪；
- `data_work/q2/q2_data.npz` 等可重建中间数据；
- 与本 E 题无关的旧 `paper/` 内容和 OHT 图片；
- `config.txt`、账号、密码、SSH 或服务器凭据。

## 6. 解释边界

- 音视频证据的时间位置是基于已测视频时长与序列位置的**近似映射**，不得写成原始 token 级精确时间戳或人工标注关键帧。
- 模型解释是模型内部归因与忠实性验证结果，不等于物理因果结论。
- Q3 的主模态、Shapley 归因、遮挡证据和证据等级必须分别表述，不能混为同一种“重要性”。
- 附件 3、附件 4 无标签，不能据此报告准确率、F1、MAE 或 Pearson。
