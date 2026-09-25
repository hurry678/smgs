# 问题二产出审核说明（优化/q2yh）

本目录是问题二“模态信息缺失条件下的情感预测建模与验证”的完整归档，供外部审核使用。
归档时对应的远端仓库提交为 `0f0aa886ddd1c868879279320560e1f2504887b6`（`origin/main`，提交信息“Add Q2 server implementation plan and validated masking toolkit”）。
服务器工作目录为 `/data2/hy/cts/e_problem/q2yh`，本次运行目录为 `runs/q2run_20260925_srv`。
本地副本与本目录已逐文件比对 SHA-256（316 个文件，0 不匹配）。

## 1. 建议审核顺序

1. `runs/q2run_20260925_srv/report.md`：**本轮审核重点**，主运行报告；全部数值由落盘 JSON/CSV 读取生成，未手工填写。
2. `runs/q2run_20260925_srv/selection.json`、`freeze_manifest.json`：候选筛选、选择规则与冻结记录。
3. `runs/q2run_20260925_srv/metrics.csv`：全部 run × condition 的逐条件指标。
4. `runs/q2run_20260925_srv/evaluation/{confirm,position,coupling,test}/`：冻结后四库审计，各含 `summary.json`（逐条件指标）、`comparison.json`（按 video_id 聚类的配对 bootstrap）、`metrics.csv`、`predictions_*.npz`（逐样本预测）与 `figures/`。
5. `runs/q2run_20260925_srv/ablations/mechanism_ablations.json`：6 项机制消融的逐项单变量对照。
6. `runs/q2run_20260925_srv/submission/`：附件3 提交物（`q2_predictions.csv`、`.json`、`validation.json`、`q2_package.zip`）。
7. `runs/q2run_20260925_srv/package/`：冻结推理包（`infer.py`、`model/`、`data/`、`resources/`、`README.md`）。
8. `runs/q2run_20260925_srv/failures.jsonl` 与 `logs/`：失败、失效重跑与修复审计。
9. `scripts/`、`src/`、`configs/protocol.json`、`docs/`：实现代码与预注册协议（`protocol.json` 在训练前冻结）。
10. `PUSH_MANIFEST_SHA256.txt`：本目录完整性校验清单。

## 2. 执行环境与协议口径

- 环境（`runs/q2run_20260925_srv/environment.json`、`environment.lock.txt`）：Python 3.10.20、torch 2.5.1+cu121、numpy 1.26.4、NVIDIA RTX 4090、`--device cuda:1`；AMP 关闭、TF32 关闭、threads 8；未安装新依赖。
- 文本编码器：`google/bert_uncased_L-2_H-128_A-2@30b0a37ccaaa32f332884b96992754e246e48c5f`，冻结、恒 eval、先遮挡再编码；音频 50×74、视觉 50×35、文本 3×50。
- 协议 SHA256 `8cc7c0444993df4f1e7528afef98c5dabc457eda34ded7a3587087346bb10973`；掩码库 select 63 / confirm 105 / position 27 / coupling 60 / clean 1 个条件。
- 模态索引：0=text、1=audio、2=vision（`src/data.py` 的 `MODALITIES`）；缺失定义为 `D = 0 < t < sep`（不含 CLS/SEP），padding 与 UNK 都不算缺失。
- 归一化仅用 train 中 O=1 的行拟合（float64、std 下界 1e-5），valid/test/附件3 不重新 fit；附件3/4 不参与训练、调参与选型。
- 预算与阶段（`execution_plan.json`，各阶段 run_ids 在首次参数更新前登记）：smoke 2 + p0 5 + screen 4 + optimize 9 + losses 5 + multiseed 6 = 31 个神经训练 run，上限 36。

## 3. 部署模型与核心结果

- 部署模型：`bigru_content_gate_time_pool`，run `s5_B3_L2_seed42`（seed 42）；`selection_status = selected_three_seed`，三 seed 成员为 `s5_B3_L2_seed42`、`s6_B3_loss_seed43`、`s6_B3_loss_seed44`，部署取 seed 42 单模型。
- select 库（63 条件平均 + clean 条件）：R_MAE **0.6566**、R_F1 **0.5442**、R_Accuracy 0.5702、R_Pearson 0.4996、最差条件 MAE 0.7192、clean F1 0.5743、clean MAE 0.6359。
- 对照 B2 简单基线（三 seed 平均 0.6617 / 0.5174）：ΔR_MAE −0.0007、ΔR_F1 +0.0292、Δclean F1 +0.0564、Δclean MAE −0.0076（负值表示优于基线）。
- 冻结后审计（描述性，不用于改模型）：

| 库 | 样本 × 条件 | R_MAE | R_F1 | 最差条件 MAE |
|---|---:|---:|---:|---:|
| confirm | 728 × 105 | 0.6575 | 0.5443 | 0.7224 |
| position | 728 × 27 | 0.6505 | 0.5537 | 0.7145 |
| coupling | 728 × 60 | 0.6618 | 0.5402 | 0.7195 |
| test | 727 × 63 | 0.7767 | 0.5136 | 0.8118 |

- 缺失规律（对 `evaluation/*/summary.json` 的 `per_condition` 按属性分组平均）：
  - **缺失比例**：0.10 → 0.50 单调变差。confirm MAE 0.6414 → 0.6577 → 0.6733、F1 0.5611 → 0.5440 → 0.5277；position MAE 0.6409 → 0.6493 → 0.6612；coupling MAE 0.6416 → 0.6606 → 0.6831；test MAE 0.7700 → 0.7764 → 0.7839。
  - **缺失模态类型**（confirm）：缺文本（type=0）最差，MAE 0.6818 / F1 0.5198；缺音频（type=1）最好，0.6297 / 0.5754；缺视觉（type=2）居中，0.6377 / 0.5639；三模态同缺（012）0.6666 / 0.5346。
  - **缺失位置**（position）：start/middle/end 差异很小，MAE 0.6506 / 0.6508 / 0.6500，F1 0.5515 / 0.5518 / 0.5579；位置效应弱于比例与类型效应。
  - **多模态同步缺失**（coupling）：仍以比例为主导，MAE 0.6416 → 0.6831；类型上 type=02 最差（0.6846 / 0.5167）、type=12 最好（0.6306 / 0.5688）。
- 配对 bootstrap（按 video_id 聚类、10,000 次重采样，`evaluation/*/comparison.json`）：confirm 中 frozen 相对 B2 的 macro-F1 差 +0.0298，95% 区间 [0.0113, 0.0499]（不跨 0）；MAE 差 −0.0003，区间 [−0.0115, 0.0113]（跨 0）。
- 机制消融 6 项全部 `completed`（见 `report.md` §5 与 `ablations/mechanism_ablations.json`）：missing_view、ce_mse_control、mask_side_channel、gap_structure、cross_time_attention、mean_pool_vs_gru。
- 附件3：30 行，`ids_in_order`、`filenames_match`、`csv_json_consistent`、`sign_rule_ok`、`finite`、`intensity_in_range` 全部 true；`q2_predictions.csv` SHA256 `6c7aa69d4b1b549c41fec06be43a99d6481dee21298b33bc6d3b34f39bf333f0`；`q2_package.zip` 17,222,868 字节；CUDA 干净进程重载为 bit-exact（logits/raw max abs diff 0.0）。

## 4. 审核时请特别注意（口径边界）

1. **test 库不是盲测**：727×63 的 test 结果在本项目历史工作中已曝光，仅作描述性审计，未用于选型；论文不得把它当作首次盲测成绩。
2. **MAE 与基线同量级**：部署模型相对 B2 的 ΔR_MAE 仅 −0.0007，按预注册容差属同量级，不得宣称 MAE 显著更优；F1 的增益有配对 bootstrap 支持（95% 区间不跨 0），但仅覆盖 select/confirm 库与单种子部署模型。
3. **掩码侧信道收益有限**：`mask_side_channel` 消融 ΔR_MAE −0.0004 而 ΔR_F1 −0.0159，方向不一致，需在论文中如实讨论，不能写成单向提升。
4. **缺失视图训练是双向取舍**：`missing_view` 消融显示 clean-only 训练在 clean 条件更好、缺失条件更差（Δclean F1 +0.0215、ΔR_F1 +0.0134），这是“缺失视图训练有效”的证据，但不宜表述为全面更优。
5. 附件3/4 无标签样本未参与训练、调参或模型选择；条件或种子不得当作独立样本（重采样单位是 video_id）。
6. `package/package.json` 中 `q1_zip`、`combined_q1_q2_bytes`、`combined_within_limit` 为 null：Q1+Q2 合包 ≤50,000,000 字节的约束在本轮未入账，需在合包阶段补算，不得直接引用为“已满足”。
7. **CPU 与 CUDA 非字节一致**：提交 CSV 由 CUDA 产出；CPU 复现行数与极性一致，但 float32 logits 漂移最高约 5e-4，见 `package/README.md` 的设备说明。
8. 失败与失效重跑已如实记录：`s5_B3_EMA_seed42` 首次运行因引擎“计算了 EMA 权重但验证/部署用原始权重”被判无效并归档重跑（`runs/_invalidated/`）；其余实现缺陷与归档说明见 `report.md` §9 与 `failures.jsonl`（7 条事件）。
9. bootstrap 只度量固定模型、固定协议下按 video_id 重采样的不确定性，不覆盖训练数据重采样误差。
10. 本轮未安装新依赖、未修改协议/数据划分/评价口径；未执行项一律写 `not_run`。

## 5. 完整性校验

- `PUSH_MANIFEST_SHA256.txt` 记录本目录除自身外全部 316 个文件的 SHA-256，格式兼容 `sha256sum -c`。
- 运行目录内保留服务器原始 `runs/q2run_20260925_srv/MANIFEST_SHA256.txt`（416 条），推送归档未重写它。
- 本次归档覆盖其中 267 条（校验结果：0 不匹配）；未随仓库归档的 149 条出于体积考虑，可在服务器 `/data2/hy/cts/e_problem/q2yh/runs/q2run_20260925_srv/` 原地复核：
  - `data/` 5 个文件（约 109 MB）、`masks/` 9 个（约 132 MB）——原始/派生张量与掩码库，可由 `scripts/prepare_data.py`、`scripts/build_mask_bank.py` 重建；
  - `evaluation/*/predictions_*.csv` 8 个（约 79 MB）——与已入库的 `predictions_*.npz` 为同一内容的文本展开；
  - `predictions/*_select.npz` 31 个（约 22 MB）——与已入库的 `runs/<run_id>/predictions_select.npz` 重复；
  - `logs/incidents/` 中间包 96 个（约 173 MB）——失败尝试的归档副本，`failures.jsonl` 已逐条记录其原因。
- 服务器另存完整同步包 `/tmp/q2run_20260925_srv_sync.tar.gz`（69,372,813 字节，SHA256 `b25be3c7c5393e3cfe07c2d666dbc57f9f138e9981b6076c8a2a06ff4425897c`），需要全量中间件时从该包取用。
- 服务器复现入口：`README.md`、`SERVER_PROMPT.md` 与 `runs/q2run_20260925_srv/logs/chain.log`（各阶段命令与退出码）。