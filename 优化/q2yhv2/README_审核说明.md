# 问题二产出审核说明（优化/q2yhv2）

本目录是问题二「DS 高算力优化 v2」方案的完整归档，供队友外部审核使用。
方案来源：`问题二/q2_ds_optimization_v2_20260925`（[GitHub](https://github.com/hurry678/smgs/tree/main/%E9%97%AE%E9%A2%98%E4%BA%8C/q2_ds_optimization_v2_20260925)）。
服务器：`myserver`（hy@192.168.182.155，4×RTX 4090），工作目录 `/data2/hy/q2_runs/q2_ds_hc_v2_20260925`。
运行日期：2026-09-25（CST），主流程 17:01:41 → 17:39:10，冻结后发布 17:42:52 → 17:43:01。
本目录已按 `PUSH_MANIFEST_SHA256.txt` 逐文件记录 SHA-256（31 个文件，1,888,923 字节）。

## 0. 结论速览

**新候选集成未通过 confirmation 门控，协议自动回退保留旧 M3 集成（ds_seed42–50 等权 1/9）。**
唯一未过项为 R_F1 降幅 `0.004064 > 0.003`；其余 4 项门控均通过（R_MAE 改善 +0.000589 ≥ 0.0005）。
这是方案预先写死的判定规则，属于「选模过拟合被正确拦截」，不是执行失败。

| confirmation 分区（n=319） | R_MAE | R_F1 | worst MAE | clean MAE | clean F1 |
|---|---:|---:|---:|---:|---:|
| 基线（旧 M3，保留） | 0.637222 | **0.578090** | **0.687386** | 0.610290 | **0.612500** |
| 新候选（8 成员加权） | **0.636634** | 0.574027 | 0.689105 | **0.610404** | 0.611258 |

对比：selector 分区（n=409，只用于初筛）候选全面占优（R_MAE 0.572627 vs 0.574584，R_F1 0.578242 vs 0.569724），但该优势未泛化到 confirmation 分区。

## 1. 建议审核顺序

1. `server_results_20260925/审核说明_问题二_DS优化v2_20260925.md`：**本轮审核重点**，中文审核报告（门控明细、三分区对照、合规声明、失败记录、论文可用结论）。
2. `server_results_20260925/confirmation.json`：confirmation 门控原始证据，含 `deltas_vs_baseline`、`gates`、`prohibitions`、逐条件 cells。
3. `server_results_20260925/selection.json`：受约束集成选模全过程（57 个候选排序、5 轮轨迹、权重上限约束、selector 门控）。
4. `server_results_20260925/arm_selection.json`：12 臂 × 3 折搜索，选出 A03_ema / A10_sqrt_class / A08_huber05 / A00_ds_exact 四臂。
5. `server_results_20260925/frozen/freeze_manifest.json`：冻结清单（成员源 SHA-256、紧凑 SHA-256、体积、共享编码器、输入 SHA-256、代码 SHA-256）。
6. `server_results_20260925/publication/`：冻结后发布物（`test_evaluation.json`、`test_predictions.npz`、`attachment3_predictions.csv/json`、`publication_summary.json`）。
7. `server_results_20260925/baseline_replay/verification.json`：基线复现一致性证明（6 项指标差值全 0）。
8. `server_results_20260925/logs/`：`all_20260925_170012.log`（首次失败：缺 `q2_eval_robustness.py`）、`all_20260925_170141.log`（成功）、`publish_20260925_174252.log`。
9. `scripts/`、`configs/protocol.json`、`docs/优化方案与验收标准.md`、`tests/test_protocol.py`：实现代码与预注册协议（协议 SHA `e50eace6662a14b632b09057f71905357e09e0d805df3fba01d27a357a232b48`）。
10. `PUSH_MANIFEST_SHA256.txt`：本目录完整性校验清单。

## 2. 关键数值

| 项目 | 值 |
|---|---|
| preflight | `passed`；单元测试 4/4 通过；旧检查点 SHA 9/9 一致 |
| 训练任务 | 84 个（36 CV + 48 全量），合计 **1.392 GPU-hours**，峰值显存 1.306 GiB |
| 冻结版本 | `M3-ensemble9-retained`，紧凑权重 12,709,084 字节，预计合并提交 41,724,828 字节（上限 50,000,000） |
| 官方测试集（n=727，冻结后描述性） | Accuracy 0.635488 / Macro-F1 0.596337 / MAE 0.643349 / Pearson 0.657517 |
| 附件 3（n=30） | Negative 6 / Neutral 9 / Positive 15；CSV SHA `aa25e119480aabd278d3249e1388522c20528520595e9e6985777e80b77470a4` |
| 全量 valid 描述性（n=728，不参与决策） | 基线 R_MAE 0.601060 / R_F1 0.579060；候选 0.599568 / 0.581642 |

## 3. 合规边界（请重点复核）

- 官方测试集与附件 3 **未**用于训练、早停、选模、定权或阈值；二者在冻结完成并以 `--allow-post-freeze` 显式授权后才读取，且 `test_policy.may_change_frozen_choice = false`。
- confirmation 分区仅允许「接受预选候选或保留基线」，不允许改权重；selector 与 confirmation 分组隔离（`group_separated = true`，索引 SHA 已记录）。
- 全量 valid 在门控之后才读取，仅作描述性对照，不得据此推翻 confirmation 结论。
- 训练未回改 `ds/`、`问题二/` 既有文件与历史目录；全部写入 `/data2/hy/q2_runs/q2_ds_hc_v2_20260925`；未安装任何新依赖（仅补齐一个缺失的既有脚本 `q2_eval_robustness.py`）。
- DS 代码版本锚点（与本地 `E题/scripts` 一致）：`q2_models.py` `659cac7d…367700`、`q2_train.py` `ee5383de…aef8d6`、`q2_eval_robustness.py` `56cbcba6…887d53f`。

## 4. 未归档内容

`search/`、`pool/`、`cache/` 等重型中间目录未回传（体积大且可由 `scripts/run_q2hc.py` 重建）；冻结的紧凑检查点在服务器 `/data2/hy/q2_runs/q2_ds_hc_v2_20260925/frozen/members/` 下，SHA 已全部记录在 `freeze_manifest.json`，如需可单独打包。