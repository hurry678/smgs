# 审核说明：问题二 DS 高算力优化 v2.1（2026-09-25）

- 方案目录：`问题二/q2_ds_optimization_v2_1_20260925/`（提交 `ac6e0fc`）
- 服务器工作目录：`/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925`
- 执行环境：`/data2/hy/anaconda3/envs/secgpt-vllm/bin/python`，numpy 1.26.4 / sklearn 1.7.2 / torch 2.3.0+cu121，CUDA 可用，4×GPU
- 时间线（CST）：`all` 21:22:06 → 21:52:00；`publish` 21:52:28 → 21:52:38；`export-audit` 21:53:21 → 21:53:39。三个阶段 `EXIT_CODE` 均为 `0`。

## 一、结论摘要

| 项目 | 结果 |
|---|---|
| confirmation 是否通过 | **否**（`candidate_rejected_baseline_retained`，`passed=false`） |
| 最终版本 | **旧 M3 回退**：`M3-ensemble9-retained`，9 成员等权 1/9 |
| 唯一未过门控项 | `worst_condition_MAE_increase = +0.0030572 > 0.003`（超出 5.7e-5） |
| 冻结状态 | `baseline_retained_without_new_test`（冻结时不存在本版本 test 结果） |
| 提交体积 | 紧凑检查点 12,709,084 B；投影合并 41,724,828 B ≤ 50,000,000 B |
| 审计包校验 | `sha256sum -c` 全 OK；`verify_audit_bundle.py` → `passed`，3 分卷 / 464 文件 / 222,885,867 B |

本轮为**重复验证修正**，上一轮 valid 结果已公开，`validation_status = previously_exposed_repeat_validation_audit`，**不构成新的盲 confirmation**。

## 二、预检与基线复现

- `resolved_inputs.json.status = passed`，`failures = []`。
- 9 个旧检查点按冻结 SHA 唯一定位成功；5 个 DS 核心脚本 SHA 全部一致，包含 `q2_eval_robustness.py`（`56cbcba6…887d53f`）。
- 数据指纹：`q2_data.npz = 2bad10a2…36c38`，`attachment3_aligned.npz = 5b964459…3f36`，MiniLM `pytorch_model.bin = c3a85f23…af256`。
- 基线 9 模型复现：`baseline_replay/verification.json` 通过（固定 `mask_seed=2026`，5 复本）。

## 三、selector（选模，n=405，`mask_seed=20261002`，3 复本）

| 指标 | 基线 M3 | 入选候选 | 差值（候选−基线） |
|---|---:|---:|---:|
| R_MAE | 0.635829 | 0.625695 | −0.010133（改善） |
| R_F1 | 0.571963 | 0.582149 | +0.010186（改善） |
| clean MAE | 0.613512 | 0.601015 | −0.012497（改善） |
| clean F1 | 0.596356 | 0.616001 | +0.019645（改善） |
| worst_condition_MAE | 0.678950 | 0.672851 | −0.006098（改善） |

`passes_selector_gates = true`；目标函数值 1.900000 → 1.861994。

入选候选权重（5 成员）：`A03_ema_seed109 = 2/9`、`ds_seed42 = 2/9`、`ds_seed43 = 2/9`、`ds_seed45 = 2/9`、`ds_seed50 = 1/9`。

## 四、confirmation（确认，n=323，`mask_seed=20261003`，5 复本）

| 指标 | 基线 M3 | 候选 | 差值（候选−基线） | 门控 | 判定 |
|---|---:|---:|---:|---|---|
| R_MAE 改善 | — | — | +0.0016176 | ≥ 0.0005 | 通过 |
| R_F1 降幅 | — | — | −0.0008694 | ≤ 0.003 | 通过 |
| clean MAE 增量 | — | — | −0.0059044 | ≤ 0.002 | 通过 |
| clean F1 降幅 | — | — | −0.0068817 | ≤ 0.003 | 通过 |
| **worst_condition_MAE 增量** | — | — | **+0.0030572** | ≤ 0.003 | **未通过** |

原始指标对照：

| 分区 | R_MAE | R_F1 | worst MAE | clean MAE | clean F1 |
|---|---:|---:|---:|---:|---:|
| 基线 | 0.550052 | 0.594693 | 0.576508 | 0.537906 | 0.604744 |
| 候选 | 0.548435 | 0.595562 | 0.579566 | 0.532002 | 0.611626 |

候选在整体 R_MAE、R_F1、干净输入 MAE/F1 上均优于基线，**仅最差缺失条件的 MAE 恶化 0.0030572**，以 5.7e-5 的极小幅度越界被硬门控拦截。协议规定硬门控不可放宽，故按预先约定回退基线。

- `confirmation_indices_sha256 = 557183068041fc8e0bad4e9c711660bdc296c44b4b30167e0c6e7f7447ec516c`，`group_separated = true`。
- `frozen_choice = baseline`；confirmation 未参与选模，仅用于接受/拒绝预选候选。

## 五、完整 valid 描述性指标（门控后，不改变冻结结论）

| 模型 | R_MAE | R_F1 | worst MAE | clean MAE | clean F1 |
|---|---:|---:|---:|---:|---:|
| 基线 M3 | 0.598691 | 0.579696 | 0.637334 | 0.579967 | 0.599707 |
| 被拒候选 | 0.593489 | 0.584305 | 0.635518 | 0.570395 | 0.613101 |

描述性区间上候选反而略优，但该区间不参与门控，且属已公开数据的重复验证，不得据此推翻 confirmation 结论。

## 六、冻结后 test 与附件 3（仅描述性）

- test（n=727）：Accuracy **0.635488**，Macro-F1 **0.596337**，MAE **0.643349**，Pearson **0.657517**（与上一轮 M3 一致，符合"保留基线"预期）。
- 附件 3（n=30）：Negative 6 / Neutral 9 / Positive 15；CSV SHA-256 `aa25e119480aabd278d3249e1388522c20528520595e9e6985777e80b77470a4`。
- `selection_unchanged_after_test = true`；`test_policy.role = post_freeze_descriptive_only_previously_exposed`，`may_change_frozen_choice = false`。

## 七、算力与失败数

- 完成训练任务：**84**（36 CV 折级 + 48 全量候选），累计训练 **3,568.8 s ≈ 0.991 GPU·h**，累计 554 个 epoch。
- 峰值显存：**1,401,854,464 B ≈ 1.3056 GiB**（`search/A04_consistency005/fold2`）。
- **失败任务数：0**（`resolved_inputs.failures = []`；84 个 `result.json` 齐全，日志中无 traceback/exception/failed 记录）。

## 八、四臂最终轮数 vs CV 中位最佳轮数（v2.1 修正点）

| 入选臂 | 三折最佳轮数 | 中位数 | 全量最终轮数 | 是否严格一致 |
|---|---|---|---:|---:|---|
| A03_ema | 7, 8, 7 | 7 | 7 | 是 |
| A10_sqrt_class | 4, 5, 3 | 4 | 4 | 是 |
| A08_huber05 | 4, 5, 3 | 4 | 4 | 是 |
| A00_ds_exact | 3, 2, 3 | 3 | 3 | 是 |

v2 中"全量轮数取整/取上界"的偏差已修正为严格中位数（`min_fixed_epochs = 1`）。

## 九、冻结成员清单（旧 M3，9 成员等权 1/9）

| 成员 | 权重 | 源检查点 SHA-256（截断） | 紧凑检查点 SHA-256（截断） |
|---|---:|---|---|
| ds_seed42 | 1/9 | 57aebf73…bd49b7 | 9a6f9f06…b91035 |
| ds_seed43 | 1/9 | 8036d8f3…c915f89 | a46370b9…531fe1 |
| ds_seed44 | 1/9 | 70701bf1…bfa42 | dd3dca23…478b82 |
| ds_seed45 | 1/9 | 5908d207…59cd7 | a2342a33…facae5 |
| ds_seed46 | 1/9 | e1b7c504…693a6 | e03a0d3a…4065ab |
| ds_seed47 | 1/9 | af70760d…1eb73 | bd314aad…d88328 |
| ds_seed48 | 1/9 | 35e1712d…c5145 | 6fd81f0a…7d02b |
| ds_seed49 | 1/9 | 02e1697a…165e3 | 176d9f77…2f9dc |
| ds_seed50 | 1/9 | 85ef9232…0fdb0 | 44b7e569…1bcb89 |

完整 64 位 SHA 见 `freeze_manifest.json` 与审计分卷内 `frozen/`。

## 十、审计包完整性

| 检查项 | 期望 | 实际 | 结果 |
|---|---:|---:|---|
| cv_results | 36 | 36 | 通过 |
| cv_logs | 36 | 36 | 通过 |
| pool_results | 48 | 48 | 通过 |
| pool_checkpoints | 48 | 48 | 通过 |
| pool_logs | 48 | 48 | 通过 |
| select_caches | 57 | 57 | 通过 |
| confirm_caches | 10 | 10 | 通过 |
| frozen_members | 9 | 9 | 通过 |

- **36 个 CV 原始折级结果与日志全部进入审计分卷**（非摘要）。
- 分卷大小：part01 89,210,690 B / part02 32,698,476 B / part03 83,265,805 B，均 < GitHub 100 MB 单文件限制。
- 共享 MiniLM 权重（90,888,945 B，`c3a85f23…af256`）随 part03 存储一次；`portable_frozen_member_paths = true`。
- 本地回传后复算 SHA-256 与服务器 `SHA256SUMS.txt` **逐字节一致**。

## 十一、未反向调参声明

- 冻结完成前未读取官方 test、附件 3、附件 4 或任何外部情感标签；`prohibitions` 明确记录 `official test was not read`、`attachment 3 was not read`。
- test 与附件 3 仅在冻结后用于描述性发布，未改变成员、权重、轮数、阈值或门控。
- 未因算力充足扩展未预注册超参数；未修改 `ds/`、`优化/q2yh/`、`优化/q2yhv2/` 及历史服务器目录。
- 门控失败项如实记录，未放宽阈值、未重跑以"刷过"。

## 十二、复现路径

```bash
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
ENTRY=/data2/hy/smgs/问题二/q2_ds_optimization_v2_1_20260925/scripts/run_q2hc.py
RUN=/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925
LEGACY=/data2/hy/cts/e_problem

$PY "$ENTRY" preflight --legacy-root "$LEGACY" --work-dir "$RUN"
$PY "$ENTRY" all --legacy-root "$LEGACY" --work-dir "$RUN" --devices cuda:0,cuda:1,cuda:2,cuda:3
$PY "$ENTRY" publish --work-dir "$RUN" --device cuda:0 --allow-post-freeze
$PY "$ENTRY" export-audit --work-dir "$RUN" --export-dir "$RUN/audit_export"
$PY .../scripts/verify_audit_bundle.py "$RUN/audit_export"
```

单元测试：`python tests/test_protocol.py` → 6/6 通过。
