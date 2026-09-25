# 问题二：基于 DS M0 的高算力优化扩展 v2

本目录只新增文件，不修改 `ds/`、`问题二/` 既有文件或 `优化/q2yh/`。目标是在严格保留
`ds` 的 M0 连续缺失结构基础上，利用多 GPU 做训练稳定化搜索和受约束集成，并以
`M3-ensemble9 + C2` 为不可退化基线。

## 为什么沿用 DS

已知同口径结果如下：

| 方案 | valid Accuracy | valid Macro-F1 | valid MAE | test Accuracy | test Macro-F1 | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| DS M3-ensemble9 + C2 | 0.620879 | 0.599707 | 0.579967 | 0.635488 | 0.596337 | 0.643349 |
| q2yh B3 + SAM + L2 seed42 | 0.594780 | 0.574336 | 0.635869 | 0.568088 | 0.521843 | 0.768138 |

因此本扩展不再以 q2yh 网络为主模型，也不重复已失败的均匀缺失采样和额外跨时注意力。

## 优化内容

1. 先按 SHA-256 找回服务器上的 9 个 DS 检查点，并用固定掩码完整复现基线。
2. 仅在原 `train` 内做 3 折按视频分组交叉验证，筛选 12 个 M0 训练变体。
3. 对前 4 个变体各训练 12 个随机种子，共 48 个新 M0 候选。
4. 在原 `valid` 的 selector 视频组上做受约束坐标贪心集成。
5. 在未参与选择的 confirmation 视频组和独立掩码种子上只确认一次。
6. 确认失败则自动保留旧 M3；确认通过才冻结新候选。
7. 冻结前不读取 `test`；`test` 和附件 3 只能通过显式开关做冻结后推理。

搜索维度只涉及训练稳定性，不改变 M0 主结构：

- 常数或 warmup-cosine 学习率；
- EMA；
- full/missing 一致性约束；
- 回归损失权重和 Huber 参数；
- DS 原缺失分布、文本重点分布及均匀分布；
- 逆频率或平方根逆频率类别权重；
- dropout。

## 运行

服务器完整运行：

```bash
cd /path/to/smgs
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
$PY 问题二/q2_ds_optimization_v2_20260925/scripts/run_q2hc.py all \
  --legacy-root /data2/hy/cts/e_problem \
  --work-dir /data2/hy/q2_runs/q2_ds_hc_v2_20260925 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

`all` 在生成 `frozen/freeze_manifest.json` 后停止。审查冻结清单后，才允许：

```bash
$PY 问题二/q2_ds_optimization_v2_20260925/scripts/run_q2hc.py publish \
  --work-dir /data2/hy/q2_runs/q2_ds_hc_v2_20260925 \
  --device cuda:0 \
  --allow-post-freeze
```

命令可重复执行；已有完整阶段会跳过。加 `--force` 才会覆盖该工作目录内的对应结果。

## 关键输出

| 文件 | 含义 |
|---|---|
| `resolved_inputs.json` | 数据、MiniLM 与旧检查点的路径和哈希 |
| `baseline_replay/verification.json` | DS 基线复现结果 |
| `arm_selection.json` | train 内部交叉验证与前 4 个训练变体 |
| `pool_inventory.json` | 48 个新候选及哈希 |
| `selection.json` | selector 分区上的冻结前候选权重 |
| `confirmation.json` | 独立 confirmation 分区的一次性通过/回退结论 |
| `frozen/freeze_manifest.json` | 最终冻结版本、紧凑检查点和全部证据 |
| `publication/publication_summary.json` | 冻结后 test 描述性结果与附件 3 输出 |
| `REPORT.md` | 指标、计算量、冻结与发布状态汇总 |

## 边界

- 原 `test` 已在历史版本中暴露，只能作为描述性报告，不能反向调参。
- 附件 3 无标签，只用于冻结推理。
- 新模型必须同时通过 MAE、Macro-F1、完整输入和最差缺失条件门槛。
- `worst_condition_F1` 按最小值计算，不沿用 q2yh 中错误的最大值口径。
- 紧凑检查点不重复保存冻结 MiniLM；部署时必须提供清单中同哈希的共享编码器。
- 算力增加不等于结果必然提升。若 confirmation 门槛未通过，正确结论是继续采用 DS M3。
