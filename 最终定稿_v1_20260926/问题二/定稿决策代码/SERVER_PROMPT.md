# 服务器执行任务

你负责执行问题二 DS 高算力优化 v2.1 修正重跑。必须严格执行本文件，不得自行更换数据、模型、
划分、指标、门槛或测试集用途。

## 绝对约束

1. 只写入 `/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925`。
2. 不修改或删除仓库中的 `ds/`、`优化/q2yh/`、`优化/q2yhv2/` 和历史服务器目录。
3. 不使用外部情感标签、附件 3、附件 4 或官方 test 做训练、早停、选模、定权或阈值搜索。
4. `all` 阶段结束前不得执行 `publish`。
5. 任一 SHA-256、基线复现或分组隔离检查失败时立即停止，不得绕过。
6. 不得因算力充足扩展未预注册的超参数；需要新增实验时先另建版本。

## 环境

```bash
cd /data2/hy/smgs
git pull --ff-only

PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
ENTRY=问题二/q2_ds_optimization_v2_1_20260925/scripts/run_q2hc.py
RUN=/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925
LEGACY=/data2/hy/cts/e_problem
```

如果仓库不在 `/data2/hy/smgs`，只替换 `cd` 路径。`LEGACY` 必须指向保存原 DS 数据、
`models/all-MiniLM-L6-v2/pytorch_model.bin` 和 9 个冻结检查点的目录。

## 第一步：依赖与预检

```bash
$PY - <<'PY'
import numpy, sklearn, torch
print({
    "numpy": numpy.__version__,
    "sklearn": sklearn.__version__,
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
})
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 4
PY

$PY "$ENTRY" preflight --legacy-root "$LEGACY" --work-dir "$RUN"
```

必须检查：

- `resolved_inputs.json.status == "passed"`；
- 9 个旧检查点均按冻结 SHA 唯一定位；
- `q2_data.npz` SHA 为
  `2bad10a2913ec3a0a03e3999d36e59117e0efbcc480f379dc61c8bcd4de36c38`；
- 5 个 DS 核心脚本 SHA 全部一致，必须包含 `q2_eval_robustness.py`；
- MiniLM 的 `pytorch_model.bin` 存在。

## 第二步：完整修正重跑

建议在 `tmux` 中执行：

```bash
$PY "$ENTRY" all \
  --legacy-root "$LEGACY" \
  --work-dir "$RUN" \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

该命令依次完成：

1. 基线 9 模型、固定 `mask_seed=2026`、5 复本复现；
2. 重新执行 36 个 train-only CV 任务；
3. 严格按新 CV 中位最佳轮数执行全量训练；同环境下预期 A03/A10/A08/A00 为 7/4/4/3；
4. 重跑 48 个 full-train 候选；
5. 使用新分组盐和新掩码种子执行 selector 选模；
6. 使用新分组与新掩码执行 confirmation；
7. 通过则冻结新模型，否则冻结旧 M3 回退版本。

运行可恢复；进程中断后重复相同命令即可，不得随意使用 `--force`。

上一轮 valid 结果已公开，本轮只能表述为“重复验证修正”，不能称为新的盲 confirmation。

## 第三步：冻结审查

读取：

```bash
$PY - <<'PY'
import json
from pathlib import Path
p = Path("/data2/hy/q2_runs/q2_ds_hc_v2_1_20260925/frozen/freeze_manifest.json")
d = json.loads(p.read_text())
print(json.dumps({
    "status": d["status"],
    "model_version": d["model_version"],
    "confirmation_status": d["confirmation_status"],
    "members": len(d["members"]),
    "compact_checkpoint_bytes": d["compact_checkpoint_bytes"],
    "projected_combined_bytes": d["projected_combined_bytes"],
    "limit": d["combined_submission_byte_limit"],
    "test_policy": d["test_policy"],
}, ensure_ascii=False, indent=2))
PY
```

只有以下条件全部满足才能继续：

- `status` 为 `candidate_frozen_without_test` 或 `baseline_retained_without_new_test`；
- `projected_combined_bytes <= 50000000`；
- 每个紧凑检查点 SHA 与清单一致；
- 此时不存在本版本的 `publication/test_evaluation.json`。

## 第四步：冻结后发布

```bash
$PY "$ENTRY" publish \
  --work-dir "$RUN" \
  --device cuda:0 \
  --allow-post-freeze
```

该结果不能反向改变冻结权重。若新候选在 test 上不理想，报告真实结果，不重新选模。

## 第五步：生成完整审计分卷

```bash
$PY "$ENTRY" export-audit \
  --work-dir "$RUN" \
  --export-dir "$RUN/audit_export"

cd "$RUN/audit_export"
sha256sum -c SHA256SUMS.txt
$PY /data2/hy/smgs/问题二/q2_ds_optimization_v2_1_20260925/scripts/verify_audit_bundle.py "$RUN/audit_export"
```

`audit_index.json.status`、全部 `count_checks`、所有 ZIP CRC 和 `SHA256SUMS.txt` 必须通过。
每个分卷小于 GitHub 100 MB 单文件限制，清单强制使用 LF 行尾。

## 第六步：回传

将 `$RUN/audit_export/` 中的全部文件上传到新的版本化结果目录，不得只上传摘要。该审计包包含：

- 36 个 v2.1 CV 折级结果及日志；
- 48 个修正轮数候选的结果、日志和紧凑检查点；
- selector/confirmation 的全部逐成员预测缓存；
- 最终冻结成员；
- train/valid/test 的 ID、分类标签与回归标签审计副本；
- 与冻结清单同 SHA 的共享 MiniLM 权重；
- 选择、确认、发布结果及源代码快照。

最终报告必须明确：

- 是否通过 confirmation；
- 最终是新候选还是旧 M3 回退；
- valid selector、valid confirmation、完整 valid 描述性指标；
- 冻结后 test 指标；
- 训练总时长、各任务失败数、峰值显存；
- 36 个 CV 原始结果是否全部进入审计分卷；
- 四个入选臂的最终轮数是否严格等于各自 CV 中位最佳轮数；
- 冻结成员、权重、源检查点 SHA 与紧凑检查点 SHA；
- 附件 3 极性计数和 CSV SHA；
- 未使用 test/附件 3 反向调参。
