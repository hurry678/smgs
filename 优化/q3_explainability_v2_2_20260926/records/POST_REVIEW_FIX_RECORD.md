# v2.2 回传后代码审查修复记录

本记录只修复可复用执行脚本与顶层推送清单，不修改模型结果、提交文件、
`server_outputs/` 历史快照或审计 ZIP。

## 修复项

1. `run_validation_v2_2.sh` 不再使用 `run_all || rc=$?`。验证函数直接在
   `set -Eeuo pipefail` 下运行，任一验证管道失败都会立即退出。
2. 使用 `EXIT` trap 将真实退出码原子写入带 `Q3_RUN_ID` 的 JSON 状态文件。
3. 三尺度报告写入 `validation_runs/$Q3_RUN_ID/`，后处理只接受相同 run ID
   的状态文件和报告，不再消费固定路径下的旧 marker。
4. 两份可复用脚本统一为 LF；顶层 `PUSH_MANIFEST_SHA256.txt` 以 LF 重建。

## 重跑方式

```bash
export Q3_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/run_validation_v2_2.sh &
validation_pid=$!
bash scripts/run_after_validation_v2_2.sh
wait "$validation_pid"
```

两个脚本必须继承同一个 `Q3_RUN_ID`。未设置时脚本会在读取任何模型或结果前
立即失败。

## 历史证据边界

以下文件保持提交 `838e08d` 时的原始字节，用于证明服务器实际执行过程：

- `server_outputs/run_validation_v2_2.sh`
- `server_outputs/run_after_validation_v2_2.sh`
- `audit_export/q3_explainability_v2_2_audit.zip`
- `server_outputs/audit_export/q3_explainability_v2_2_audit.zip`

当前科学结果已独立核验通过；本修复只提高失败检测、重复执行隔离和跨平台
清单可用性，不要求重跑实验。
