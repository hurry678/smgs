# Q1 Comparison Extension v1

目录标识：`q1_comparison_extension_v1_20260925`

用途：存放2026-09-25新增的问题一模型与优化算法系统对比方案。该目录是独立扩展区，不替换、不重命名、不删除仓库原有的问题一文件。

## 文件边界

- 上级目录中原有的`README.md`、`MANIFEST_SHA256.txt`、`docs/`、`scripts/`和`artifacts/`保持基线提交`576795a`的内容。
- 本扩展目录拥有独立的文档、配置、脚本、模板、依赖说明和SHA-256清单。
- 运行产生的缓存、日志和实验结果必须写入仓库外部的`${RUN_DIR}`，不写回上级原始目录。
- 题给附件1—4、当前本地问题一产物和模型缓存均不得上传到此目录。

## 内容

| 路径 | 用途 |
|---|---|
| `docs/问题一_模型与优化算法系统对比方案.md` | A/B/C候选、理论分析、指标、统计与决策框架 |
| `docs/问题一_实验数据说明与使用规范.md` | 附件1—4角色、拆分、防泄漏与保留规则 |
| `docs/AI服务器执行说明.md` | 服务器预检、冒烟、全量比较和验证命令 |
| `configs/q1_comparison_protocol.json` | 机器可读冻结协议 |
| `scripts/run_q1_comparison_protocol.py` | 统一编排、环境记录、资源记录与输出验证 |
| `scripts/compare_q1_versions.py` | 当前MFA方案与Git DP方案的共同边界比较 |
| `scripts/build_q1_probe_vectors.py` | A/B/C统一样本级向量导出 |
| `scripts/evaluate_q1_candidate_vectors.py` | 按`video_id`分组的嵌套线性探针 |
| `templates/` | 实验记录、资源记录和论文结果模板 |
| `FILE_MANAGEMENT_AUDIT.json` | 原始基线文件逐项字节一致性审计 |
| `MANIFEST_SHA256.txt` | 本扩展目录独立摘要 |

## 快速预检

```bash
export REPO="/path/to/smgs"
export PROTOCOL_ROOT="$REPO/问题一/q1_comparison_extension_v1_20260925"

python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "/path/to/E题" \
  --work-dir "/path/to/benchmark_runs/q1_smoke" \
  --current-q1-dir "/path/to/当前第一问" \
  --profile smoke \
  --stage preflight
```

完整顺序和输出契约见[AI服务器执行说明](docs/AI服务器执行说明.md)。

## 基线与审计

- 原仓库基线：`576795a0f60eccc4bae6a17ceb67ffbfe45b4963`
- 首次扩展提交：`254e3e370e910e8e4876020d1c67e6595d74769c`
- 当前文件管理规则：原始基线文件保留；扩展仅位于本目录；不改写Git历史。
