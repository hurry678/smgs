# 问题三 v2.2 服务器执行记录

## 1. 任务范围

本次执行对象为 Git 提交 `808b602ac054540947772fac7f9906f35e0b5f13` 中的
`问题三/q3_explainability_v2_2_20260926`。该版本仅修正 v2 的 top-support
单边择优偏差，预测器、Shapley 分解和附件4推理规则均未改变。

- 服务器：`myserver`
- GPU：`cuda:2`
- 运行目录：`/data2/hy/q3_runs/q3_explainability_v2_2_20260926`
- 扩展源码：`/data2/hy/smgs_git_808b602/问题三/q3_explainability_v2_2_20260926`
- 协议：`configs/protocol.json`
- 协议 SHA-256：`831d0079b543b7280ea51a3770de11d9a96920ab096aca498e1c264b97769637`
- Python 环境：`/data2/hy/anaconda3/envs/secgpt-vllm/bin/python`
- 依赖路径：`/data2/hy/q3_runs/q3_explainability_v2_20260925/python_deps`
- 本次没有安装或升级任何依赖，没有修改共享环境。

## 2. 执行时间线

| 时间（CST） | 事件 | 结果 |
|---|---|---|
| 2026-09-26 10:20 | 启动三尺度验证 tmux 会话 `q3v22_validation` | 开始 |
| 2026-09-26 10:26 | `W=3` 验证完成 | PASS |
| 2026-09-26 10:30 | `W=5` 验证完成 | 对称 scatter 门槛未通过 |
| 2026-09-26 10:33 | `W=10` 验证完成，验证总退出码 0 | 对称 scatter 门槛未通过 |
| 2026-09-26 10:33:57 | 冻结解释参数 | 选中 `W=3` |
| 2026-09-26 10:34:01 | 附件4冻结后推理 | 20/20 样本 |
| 2026-09-26 10:34:19 | 打包、预测器不变性比较、独立验收、审计导出 | `postprocess.exit_code=0` |

三尺度验证耗时：

- `W=3`：`6:10.54`，退出码 0；
- `W=5`：`4:25.94`，退出码 0；
- `W=10`：`3:02.24`，退出码 0。

## 3. 三尺度对称 top-support 结果

比较 schema 均为 `q3v2.2-symmetric-top-support-v1`，
`control_selection_symmetric=true`。

| W | continuous | random-contiguous | point-scatter | continuous-random 95% CI | continuous-scatter 95% CI | 是否合格 |
|---:|---:|---:|---:|---|---|---|
| 3 | 0.053386397863017875 | 0.04202510709329737 | 0.04423371006939197 | [0.008098506908349401, 0.014758199750378854] | [0.004684681667907746, 0.014318634709801025] | 是 |
| 5 | 0.036516363897233624 | 0.030037903027601547 | 0.033294696823331846 | [0.004103997697817101, 0.008764402492685548] | [-0.0006959137712652495, 0.006301216831072192] | 否 |
| 10 | 0.020923288929501127 | 0.01912397106061157 | 0.019480478667992084 | [0.00019689033650811645, 0.0033859033855972095] | [-5.1999747196204575e-05, 0.0029712869349022312] | 否 |

选择规则在读取附件4前已经固定：只有同时满足
`continuous - random_contiguous` 与 `continuous - point_scatter` 两组
视频级聚类 95% CI 下界均大于 0 的尺度才合格；合格尺度中取
budget-normalized top-support 均值最大者。`W=3` 是唯一合格尺度，因此
最终冻结为 `W=3`，没有触发 `W=5` 回退。

验证集预测指标（三个尺度一致）：

- Accuracy：`0.6208791208791209`
- Macro-F1：`0.5997068114591327`
- MAE：`0.5799670219421387`
- Pearson：`0.6635369958302701`
- 样本数：`728`

## 4. 附件4冻结后推理

- 样本数：`20`
- 参与统计的 sample-modality 数：`59`
- 每个控制模式扰动数：`566`
- 附件4推理耗时：`22.950748920440674` 秒
- MFA 对齐样本：`20`
- 比例回退样本：`0`
- `alignment_audit.status=PASS`

复用的三个 v2 已验证附件4产物：

| 产物 | SHA-256 |
|---|---|
| `attachment4_aligned.npz` | `7572f8bd6fdfb0e40ff0f7446894a6ace44d63a90a17c7ee2a6376d5b3b7567b` |
| `attachment4_offsets.npz` | `36b8ec63d87e063c9598590e114d40e7ee4ea2ed45e9d4fd97f75be1d6630c5b` |
| `alignment_audit.json` | `2002a77e72809677da74151534e3f8b44f51036b258952a06efd83a0048f886d` |

预测器不变性比较结果：

- v2 raw SHA：`a15978c4ee057617ef057a684712dac84aa7d8886d09ea18fce79379ff0687f3`
- v2.2 raw SHA：`e3f6e79023e819a67614ebcb615f09ae4310cd4131476d047f9b166c0729b5ae`
- 极性变化数：`0`
- 最大强度差：`0.0`
- 最大 log-prob 差：`0.0`
- 主要模态变化数：`0`
- 证据变化数：`0`
- 主要模态计数：text `16`、audio `1`、vision `3`
- 关键证据计数：text `40`、audio `31`、vision `33`

## 5. 独立验收与审计

`verification.json`：

- `status=PASS`
- 源码清单文件数：`26`
- 源码清单 SHA：`400028a469ac40255250428f151921eb910c512e211840b124ab1b53f307224b`
- submission 大小：`321284` 字节
- 受保护基线差异：`clean`

审计包：

- `q3_explainability_v2_2_audit.zip`
- 字节数：`3996851`
- SHA-256：`35e93f87d3c27aa18991f59ae085d77199d705da5ff7232befbd621ee6ff33c4`
- 归档条目数：`74`
- `SHA256SUMS.txt` 本地复核通过。

## 6. 复现入口

服务器脚本已随本目录保存：

- `scripts/run_validation_v2_2.sh`：静态检查、验证预检、三尺度验证；
- `scripts/run_after_validation_v2_2.sh`：等待验证完成后执行冻结、附件4预检、附件4推理、打包、比较、验收和审计导出。

完整日志位于 `server_outputs/logs/`，关键 JSON 位于
`server_outputs/` 和 `server_outputs/submission/`，独立审计包位于
`audit_export/`。