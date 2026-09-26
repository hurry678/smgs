# 问题二定稿说明

## 权威版本

- 模型：`M3-ensemble9`
- 成员：M0 seeds 42 至 50，九模型等权集成
- 发布规则：`C2`
- 决策：v2.1 新候选未通过 confirmation 门槛，保留 DS 基线
- 正式版本清单 SHA-256：
  `3e27642384f016cb06c256ce150623aa1348b14cc8701fa8d4843b9bdb4ef018`
- v2.1 冻结清单 SHA-256：
  `a4a27ee302749b0db79b71bb670a1215f663e2c1cb80a0c98e868db135507b49`

验证集指标：Accuracy `0.620879`、Macro-F1 `0.599707`、MAE `0.579967`。

测试集描述性指标：Accuracy `0.635488`、Macro-F1 `0.596337`、MAE
`0.643349`。

## 内容

- `正式预测器代码/`：DS 的 Q2 数据准备、模型、训练、集成、鲁棒性评估与
  发布脚本。
- `正式结果_M3_ensemble9/`：冻结清单、验证/测试结果、附件3预测和结果 ZIP。
- `定稿决策代码/`：v2.1 高算力复核协议、实现与测试。
- `定稿决策审计/`：服务器执行记录、选择/确认结果和三卷审计 ZIP。
- `正式提交/`：最终附件3 CSV 与 JSON，提交时以此处文件为准。

## 权重位置

九个冻结 checkpoint 已归档在：

`定稿决策审计/audit_export/q2_ds_hc_v2_1_audit_part01.zip`

共享 `all-MiniLM-L6-v2/pytorch_model.bin` 已归档在：

`定稿决策审计/audit_export/q2_ds_hc_v2_1_audit_part03.zip`

审计 ZIP 不重复解压，以避免同一权重占用两份空间。`part02` 保存搜索、选择、
发布及其他运行证据。

## 附件3舍入说明

`定稿决策审计/publication/attachment3_predictions.csv` 是 v2.1 回放后导出，
与历史正式文件仅有一处十进制舍入差：

- 附件3第16条：正式值 `1.482133`，回放导出值 `1.482134`

极性和模型选择均不变。权威提交文件固定为
`正式提交/q2_predictions.csv`，其 SHA-256 为
`7a03dc3243e9bdc6397642afa18119f352a946b512c528f3f4a80018419b616e`。
