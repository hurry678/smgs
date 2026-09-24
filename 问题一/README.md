# 问题一：多模态情感特征提取与时序对齐

本目录归档 E 题问题一的正式生成物、全过程报告与复现脚本。

## 目录

- docs/问题一_全过程记录.md：问题一完整过程、方法、验收、对比实验和复现说明。
- docs/问题一_模型与优化算法系统对比方案.md：A/B/C候选、理论分析、指标、统计和执行计划。
- docs/问题一_实验数据说明与使用规范.md：附件1—4的数据角色、拆分与防泄漏规则。
- docs/AI服务器执行说明.md：服务器预检、冒烟、全量、跨版本比较和验证命令。
- configs/q1_comparison_protocol.json：AI可读取的冻结协议、候选、指标、硬门和输出契约。
- templates/：实验记录、资源记录和论文结果分析模板。
- scripts/：数据筛选、三模态特征提取、方法对比和统计分析脚本。
- artifacts/q1/data_cleaning/：100条视频的数据清洗与质量审计结果。
- artifacts/q1/pipeline_smoke/：全链路冒烟测试产物。
- artifacts/q1/server_final_100/：服务器正式全量结果，包含100个NPZ特征、100个metadata、对齐记录、处理日志、审计报告及压缩包。
- artifacts/q1/comparison/：问题一候选方法对比、配对统计和最终选型报告。
- MANIFEST_SHA256.txt：本目录所有文件的SHA-256清单。

## 正式结果

- 原始样本覆盖：100/100
- 正式特征文件：100个NPZ
- 元数据文件：100个metadata
- 对齐记录：100条
- 处理日志：100条
- 自动审计：100/100成功，NaN/Inf为0，词区间异常为0
- 正式对齐后端：`energy_pause_dp_v1`
- 文本特征：BERT
- 语音特征：40维声学表示
- 视觉推荐特征：去除HOG后的54维子集
- 时间分辨率：20Hz对齐；视觉采样5fps；平滑sigma=0.75

详细结论和证据边界见 docs/问题一_全过程记录.md。

## 新版系统对比入口

在包含`E题数据/`的目录上先运行只读预检：

```bash
python scripts/run_q1_comparison_protocol.py \
  --e-root "/path/to/E题" \
  --work-dir "/path/to/benchmark_runs/q1_smoke" \
  --current-q1-dir "/path/to/当前第一问" \
  --profile smoke \
  --stage preflight
```

预检通过后按[AI服务器执行说明](docs/AI服务器执行说明.md)依次运行冒烟、全量组件筛选、跨版本共享边界比较和结果验证。`smoke`不得用于论文结论；正式比较使用`full`。

## 说明

原始竞赛视频未上传，仅上传从附件1视频提取的派生特征、元数据、日志和报告。复现实验中的服务器路径保留在报告和元数据中，用于说明处理来源；不包含密钥或口令。
