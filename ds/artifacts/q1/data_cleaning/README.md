# 问题1数据清洗筛选

本目录由 `scripts/filter_q1_samples.py` 生成。

清洗范围仅限附件1的100条原始样本：

- 文本必须非空；
- 视频文件必须存在且非空；
- 必须同时包含视频流和音频流；
- 媒体时长必须有效；
- 默认执行完整音视频解码，排除损坏文件；
- 标签必须位于 `[-3,3]`；
- `annotation` 必须为 `Negative/Neutral/Positive` 且与标签符号一致；
- 重复 `sample_id` 全部排除。

脚本不会复制、删除或修改原始数据，只输出：

- `qualified_samples.csv`：后续仅使用该清单中的样本；
- `excluded_samples.csv`：被排除样本及原因；
- `all_samples_quality.csv`：全部样本的质检明细；
- `quality_report.json`：筛选规则、数量统计、工具版本和输出路径。

运行方式：

```powershell
C:\Users\Admin\anaconda3\python.exe scripts\filter_q1_samples.py
```
