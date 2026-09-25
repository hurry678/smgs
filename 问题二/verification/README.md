# 本地交付前检查

日期：2026-09-25。此处只记录执行计划和辅助工具的检查，不是问题二新模型性能。

| 项目 | 结果 | 证据 |
|---|---|---|
| 参考掩码与统计测试 | 16/16通过 | `protocol_tests.log` |
| 冻结小BERT真实权重检查 | 7/7通过；被遮挡内容变化的输出最大差0 | `text_encoder_check.json` |
| 附件2输入预检 | train3395、valid728，test727仅检查shape/ID；划分间无视频交叉 | `local_preflight/preflight.json` |
| 原生视觉空模态 | train110、valid15；保留样本并显式标记 | 同上 |
| 附件3接口 | 30/30；确认为text_bert/audio/vision，无预计算text | 同上 |
| 来源文件重验 | 附件2文件＋附件3的30个文件SHA均未变化 | `file_management_audit.json` |
| 旧仓库文件保护 | 基线1200个已跟踪文件，无本轮范围外修改 | 同上 |
| 问题一定稿保护 | 原ZIP SHA与复审前一致 | 同上 |
| 神经候选训练 / 模型选型 / 附件3新预测 | not_run，交服务器实施 | `../docs/服务器实施任务.md` |

本地环境：Python3.11.16、NumPy1.26.4、PyTorch2.8.0、Transformers4.57.6。
协议测试使用NumPy与标准库；编码器检查用已存在的独立Python环境。
没有在本地训练新融合模型，没有对附件2 test执行预测评价，没有生成伪造的附件3新结果。

`local_preflight`中的mask_smoke只验证生成器在真实train/valid上可运行，不能作为鲁棒性实验成绩。
服务器必须保留自己的环境锁与全部训练/评价证据，本地passed字段不能代替服务器完成状态。
