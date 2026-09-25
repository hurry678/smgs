# 问题二：局部模态缺失鲁棒预测执行包

版本：`q2-plan-1.0`；日期：2026-09-25；参考仓库：`aacd3ee8e1e227229468872e86c9046b082e2ea2`。

这是服务器 AI 的实施协议、实验计划与经过检查的辅助工具，**不是已经训练完成的新模型或新结果**。
服务器须按任务表实现训练入口、完成验证后再冻结。全部新增源码、数据缓存、权重、日志及结果放在本目录；
原 `ds/`、`优化/`、`问题一/` 只读。不要覆盖 `ds/submission/`。

## 推荐路线

统一使用附件2 `aligned_50.pkl` 与附件3 `对齐版本/`；文本均从 `text_bert` 经同一个冻结编码器生成。
随包提供与本地问题一一致的 128 维小 BERT（含权重、词表、配置及来源摘要），避免服务器再次缺输入。
附件3只有 `text_bert/audio/vision`，不能将附件2预计算的 `text` 直接当作部署主线。

你的“二进制掩码＋注意力融合＋分类回归双任务”作为核心候选 `M1`。
同时设置简单池化、普通门控、连续缺失结构、跨时刻注意力等对照；
优化先用 AdamW，再按预算比较 SGD、SAM、PCGrad、GradNorm；不能预先宣布注意力或某优化器更好。

## 阅读与执行顺序

1. [SERVER_PROMPT.md](SERVER_PROMPT.md)：可直接交给服务器 AI 的指令。
2. [docs/方案与旧版取舍.md](docs/方案与旧版取舍.md)：数学模型、双任务改进、旧版可继承与不可照搬之处。
3. [docs/数据与缺失协议.md](docs/数据与缺失协议.md)：字段、掩码、连续区间、文本遮挡与防泄漏。
4. [docs/实验矩阵与验收标准.md](docs/实验矩阵与验收标准.md)：分阶段预算、选择规则、统计、消融、交付。
5. [docs/服务器实施任务.md](docs/服务器实施任务.md)：实施顺序、训练接口、日志与失败处理。
6. [configs/protocol.json](configs/protocol.json)、[configs/candidates.json](configs/candidates.json)：机器可读协议。
7. [scripts/preflight.py](scripts/preflight.py)、[scripts/protocol_core.py](scripts/protocol_core.py)、
   [scripts/evaluate_predictions.py](scripts/evaluate_predictions.py)：预检、掩码参考实现和独立指标/聚类区间工具。

## 可立即执行的命令

在仓库根目录执行；`--aligned` 明确指向附件2文件，`--attachment3-dir` 明确指向30个对齐版文件所在目录。

```bash
python -m unittest discover -s 问题二/tests -v
python 问题二/scripts/preflight.py \
  --aligned "/实际数据路径/附件2-数据集特征文件/aligned_50.pkl" \
  --attachment3-dir "/实际数据路径/附件3-模态缺失特征样本/对齐版本" \
  --output-dir 问题二/runs/preflight_server
```

预检仅作附件3接口检查，不根据其缺失分布改动训练方案。输出 `preflight.json` 为真之后，
服务器按任务表实现并运行训练；此包没有虚构一个尚未实现的“一键全量训练”命令。
`tests/` 只需 NumPy；训练另需 PyTorch、Transformers、Safetensors。

## 完成与否的标准

- 先完成 `B0 + B2 + M1`、完整/缺失验证、题目规定的类型/位置/时长分析及附件3全部30条预测；
  再在预算内增加模型和优化算法。
- 模型/超参数/发布规则只由附件2 train/valid 决定；附件3/4不参与调参，既有 test 已曝光需如实披露。
- 所有条件保留逐样本预测和掩码摘要，以 `video_id` 为重采样单位，不能把条件或种子冒充独立样本。
- 验证必须同时评价原始双头输出与最终发布输出。中性强度为0；正负类强度符号一致。
- 包括必需编码器权重在内，最终 Q1/Q2/Q3 **合包≤50,000,000字节**。三个问题共用同一小BERT时只保留一份。
- 各候选实际完成状态、失败结果、选择理由、环境锁、检查点摘要及附件3 CSV 都须回传新目录。

本地预检与工具验证结果保存在 `verification/`。训练和性能验收状态仍为 `not_run`。
