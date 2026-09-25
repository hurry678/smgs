# 问题三可解释性增强方案 v2

本目录是问题三的独立增量实现。它只读取已冻结的 `M3-ensemble9 + C2`，
不训练或替换问题二模型，也不修改 `ds/` 与 `优化/q2yhv2_1/`。

## 核心修正

旧实现的 8 个模态联盟按
`(), T, A, V, TA, TV, AV, TAV` 存储，却按整数位掩码列索引计算 Shapley，
导致 `V` 与 `T+A` 两列互换。加和恒等式仍可成立，因此旧验收没有捕获该问题。
v2 固定采用整数位掩码顺序：

```text
(), T, A, T+A, V, T+V, A+V, T+A+V
```

单元测试同时覆盖正确顺序、旧顺序拒绝、加性函数和交互函数解析解。

## 方法

- 预测器：冻结的 9 模型等权集成，发布规则为 C2。
- 模态贡献：三模态 8 联盟精确 Shapley，同时输出三组两两交互。
- 主要参考模态：预测类别 Shapley 中最大的正贡献；若没有正贡献，则回退到
  最大绝对影响模态，并令 `support_modality=null`。
- 局部证据：连续窗口遮挡，和同预算随机连续、点散遮挡比较。
- 强度方向：Positive 使用强度下降，Negative 使用负向强度下降，
  Neutral 使用向零点靠近程度。
- 统计单元：最高支持窗口的差值按 `video_id` 聚类 bootstrap。
- 尺度选择：验证集完整 728 条比较 `w=3/5/10`；若没有尺度同时以正的
  95% CI 下界胜过两类控制，使用预注册回退 `w=5`。
- 时轴定位：附件4冻结后运行 MFA 3.4.1；单样本失败才使用明确标记的比例
  回退，最终要求 MFA 样本覆盖率至少 80%。

## 数据边界

1. 验证阶段只读取 Q2 `valid` 和冻结模型。
2. 参数冻结文件生成后，才允许读取附件4。
3. 附件4不参与窗口选择、排序规则、阈值或模型选择。
4. 官方 test 与附件3不参与问题三流程。

## 目录

```text
configs/protocol.json              预注册协议与不可变哈希
scripts/q3_preflight_v2.py         环境、数据、模型与 MFA 预检
scripts/q3v2_math.py               纯 NumPy 数学核心
scripts/q3_explain_core_v2.py      冻结预测器解释
scripts/q3_select_freeze_v2.py     验证集尺度选择与冻结
scripts/prepare_q3_data_v2.py      冻结后构建附件4 NPZ
scripts/q3_prepare_offsets_v2.py   分作用域 tokenizer 偏移审计
scripts/q3_prepare_alignment_v2.py MFA 主路径时轴映射
scripts/q3_build_outputs_v2.py     最终 CSV/JSONL/清单
scripts/verify_q3v2.py             独立完整性验证
tests/                             数学、映射、协议与选择测试
```

服务器完整命令见 `SERVER_PROMPT.md`。数学定义、选择规则与论文边界见
`docs/问题三_v2建模与验收方案.md`。

## 本地静态验证

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py tests/*.py
ruff check .
```

本机没有冻结检查点、Q2 NPZ、MiniLM 与 MFA 资源，因此完整推理必须在服务器
执行。禁止把本地静态测试描述为模型结果。
