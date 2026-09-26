# 问题三可解释性增强方案 v2.2

本目录是对 v2 服务器结果的独立修正版。它只读取已冻结的
`M3-ensemble9 + C2`，不训练或替换问题二模型，也不修改 v2、`ds/` 或
`优化/q2yhv2_1/`。

## v2.2 核心修正

v2 在每个样本-模态内先选出分类贡献最大的连续窗，却只拿该窗口索引对应的
单次随机控制做比较。连续窗获得了择优权，控制组没有；即使两组候选值集合
完全相同，只要排列不同，也可能产生虚假的正差值。

v2.2 对 `continuous`、`random_contiguous`、`point_scatter` 三种模式执行完全
对称的处理：

1. 每个候选先计算 `signed class delta / actual masked positions`；
2. 每种模式独立选择单位预算分数最大的候选；
3. 比较 continuous top 与两种 control top；
4. 仍按 `video_id` 做 5000 次 cluster bootstrap。

若没有尺度同时以正的 95% CI 下界胜过两类对称控制，严格回退到预注册
`w=5`。该修正只影响窗口冻结与局部证据，不影响预测、精确 Shapley 或 MFA。

## 延续的 v2 修正

旧实现的 8 个模态联盟按
`(), T, A, V, TA, TV, AV, TAV` 存储，却按整数位掩码列索引计算 Shapley，
导致 `V` 与 `T+A` 两列互换。加和恒等式仍可成立，因此旧验收没有捕获该问题。
v2/v2.2 固定采用整数位掩码顺序：

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
- 统计单元：三种模式分别独立择优后，差值按 `video_id` 聚类 bootstrap。
- 尺度选择：验证集完整 728 条比较 `w=3/5/10`；若没有尺度同时以正的
  95% CI 下界胜过两类控制，使用预注册回退 `w=5`。
- 时轴定位：正式纳入服务器验证过的 MFA v2.1 解析器，支持圆/方括号事件及
  纯标点 token；已有 20/20 MFA 审计可在新冻结后按 SHA 复用。

## 数据边界

1. 验证阶段只读取 Q2 `valid` 和冻结模型。
2. v2.2 参数冻结文件生成后，才允许复用或读取附件4产物。
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
scripts/compare_q3v2_2_results.py  强制验证预测器输出相对 v2 不变
scripts/verify_q3v2.py             独立完整性验证
tests/                             数学、映射、协议与选择测试
```

服务器增量重跑命令见 `SERVER_PROMPT.md`。数学定义、选择规则与论文边界见
`docs/问题三_v2.2建模与验收方案.md`，v2 证据与修正原因见
`docs/v2结果审查与v2.2修正记录.md`。

## 本地静态验证

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py tests/*.py
ruff check .
```

本机没有冻结检查点与 Q2 NPZ，因此验证集三尺度比较必须在服务器执行。
禁止沿用 v2 的 W=3 作为 v2.2 结论，必须由新报告重新冻结。
