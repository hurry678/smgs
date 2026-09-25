# 问题一模型与优化算法对比结果

> 本文件由实测产物填写。`不可得`表示该项在当前服务器上确实不存在对应产物，不以推测值代替。
> **执行范围声明**：本次服务器执行完成的是协议中的 **B（Git冻结版）内部组件全量筛选 + 配对统计 + 输出验证**。
> 协议中的 **A（当前审计版）** 与 **C（容量增强混合版）** 因服务器上不存在其审计产物（`outputs/alignment.jsonl`、
> `outputs/features/*.npz`、`reports/boundary_audit.json`、`reports/validation.json`）且无 MFA 3.4.1 环境，
> 本次**未执行**跨版本阶段。相关章节按要求填 `不可得`，不做任何替代或伪造。

## 1. 实验标识

| 字段 | 值 |
|---|---|
| run_id | `q1_compare_20260925_full` |
| 协议版本 | `q1-comparison-1.0` |
| Git提交 | `不可得`（服务器工作目录 `/data2/hy/cts/e_problem` 非 Git 仓库）；扩展包以 `MANIFEST_SHA256.txt` 独立摘要冻结，校验 `CHECKED=14, BAD=0` |
| 当前版代码摘要 | `不可得`（候选A代码与产物均不在服务器） |
| 输入清单摘要 | `qualified_samples.csv` = `08f03d967a02bc3c7d3ac27a738cefd318583b47d02afe88bb264f8fc95979d7`（100条，仅清洗筛选表） |
| 开始/结束时间 | `2026-09-25 10:47 CST` / `2026-09-25 11:04:45 CST` |
| CPU/GPU/RAM | Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz，92 逻辑核，251 GB RAM，**纯CPU**（未使用GPU） |
| Python/核心依赖 | Python 3.10.20；numpy 1.26.4；scipy 1.15.3；scikit-learn 1.7.2；OpenCV 4.13.0；ffmpeg/ffprobe 4.4.2 |

线程与随机性：`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`，`PYTHONHASHSEED=20260924`，`TOKENIZERS_PARALLELISM=false`。

## 2. 候选与假设

| 候选 | 固定组成 | 检验假设 | 本次状态 |
|---|---|---|---|
| A 当前审计版 | BERT Miniature L-2 H-128 A-2（128维）；16维标准MFCC+log-RMS+ZCR+pYIN F0；MediaPipe 6维眼口几何；MFA 3.4.1 主词边界 | 更可靠的时间与缺失语义能提高整体可用性 | **未执行**（产物缺失、无MFA环境） |
| B Git冻结版 | bert-base-uncased（768维）；40维冻结声学；OpenCV Haar ROI 90维；`energy_pause_dp_v1` 20Hz 词级DP | 更大BERT和丰富特征能提高表示效用 | **已执行**（100样本、10次重复） |
| C 容量增强混合版 | 较大冻结BERT + A的声学/视觉定义 + A的MFA对齐 | 保留A的审计能力并获得B的文本容量增益 | **未执行**（依赖A） |

B 的实际运行配置（由正式产物 `artifacts/q1/final_100/metadata/*.json` 与 `pipeline_environment.json` 记录）：
`text.backend=precomputed`（bert-base-uncased 768维，word级）、`audio=40维@16kHz/25ms窗/10ms跳`、
`vision=opencv_haar_face_region 90维@5fps`、`alignment=energy_pause_dp_v1@20Hz`、
`smoothing=gaussian_filter1d sigma=0.75（仅在连续全维有效段内平滑）`。

## 3. 硬门结果

本阶段（组件筛选）可直接核验的硬门如下；B 的 23 个候选**全部通过**。

| 指标 | 阈值 | A | B | C | 结论 |
|---|---:|---:|---:|---:|---|
| 样本覆盖率 | 100% | 不可得 | 100/100（23/23候选） | 不可得 | B通过 |
| 源摘要一致率 | 100% | 不可得 | 本阶段未测量（属`version-comparison`阶段） | 不可得 | 未测量 |
| 有限数值率 | 100% | 不可得 | 1.0000（23/23候选） | 不可得 | B通过 |
| 合法时间区间率 | 100% | 不可得 | 1.0000（23/23候选） | 不可得 | B通过 |
| 静音伪词边界数 | 0 | 不可得 | 本阶段未测量（属A的`boundary_audit`） | 不可得 | 未测量 |
| 掩码一致率 | 100% | 不可得 | 1.0000（区间合法率与有效率均满值） | 不可得 | B通过 |
| 原始素材可追溯率 | 100% | 不可得 | 100/100（缓存由附件1原视频重建） | 不可得 | B通过 |

`failures.jsonl` 为空（0行），无候选、无样本失败，无需修复重跑。

## 4. 边界与映射

### 4.1 人工参考

| 指标 | A | B | C |
|---|---:|---:|---:|
| 独立人工短语数 | 不可得 | 0（无人工词边界真值） | 不可得 |
| 端点Median AE/s | 不可得 | null | 不可得 |
| 端点P90 AE/s | 不可得 | null | 不可得 |
| 平均短语IoU | 不可得 | null | 不可得 |

标注者间差异：`不可得`（本题未提供人工边界标注）。

### 4.2 自动独立证据

| 指标 | A | B | C |
|---|---:|---:|---:|
| ASR精确匹配短语/总短语 | 不可得 | 不可得（该证据由`version-comparison`阶段生成，本次未执行） | 不可得 |
| 可比端点数 | 不可得 | 不可得 | 不可得 |
| ASR端点分歧中位数/s | 不可得 | 不可得 | 不可得 |
| ASR端点分歧P90/s | 不可得 | 不可得 | 不可得 |
| VAD重叠中位数 | 不可得 | 不可得 | 不可得 |

**必须附句**：上述自动指标若存在，也只是模型间一致性证据，不是人工边界误差。本次因该阶段未执行，相关字段全部留空。

## 5. 表示效用

本次执行的是**组件筛选探针**：按 `video_id` 分组重复交叉验证、`probe_repeats=10`，指标为跨重复均值，
**不是**协议第5节规定的"固定5外层×3内层、5个种子"的 A/B/C 嵌套探针。该嵌套探针依赖候选A，本次未执行。

全部 23 个候选：`n_samples=100`，`status=ok`。

| family | method | Accuracy | Macro-F1 | MAE | Pearson | 维度 | duration_fit |
|---|---|---:|---:|---:|---:|---:|---:|
| alignment | uniform | 0.50 | 0.4458±0.0369 | 0.5613 | 0.3121±0.0446 | 1799 | 0.7634 |
| alignment | length_weighted | 0.51 | 0.4542±0.0440 | 0.5628 | 0.3094±0.0427 | 1799 | 0.9702 |
| alignment | energy_dp_20 | 0.50 | 0.4445±0.0415 | 0.5670 | 0.2972±0.0465 | 1799 | 0.8396 |
| alignment | full_dp_10 | 0.50 | 0.4422±0.0365 | 0.5551 | 0.3199±0.0478 | 1799 | 0.8461 |
| alignment | full_dp_20 | 0.50 | 0.4465±0.0398 | 0.5628 | 0.3063±0.0473 | 1799 | 0.8569 |
| alignment | full_dp_40 | 0.51 | 0.4512±0.0429 | 0.5614 | 0.3116±0.0483 | 1799 | 0.8534 |
| smoothing | sigma_0p0 | 0.50 | 0.4463±0.0447 | 0.5635 | 0.3073±0.0471 | 1799 | 0.8569 |
| smoothing | sigma_0p5 | 0.50 | 0.4425±0.0441 | 0.5632 | 0.3070±0.0472 | 1799 | 0.8569 |
| smoothing | sigma_0p75 | 0.50 | 0.4465±0.0398 | 0.5628 | 0.3063±0.0473 | 1799 | 0.8569 |
| smoothing | sigma_1p0 | 0.50 | 0.4492±0.0409 | 0.5627 | 0.3054±0.0476 | 1799 | 0.8569 |
| text | bert | 0.51 | 0.4557±0.0277 | 0.5583 | 0.3486±0.0488 | 1537 | 0.8569 |
| text | hash | 0.43 | 0.3929±0.0292 | 0.7237 | 0.0989±0.0348 | 1537 | 0.8569 |
| audio_features | full_40 | 0.45 | 0.3761±0.0583 | 0.7888 | 0.1243±0.0396 | 81 | 0.8569 |
| audio_features | mfcc_delta_23 | 0.42 | 0.3621±0.0442 | 0.7805 | 0.0615±0.0721 | 47 | 0.8569 |
| audio_features | spectral_mel_31 | 0.44 | 0.3677±0.0450 | 0.8047 | 0.1105±0.0577 | 63 | 0.8569 |
| audio_features | core_prosody_9 | 0.50 | 0.3826±0.0422 | 0.6943 | 0.0338±0.0428 | 19 | 0.8569 |
| audio_features | energy_voiced_f0_3 | 0.51 | 0.3641±0.0320 | 0.6269 | 0.1303±0.0479 | 7 | 0.8569 |
| vision_features | full_90 | 0.39 | 0.3116±0.0419 | 0.9407 | -0.1528±0.0680 | 181 | 0.8569 |
| vision_features | no_hog_54 | 0.38 | 0.3149±0.0594 | 0.8457 | -0.0980±0.0708 | 109 | 0.8569 |
| vision_features | compact_38 | 0.41 | 0.3426±0.0535 | 0.8452 | -0.1856±0.0804 | 77 | 0.8569 |
| vision_fps | fps_2p5 | 0.39 | 0.3089±0.0492 | 0.9144 | -0.1750±0.1073 | 181 | 0.8569 |
| vision_fps | fps_5p0 | 0.39 | 0.3116±0.0419 | 0.9407 | -0.1528±0.0680 | 181 | 0.8569 |
| vision_fps | fps_10p0 | 0.40 | 0.3252±0.0556 | 0.9018 | -0.1663±0.0614 | 181 | 0.8569 |

分类支持数：`不可得`（本阶段未导出每折类别支持数；筛选探针输出为跨重复汇总）。
超参数只在内层选择：`不适用`（筛选探针为固定超参数的诊断性分组CV，不含内层超参搜索）。
每折结果文件：`不可得`（筛选阶段未落盘每折明细；`git_screening/comparison_summary.json` 含10次重复明细）。

### 5.1 选择规则（协议冻结，非事后调整）

`hard constraints first; 0.45*macro_f1 + 0.25*max(pearson,0) + 0.15*duration_fit + 0.10*active - 0.05*(dim/max_dim)`

## 6. 配对统计与效应量

同一次重复内所有候选使用相同的 `video_id` 分组折分，仅比较共享的10次重复。
**重复间不是独立样本**，p值只作方法选型证据，不作正式泛化显著性声明。

### probe_repeat_macro_f1

| family | candidate | reference | 平均差 | 95%区间 | W/T/L | paired t p | Holm p | Wilcoxon p |
|---|---|---|---:|---|---:|---:|---:|---:|
| alignment | length_weighted | full_dp_20 | 0.0077 | [-0.0096, 0.0249] | 5/1/4 | 0.3422 | 1.0000 | 0.4258 |
| alignment | full_dp_40 | full_dp_20 | 0.0046 | [-0.0075, 0.0168] | 6/0/4 | 0.4096 | 1.0000 | 0.4316 |
| alignment | uniform | full_dp_20 | -0.0007 | [-0.0157, 0.0143] | 5/0/5 | 0.9206 | 1.0000 | 0.9219 |
| alignment | energy_dp_20 | full_dp_20 | -0.0020 | [-0.0168, 0.0128] | 5/0/5 | 0.7661 | 1.0000 | 0.9219 |
| alignment | full_dp_10 | full_dp_20 | -0.0043 | [-0.0178, 0.0092] | 5/0/5 | 0.4878 | 1.0000 | 0.7695 |
| audio_features | core_prosody_9 | full_40 | 0.0065 | [-0.0398, 0.0527] | 7/0/3 | 0.7595 | 1.0000 | 0.6250 |
| audio_features | spectral_mel_31 | full_40 | -0.0084 | [-0.0382, 0.0214] | 4/0/6 | 0.5398 | 1.0000 | 0.5566 |
| audio_features | energy_voiced_f0_3 | full_40 | -0.0120 | [-0.0592, 0.0352] | 6/0/4 | 0.5785 | 1.0000 | 0.7695 |
| audio_features | mfcc_delta_23 | full_40 | -0.0140 | [-0.0523, 0.0242] | 5/0/5 | 0.4278 | 1.0000 | 0.4316 |
| smoothing | sigma_1p0 | sigma_0p75 | 0.0027 | [-0.0040, 0.0093] | 4/5/1 | 0.3884 | 0.9478 | 0.4375 |
| smoothing | sigma_0p0 | sigma_0p75 | -0.0002 | [-0.0093, 0.0088] | 5/2/3 | 0.9536 | 0.9536 | 0.6406 |
| smoothing | sigma_0p5 | sigma_0p75 | -0.0040 | [-0.0126, 0.0045] | 2/4/4 | 0.3159 | 0.9478 | 0.3125 |
| text | hash | bert | -0.0627 | [-0.0848, -0.0407] | 0/0/10 | 0.0001 | 0.0001 | 0.0020 |
| vision_features | compact_38 | full_90 | 0.0310 | [0.0093, 0.0527] | 9/0/1 | 0.0103 | 0.0207 | 0.0137 |
| vision_features | no_hog_54 | full_90 | 0.0033 | [-0.0262, 0.0328] | 5/0/5 | 0.8081 | 0.8081 | 0.8457 |
| vision_fps | fps_10p0 | fps_5p0 | 0.0135 | [-0.0093, 0.0363] | 8/0/2 | 0.2133 | 0.4267 | 0.1934 |
| vision_fps | fps_2p5 | fps_5p0 | -0.0028 | [-0.0309, 0.0253] | 4/0/6 | 0.8270 | 0.8270 | 0.9219 |

### probe_repeat_pearson

| family | candidate | reference | 平均差 | 95%区间 | W/T/L | paired t p | Holm p | Wilcoxon p |
|---|---|---|---:|---|---:|---:|---:|---:|
| alignment | full_dp_10 | full_dp_20 | 0.0136 | [0.0098, 0.0175] | 10/0/0 | 0.0000 | 0.0001 | 0.0020 |
| alignment | uniform | full_dp_20 | 0.0058 | [-0.0001, 0.0117] | 8/0/2 | 0.0548 | 0.1095 | 0.0488 |
| alignment | full_dp_40 | full_dp_20 | 0.0053 | [0.0008, 0.0098] | 8/0/2 | 0.0254 | 0.0763 | 0.0273 |
| alignment | length_weighted | full_dp_20 | 0.0031 | [-0.0025, 0.0086] | 5/0/5 | 0.2447 | 0.2447 | 0.4922 |
| alignment | energy_dp_20 | full_dp_20 | -0.0091 | [-0.0135, -0.0046] | 0/0/10 | 0.0013 | 0.0051 | 0.0020 |
| audio_features | energy_voiced_f0_3 | full_40 | 0.0060 | [-0.0330, 0.0451] | 5/0/5 | 0.7350 | 0.7350 | 1.0000 |
| audio_features | spectral_mel_31 | full_40 | -0.0137 | [-0.0368, 0.0094] | 2/0/8 | 0.2117 | 0.4233 | 0.1055 |
| audio_features | mfcc_delta_23 | full_40 | -0.0628 | [-0.0940, -0.0316] | 1/0/9 | 0.0014 | 0.0041 | 0.0039 |
| audio_features | core_prosody_9 | full_40 | -0.0905 | [-0.1315, -0.0495] | 1/0/9 | 0.0007 | 0.0030 | 0.0039 |
| smoothing | sigma_0p0 | sigma_0p75 | 0.0010 | [-0.0006, 0.0026] | 6/0/4 | 0.1881 | 0.1881 | 0.3750 |
| smoothing | sigma_0p5 | sigma_0p75 | 0.0007 | [-0.0001, 0.0016] | 6/0/4 | 0.0860 | 0.1720 | 0.1602 |
| smoothing | sigma_1p0 | sigma_0p75 | -0.0008 | [-0.0014, -0.0003] | 2/0/8 | 0.0068 | 0.0205 | 0.0098 |
| text | hash | bert | -0.2498 | [-0.2961, -0.2034] | 0/0/10 | 0.0000 | 0.0000 | 0.0020 |
| vision_features | no_hog_54 | full_90 | 0.0548 | [0.0248, 0.0848] | 9/0/1 | 0.0026 | 0.0051 | 0.0059 |
| vision_features | compact_38 | full_90 | -0.0328 | [-0.0763, 0.0107] | 3/0/7 | 0.1219 | 0.1219 | 0.2324 |
| vision_fps | fps_10p0 | fps_5p0 | -0.0135 | [-0.0345, 0.0075] | 4/0/6 | 0.1800 | 0.3600 | 0.2754 |
| vision_fps | fps_2p5 | fps_5p0 | -0.0222 | [-0.0721, 0.0277] | 2/0/8 | 0.3412 | 0.3600 | 0.4316 |

**关键读法（以区间与效应为主）**：

1. 文本容量是唯一"大效应且稳定"的组件：`hash` 相对 `bert` 在 Macro-F1 上 **10/10 全负**（-0.0627，Holm p=0.0001），
   Pearson 上 **10/10 全负**（-0.2498，Holm p<0.0001）。BERT 768维的必要性有本次最强证据。
2. 视觉 `compact_38` 相对 `full_90` 在 Macro-F1 上 **9胜1负**（+0.0310，Holm p=0.0207），说明 HOG 等大块冗余维度
   对小样本分组CV是负担；但它在 Pearson 上反而更低（-0.0328，区间跨0），**分类与回归结论不一致**，不能只报一个方向。
3. 对齐族内 5 个候选的 Macro-F1 差异**全部跨0、Holm p 全为1**，即"对齐方式"在样本级探针上几乎不可分辨；
   但 `length_weighted` 的 duration_fit=0.9702 明显优于其余（0.76—0.86），这是它被规则选中主因，属**约束驱动**而非性能驱动。
4. 声学族：`energy_voiced_f0_3`（7维）MAE 最低（0.6269）且准确率最高（0.51），但 Pearson 最低（0.1303）——
   低维强正则的典型权衡；`core_prosody_9` 在 Pearson 上显著劣于 `full_40`（-0.0905，Holm p=0.0030）。
5. 平滑 `sigma`：Macro-F1 差异均不显著；`sigma_1p0` 在 Pearson 上**显著低于** `sigma_0p75`（-0.0008，Holm p=0.0205），
   效应量极小（8/10 为负但幅度 <0.001），实务上属于"可忽略但方向一致"。
6. 视觉帧率：`fps_10p0` 优于 `fps_5p0`（8/2）但区间跨0，`fps_2p5` 更差，方向一致但证据不足。

## 7. 效率与资源

| 候选 | CPU冷启动/s | CPU热运行/s | GPU/s | 峰值RSS/MB | 峰值VRAM/MB | 特征/MB | 模型/MB | 预计整题包/MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 不可得 | 不可得 | 不适用 | 不可得 | 不适用 | 不可得 | 不可得 | 不可得 |
| B（全量筛选，含100条缓存重建） | 1013.330 | 1.095（配对统计） | 未使用 | 506.676 | 不适用 | 见下 | 不适用 | 见下 |
| C | 不可得 | 不可得 | 不适用 | 不可得 | 不适用 | 不可得 | 不可得 | 不可得 |

冷/热定义：本次为**半冷**——`cache/` 目录为新建空目录，100 条样本的音频/视觉/对齐缓存全部在本 run 内重建；
BERT 文本特征复用正式产物 `artifacts/q1/final_100/features`（协议允许，且不重新下载模型）。
线程数与批大小：`OMP/MKL/OPENBLAS=4`；无批处理（逐样本处理）。

规模参考：本 run 目录总占用 22 MB（含 100 条缓存 NPZ）。CPU 累计 11944.4 s，wall 1013.3 s，
并行加速比约 11.8×（受 I/O 与 FFmpeg 解码限制）。

## 8. 异常与失败案例

`failures.jsonl` 为空；23 个候选 × 100 样本全部 `status=ok`，无失败样本。故本表无可填行的**运行失败**。

但必须记录以下**已知系统性局限**（不是运行失败，属方法边界）：

| 样本ID | 场景 | A | B | C | 原因与证据 | 处理 |
|---|---|---|---|---|---|---|
| 全体100条 | 无人工作词边界真值 | 不可得 | 全部候选的边界指标均为无监督代理 | 不可得 | 题给附件1只含文本与情感标签，无词级时间标注 | 已在报告中声明：不得把代理指标解释为真实边界误差 |
| 全体100条 | 视觉特征与情感标签负相关 | 不可得 | `full_90` Pearson = -0.1528；`compact_38` = -0.1856 | 不可得 | Haar ROI + 手工块特征在5fps下信息量有限；100样本不足以稳定回归 | 视觉族结论只作"容量/冗余"判断，不用于情感预测结论 |
| 声学低维候选 | 小样本过拟合风险 | 不可得 | `energy_voiced_f0_3` 准确率最高但 Pearson 最低 | 不可得 | 7维强正则 vs 40维弱正则的典型权衡 | 两方向同时报告，不择优隐藏 |

## 9. 适用性矩阵

评分 0—3（3=最适合）。A/C 列因产物缺失全部留空。

| 场景 | A | B | C | 证据 |
|---|---:|---:|---:|---|
| CPU限时 | 不可得 | 2 | 不可得 | 全量100条缓存重建 1013 s / 92核CPU；峰值RSS 507 MB |
| 50MB限制 | 不可得 | 3 | 不可得 | 本 run 全量产物 22 MB（含缓存），远低于50MB |
| 静音/语音不可观测 | 不可得 | 2 | 不可得 | 对齐族 `mean_active_fraction` 0.43—0.49，`mean_pause_quality` 0.58—0.73，具备静音分支 |
| 文本语义容量 | 不可得 | 3 | 不可得 | BERT vs Hash：Macro-F1 +0.0627（10/10），Pearson +0.2498（10/10），Holm p≤0.0001 |
| 小脸与多人 | 不可得 | 1 | 不可得 | Haar 级联检测，无小脸/多人专门分支；视觉族 Pearson 为负 |
| 精确时间回放 | 不可得 | 2 | 不可得 | `length_weighted` duration_fit 0.9702；但无人工真值可核验绝对精度 |

## 10. 最终选择

Pareto非支配候选：在**本阶段可评估的B内部组件**中，文本选 `bert`、视觉选 `compact_38` 具有明确优势；
对齐族 5 者性能不可分辨，由 duration_fit 约束区分。

最终选择（按协议冻结规则自动产出，见 `comparison_report.md`）：

| 组件族 | 选中 | 综合分 | 主要依据 |
|---|---|---:|---|
| alignment | `length_weighted` | 0.4200 | duration_fit 0.9702 最优；Macro-F1 0.4542 族内最高 |
| smoothing | `sigma_1p0` | 0.4044 | Macro-F1 0.4492 族内最高（但 Pearson 略低，效应量<0.001） |
| text | `bert` | 0.4181 | 相对 hash 的强显著优势 |
| audio_features | `energy_voiced_f0_3` | 0.3680 | MAE 0.6269 最低、准确率 0.51 最高、仅7维 |
| vision_features | `compact_38` | 0.3088 | Macro-F1 9/10 优于 full_90，维度减半 |
| vision_fps | `fps_10p0` | 0.2722 | Macro-F1 族内最高（证据强度不足，区间跨0） |

选择理由：

1. **硬门**：全部候选通过覆盖、有限性、区间合法、掩码一致；无候选被硬门淘汰。
2. **对齐**：族内性能差异统计上不可分辨，故由"时长拟合"这一可核验约束决定，避免用噪声选型。
3. **表示效用**：文本容量是唯一强显著组件；声学以低维换取 MAE；视觉以 `compact_38` 去冗余。
4. **效率**：全量100条在纯CPU上 17 分钟完成，产物 22 MB，满足"50MB/CPU限时"的竞赛约束。

未选择B的具体原因：**不适用**——本阶段执行的即是B内部筛选，不存在"选B"的取舍。
若将来完成 A/B/C 跨版本阶段，该栏需按当时的嵌套探针结果重填。

未选择C的具体原因：**不可得**——候选C依赖候选A的对齐与审计产物，A不存在时C无法构造，
因此本次不能给出C的取舍理由，也不能宣称"已比较A/B/C"。

## 11. 结论边界

- 人工真值规模：**0**（本题未提供词级人工边界标注）。
- 未覆盖样本/短语：候选A、候选C的全部内容未覆盖；协议中 `version-comparison` 与 `representation-probe` 两个阶段未执行。
- 不能从本实验推出：
  1. 任何**绝对**词边界精度（无人工真值，全部为无监督代理）；
  2. A/B/C 三方案的系统级优劣（A、C未执行）；
  3. 问题二/三的模型结论（附件3、4 本次完全未使用）；
  4. 泛化显著性（10次重复共享同一批100样本，重复间非独立）。
- 后续最小必要实验：
  1. 取得候选A的审计产物（`outputs/alignment.jsonl` + 100个 `outputs/features/*.npz` + `reports/boundary_audit.json` + `reports/validation.json`），
     在具备 MFA 3.4.1 的环境重放A；
  2. 用同一 `RUN_DIR` 追加 `--current-q1-dir` 执行 `version-comparison` 与 `representation-probe`；
  3. 若需人工边界真值，须另行标注至少数十条短语并报告标注者间一致性。

## 12. 产物与摘要

| 文件 | SHA-256 |
|---|---|
| `comparison_metrics.csv` | `15eb4f708a87443474e3329e94bece01c0ac30b3254852a20590cffa76aa9ee5` |
| `comparison_summary.json` | `11ef81448956a56f753b8fcd44d53cfaac94ca085de7c2eb028aff9050abe69a` |
| `pairwise_statistics.json` | `f82ace1914793a5d12de32e891eabe817a64eae3fe564220acbc7ebb1d821792` |
| `version_comparison.json` | 不存在（阶段未执行） |
| `resource_usage.csv` | `f3cbaed2a7cc646dc3990bbdeb26b43073163260451bfcf4a1922f0802858438` |
| `run_manifest_sha256.txt` | 自引用文件，摘要见文件本身（不再内嵌） |
| `preflight.json` | `6d81101c1870c865d218c7da68daa5b041efb7ab8b0a7a10f3f23be25b97431e` |
| `validation.json` | `d2bb20168387c57c11d5b5b1670f06510910ca3822cf82978aeba43ce186b719` |

`run_manifest_sha256.txt` 覆盖本目录全部 116 个文件（含 `result_analysis.md` 本身），服务器端 `sha256sum -c` 校验 **116/116 OK**；本地回传副本复核 **116/116 OK**。

因清单包含 `result_analysis.md`，二者存在自引用关系，故本表不内嵌清单自身摘要，以服务器实测校验结果为准。

## 13. 复现命令（服务器）

```bash
export PROTOCOL_ROOT=/data2/hy/cts/e_problem/q1_comparison_extension_v1_20260925
export E_ROOT=/data2/hy/test_5/E题/E题数据
export RUN_DIR=/data2/hy/cts/e_problem/benchmark_runs/q1_compare_20260925_full/full
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=20260924 TOKENIZERS_PARALLELISM=false

# 1) 预检（要求 preflight.json.passed=true）
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" --work-dir "$RUN_DIR" --profile full --stage preflight

# 2) 全量组件筛选（本次 1013 s）
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" --work-dir "$RUN_DIR" --profile full --stage git-screening

# 3) 结果验证（要求 validation.json.passed=true）
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" --work-dir "$RUN_DIR" --profile full --stage validate
```

> 注：`--stage all` 会连带执行 `version-comparison`/`representation-probe`，在无候选A时会以
> `ValueError: --current-q1-dir is required for version-comparison` 中止。本次因此分阶段执行。

