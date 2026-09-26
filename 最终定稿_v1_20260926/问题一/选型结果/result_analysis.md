# 问题一模型与优化算法对比结果

> 本文件按 `templates/问题一_对比实验结果分析模板.md` 用实测产物填写，全部数字来自本 run 的冻结产物；
> 缺少人工参考的字段填 `null/不可得`，未在本 run 产生的判定标"未判定"，不做推测。
> 第 5、6 节的区间与配对统计来自附加目录 `supplementary_pairwise/`，其方法与口径在该目录
> `pairwise_statistics_supplementary.json` 中完整记录。

## 1. 实验标识

| 字段 | 值 |
|---|---|
| run_id | `q1_ab_frozen_20260925` |
| 协议版本 | `q1-comparison-1.0` |
| Git提交 | `4522c02069a67789b3e7e898dfe96817239195dc`（Enable frozen Q1 comparison without MFA or raw data） |
| 当前版代码摘要 | 候选A冻结包 SHA-256 `653dc1e5e5fe6b1b67720f6afdad705331a996d5f0f284f473a750d0b2935058`（`A_current_audited_q1_2_1.zip`，147 条目，100 NPZ） |
| 输入清单摘要 | 扩展包 `MANIFEST_SHA256.txt` SHA-256 `ff60b8fc91d30dff1d189a15a2b80e4ec486a23f257a3fe7da1178bc6a9bc0cd`；服务器侧 18/18 条目校验 OK |
| 开始/结束时间 | 2026-09-25 12:46:33 / 12:47:21（CST，由 `preflight.json`、`validation.json` 的 UTC 时间戳换算） |
| CPU/GPU/RAM | Intel Xeon Gold 6330 @2.00GHz（92 逻辑核）/ 未使用 GPU（全流程 CPU）/ 251 GB |
| Python/核心依赖 | Python 3.10.20；numpy 1.26.4；scipy 1.15.3；scikit-learn 1.7.2；ffmpeg、ffprobe 4.4.2 |

执行命令：`python ds/q1_comparison_extension/scripts/run_q1_comparison_protocol.py --profile full --stage frozen-all`。

声明门槛（协议 `claim_gate`）核验：`preflight.json.passed=true`（9/9）、`validation.json.passed=true`（7/7）、
`version_comparison` 共享端点 546 > 0、`representation_probe` 覆盖 A/B/C 三候选、环境与资源记录齐备 —— **全部满足**。
`failures.jsonl` 为 0 行，即本 run 无协议级失败样本。

## 2. 候选与假设

| 候选 | 固定组成 | 检验假设 |
|---|---|---|
| A 当前审计版 | 文本 Google BERT Miniature L-2 H-128 A-2（128维）；声学 13维标准MFCC + log-RMS + ZCR + pYIN F0（16维，逐维掩码）；视觉 MediaPipe Face Landmarker 眼口几何（6维）；对齐 MFA 3.4.1 主词边界 + 确定性短语分组 | 更可靠的时间与缺失语义能提高整体可用性 |
| B Git冻结版 | 文本 bert-base-uncased（768维）；声学 Git 冻结实现 40维；视觉 OpenCV Haar 人脸ROI 90维（推荐 no_hog_54 切片）；对齐 energy_pause_dp_v1，20Hz 词级动态规划 | 更大 BERT 和丰富特征能提高表示效用 |
| C 容量增强混合版 | 文本沿用 B 的 bert-base-uncased；声学沿用 A 的 16维逐维掩码；视觉沿用 A 的 MediaPipe 6维；对齐沿用 A 的 MFA 分支 | 保留 A 的审计能力并获得 B 的文本容量增益 |

探针实际向量维度（含掩码通道，`candidate_vectors/candidate_vector_manifest.json`）：
A = 文本 257 / 声学 33 / 视觉 13（fused 306）；B = 1537 / 81 / 181（fused 1801）；
C = 1537 / 33 / 13（fused 1586）。三者样本集合完全一致（100 样本，37 个 video_id 分组）。

## 3. 硬门结果

| 指标 | 阈值 | A | B | C | 结论 |
|---|---:|---:|---:|---:|---|
| 样本覆盖率 | 100% | 100/100 | 100/100 | 100/100 | 三者均通过（`preflight.json`） |
| 源摘要一致率 | 100% | 100/100 | 100/100 | 继承 A/B | 通过（`version_comparison.source_hash_match_count=100`） |
| 有限数值率 | 100% | 100%（NaN/Inf = 0） | 100%（`all_arrays_finite=true`） | 100%（向量构建期校验） | 通过 |
| 合法时间区间率 | 100% | 100%（有序真值时间区间 + 帧索引到 ffprobe PTS） | 100%（`all_word_intervals_valid=true`） | 继承 A | 通过 |
| 静音伪词边界数 | 0 | 0（2 个数字静音样本不产生词边界，改走 clip-context/audio-missing 分支） | 未判定（本 run 仅保留 fallback 词计数 13/11，未对"静音伪词"出具独立判定） | 继承 A | A/C 通过；B 待补判 |
| 掩码一致率 | 100% | 100%（150 维掩码语义检查） | 未判定（B 为稠密特征 + observed ratio，无逐维掩码通道） | 100%（继承 A 掩码通道） | A/C 通过；B 不适用/待补判 |
| 原始素材可追溯率 | 100% | 100/100（metadata 关联 video_id/clip_id 与 PTS） | 100/100（metadata + `q1_manifest.csv`） | 继承 A/B | 通过 |
| 身份信息命中 | 0 | 0（`package_check.anonymity_path_scan_passed=true`） | 未判定 | 未判定 | A 通过 |

未通过项处理说明：B 的两项"未判定"不是硬门失败，而是本 run 的冻结材料未包含对应独立判定（B 侧冻结包不含逐维掩码与静音伪词判定器）。
按协议"不得用后续性能抵消硬门失败"的原则，本文不把这两项计入 B 的通过数；若需在论文中给出 B 的完整硬门结论，须另跑一次带掩码与静音判定的 B 侧提取（新 run_id）。

## 4. 边界与映射

### 4.1 人工参考

| 指标 | A | B | C |
|---|---:|---:|---:|
| 独立人工短语数 | 0 | 0 | 0 |
| 端点Median AE/s | 不可得 | 不可得 | 不可得 |
| 端点P90 AE/s | 不可得 | 不可得 | 不可得 |
| 平均短语IoU | 不可得 | 不可得 | 不可得 |

标注者间差异：不可得。

依据：A 包 `reports/validation.json` 的 `human_boundary_evaluation` 明确为
`status="not_required"`、`required_samples=0`、`reviewed_boundaries=0`、`median_abs_error_s=null`、`p90_abs_error_s=null`；
`result_statistics.json` 的 `aligned_sequences.human_reviewed_boundaries=0`。
本 run 未采集任何人工边界真值，因此第 4.1 节四项全部不可得，论文中不得以任何模型间分歧冒充实测人工误差。

### 4.2 自动独立证据

| 指标 | A | B | C |
|---|---:|---:|---:|
| ASR精确匹配短语/总短语 | 273 / 392 | 273（同一共享子集） | 继承 A |
| 可比端点数 | 546 | 546 | 继承 A/B |
| ASR端点分歧中位数/s | **0.1000** | 0.3100 | 继承 A |
| ASR端点分歧P90/s | **0.5472** | 0.8300 | 继承 A |
| VAD重叠中位数 | 96/100 样本存在 Silero VAD 语音（本 run 未产出重叠比例中位数） | 未产出 | 继承 A |

补充证据：端点分歧 ≤0.25s 比例 A 71.79% vs B 40.84%；≤0.50s 比例 A 85.71% vs B 72.89%；
逐端点胜负 A 358 胜 / 171 负 / 17 平；`git_minus_current_mean_difference_s = 0.1658`，
video-cluster bootstrap 95% 区间 **[0.1159, 0.2183]**（不跨 0）。
A 侧另有跨对齐器（stable-ts）证据：734 个端点分歧中位数 0.2000s、P90 1.2310s，22 个短语、14 个样本的次级证据不完整。

**这些是模型间一致性证据，不是人工边界误差。**

## 5. 表示效用

固定 5 外层 × 3 内层、按 `video_id` 分组、5 个种子（20260924–20260928），标准化只在各折训练集内拟合。
下表为 fused 视图的外层测试结果：均值 ± 标准差（种子间）与 video_id 聚类自助 95% 区间（10,000 次重采样，区间上下界对种子取平均）。

| 候选 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| A | 0.5920 ± 0.0295 [0.4689, 0.7088] | **0.4971 ± 0.0299** [0.3900, 0.5825] | 0.6373 ± 0.0586 [0.4854, 0.8339] | 0.2460 ± 0.0617 [0.1560, 0.4691] |
| B | 0.5500 ± 0.0187 [0.4156, 0.6863] | 0.3966 ± 0.0120 [0.2912, 0.4967] | 0.5555 ± 0.0155 [0.4423, 0.6884] | 0.3193 ± 0.0718 [0.1362, 0.4985] |
| C | 0.5860 ± 0.0219 [0.4613, 0.7112] | 0.4531 ± 0.0138 [0.3470, 0.5501] | **0.5441 ± 0.0252** [0.4395, 0.6672] | **0.3801 ± 0.0921** [0.2096, 0.5384] |

单模态（fused 之外的视图，5 种子均值，完整表见 `probe_metrics.csv`）：

| 候选 | 视图 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|
| A | text / audio / vision | 0.506 / 0.532 / 0.456 | 0.4277 / 0.4668 / 0.4135 | 0.5571 / 0.6140 / 0.5893 | 0.3522 / 0.1291 / 0.0750 |
| B | text / audio / vision | 0.554 / 0.454 / 0.422 | 0.4081 / 0.3565 / 0.3548 | 0.5542 / 0.6403 / 0.7200 | 0.3669 / 0.0505 / **-0.1967** |
| C | text / audio / vision | 0.554 / 0.532 / 0.456 | 0.4081 / 0.4668 / 0.4135 | 0.5542 / 0.6140 / 0.5893 | 0.3669 / 0.1291 / 0.0750 |

分类支持数（每个种子 n=100）：Positive 57 / Neutral 25 / Negative 18。
超参数只在内层选择：文本与视觉用逻辑回归（C ∈ {0.01, 0.1, 1, 10}，L1/L2 混合 α ∈ {0.1, 10, 100}），
分类按内层 macro-F1、回归按内层 MAE 选参，逐折记录见 `probe_metrics.csv` 的 `selected_hyperparameters`。
每折逐样本结果文件：`representation_probe/probe_predictions.csv`（6000 行 = 3 候选 × 4 视图 × 5 种子 × 100 样本）。

## 6. 配对统计与效应量

fused 视图，B−A 与 C−A；差值为正表示前者的指标更优（MAE 为误差，差值正表示 B/C 误差更大）。
区间为配对 video_id 聚类自助 95%（同一重采样索引同时作用于对比双方，10,000 次）；W/T/L 与 Wilcoxon
基于"种子平均后的逐样本差"，Holm 校正覆盖每个指标内的 2 个对比。

| 指标 | 对比 | 平均差 | 95%区间 | W/T/L | Wilcoxon p | Holm p | 效应量 |
|---|---|---:|---|---:|---:|---:|---:|
| Accuracy | B−A | -0.0420 | [-0.1433, 0.0641] | 0/0/5 | 0.0625 | 0.1250 | d_z = -1.202 |
| Accuracy | C−A | -0.0060 | [-0.1061, 0.0965] | 1/3/1 | 1.0000 | 1.0000 | d_z = -0.230 |
| Macro-F1 | B−A | -0.1005 | [-0.2299, 0.0425] | — | — | — | 自助 P(B>A) = 0.099 |
| Macro-F1 | C−A | -0.0440 | [-0.1746, 0.0931] | — | — | — | 自助 P(C>A) = 0.273 |
| MAE | B−A | -0.0818 | [-0.2823, 0.0610] | 5/0/0 | 0.0625 | 0.1250 | d_z = -1.128 |
| MAE | C−A | -0.0932 | [-0.2851, 0.0393] | 5/0/0 | 0.0625 | 0.1250 | d_z = -1.266 |
| Pearson | B−A | +0.0733 | [-0.1703, 0.2329] | — | — | — | 自助 P(B>A) = 0.596 |
| Pearson | C−A | +0.1341 | [-0.0905, 0.2685] | — | — | — | 自助 P(C>A) = 0.762 |

读法与限制：
1) 所有配对区间都跨 0，即在 100 样本、37 个视频组上，A/B/C 的表示效用差异**没有达到统计显著**；结论只能按效应量与方向表述。
2) 方向一致且效应量较大的是 MAE：C 在 5/5 个种子上都优于 A，B 在 5/5 个种子上都优于 A（d_z ≈ -1.13 ~ -1.27）；
   但样本量小、且 A 的回归头在 fused 视图上方差更大（MAE 0.6373 ± 0.0586），因此只作为方向性证据。
3) Accuracy/Macro-F1 方向偏向 A：B 在 5/5 个种子上都低于 A（Accuracy d_z = -1.202），C 与 A 基本持平。
4) Wilcoxon 的输入是种子平均后的逐样本差，非零差样本极少（Accuracy 仅 5 个），p 值只作辅助；重复折与同视频样本都不作为独立样本。
5) 单模态视图的同类统计见 `supplementary_pairwise/pairwise_statistics_supplementary.csv`。注意 C 的 audio/vision 视图与 A 完全同源
   （同一向量），因此这两列的差值为恒 0，仅用于核对而非独立证据。

## 7. 效率与资源

| 候选 | CPU冷启动/s | CPU热运行/s | GPU/s | 峰值RSS/MB | 峰值VRAM/MB | 特征/MB | 模型/MB | 预计整题包/MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 未测 | 未测 | 未测（本 run 全 CPU） | 148.8（全 run 峰值） | 不适用 | 0.573（NPZ）+ 3.117（对齐 JSONL）；探针向量 0.111 | 包内含 BERT miniature 权重（未单列字节） | 24.0157（300 文件，ZIP CRC 通过，< 50MB） |
| B | 未测 | 未测 | 未测 | 同上 | 不适用 | 19.561（100 文件）+ 0.840（metadata）；探针向量 0.634 | bert-base 权重未打包 | 未生成 |
| C | 未测 | 未测 | 未测 | 同上 | 不适用 | 探针向量 0.558（文本取 B、声学/视觉取 A） | 同 B | 未生成 |

本 run 分段资源（`resource_usage.csv`）：

| 阶段 | wall/s | CPU/s | 峰值RSS/MB | 状态 |
|---|---:|---:|---:|---|
| A_vs_B version_comparison | 0.360 | 0.763 | 56.055 | completed |
| A_B_C build_candidate_vectors | 0.748 | 1.150 | 56.055 | completed |
| A_B_C nested_representation_probe | 46.990 | 293.822 | 148.848 | completed |

全 run（`full_resource.txt`）：wall 48.64 s、user 208.18 s、sys 88.10 s、CPU 609%、峰值 RSS 152,420 KB、退出码 0。
附加配对统计（`supplementary_pairwise/supplementary_time.txt`）：wall 55.03 s、user 63.64 s、CPU 128%、峰值 RSS 97,664 KB。

冷/热定义：本 run 未做冷/热对照（`resource_usage.csv` 标 `cache-dependent`），故冷启动列填"未测"，不得引用为冷启动成本。
线程数与批大小：未显式限制线程（609% CPU 表明 sklearn 内部并行）；探针在内存内一次处理 100 样本，无 batch 概念。

## 8. 异常与失败案例

`failures.jsonl` 为 0 行，本 run 无协议级失败样本。下表覆盖冻结产物中标记的全部异常样本与预定分层异常。

| 样本ID | 场景 | A | B | C | 原因与证据 | 处理 |
|---|---|---|---|---|---|---|
| `-NFrJFQijFE$_$1` | 语音对齐不可用 | 走 clip-context 分支，无 MFA 词边界 | 16 词全部 audio-valid | 继承 A | `version_comparison.special_cases`：`current_speech_alignment_available=false` | 保留样本，标记 `clip_context_speech_unavailable_accepted` |
| `-mJ2ud6oKI8$_$1` | 数字静音 | 判定音频缺失，走 clip-context/audio-missing | 13 个 fallback 词、0 个 audio-valid 词 | 继承 A | 同上 + `audio.digitally_silent_samples` | 保留样本，标记 `clip_context_audio_missing_accepted` |
| `-mJ2ud6oKI8$_$2` | 数字静音 | 同上一行 | 11 个 fallback 词、0 个 audio-valid 词 | 继承 A | 同上 | 保留样本，标记 `clip_context_audio_missing_accepted` |
| 6 个零人脸样本 | 小脸/无有效人脸 | 使用裁剪回退，视觉维度置掩码 | 视觉 valid ratio 可为 0 | 继承 A | `visual.zero_face_samples`（含 `-HwX2H8Z4hY$_$9`、`-ri04Z7vwnc$_$0` 等） | 视觉维逐维掩码，不伪造人脸几何 |
| 28 个样本 / 381 帧 | 小脸回退 | 裁剪回退 378/381 帧有效 | — | 继承 A | `visual.crop_fallback_*` | 记录回退来源，供消融使用 |
| 2 个样本 / 4 帧 | 几何被拒 | 拒绝非法几何，置掩码 | — | 继承 A | `visual.geometry_rejected_frames=4` | 置掩码并记录 |
| 82 帧 / 89 帧 | 多人 / 人脸歧义 | 记录歧义与多人计数，按规则选主脸 | — | 继承 A | `visual.multi_face_frames=82`、`ambiguous_face_frames=89` | 记录不确定性，不静默丢弃 |
| 22 个短语 / 14 个样本 | 次级证据不完整 | 主边界仍可用，次级（stable-ts）缺失 | — | 继承 A | `automatic_boundary_protocol.secondary_unavailable_phrases=22` | 降级为"主边界可用、次级不可用" |

## 9. 适用性矩阵

评分 0—3（3 为最适合），证据列给出结果路径或表号。

| 场景 | A | B | C | 证据 |
|---|---:|---:|---:|---|
| CPU限时 | 3 | 2 | 2 | 本 run 全流程 48.64 s（含三候选探针 46.99 s），CPU-only；B 侧冻结包体量更大（特征 19.56 MB vs A 0.573 MB） |
| 50MB限制 | 3 | 2 | 2 | A 整题包 24.0157 MB / 300 文件（`package_check.json`，CRC 与匿名扫描通过）；B/C 未生成整题包 |
| 静音/语音不可观测 | 3 | 1 | 3 | A/C 有显式静音→clip-context 分支（2 样本），B 在数字静音上仍产生 fallback 词（13/11） |
| 文本语义容量 | 2 | 3 | 3 | 单模态文本：B/C Accuracy 0.554 > A 0.506；但 fused 上 A 的 Macro-F1 0.4971 最高（表 §5） |
| 小脸与多人 | 2 | 2 | 2 | A/C 依赖 MediaPipe 几何（6 个零人脸样本走裁剪回退）；B 的 Haar 视觉 Pearson 为 -0.1967 |
| 精确时间回放 | 3 | 1 | 3 | A 端点分歧中位数 0.1000 s / ≤0.25s 比例 71.79%；B 为 0.3100 s / 40.84%（表 §4.2） |

## 10. 最终选择

Pareto 非支配候选（fused 视图）：`{A_current_audited, C_hybrid_capacity}` —— A 在 Accuracy / Macro-F1 上最优，
C 在 MAE / Pearson 上最优，B 在四个指标上均被支配（Accuracy、Macro-F1 显著低于 A，MAE 低于 C，Pearson 低于 C）。

最终选择：`A_current_audited`。

选择理由：

1. **硬门**：A 是三者中唯一在全部硬门项上取得明确通过的候选（覆盖率 100/100、源摘要一致率 100/100、有限数值率 100%、合法时间区间率 100%、静音伪词边界 0、掩码一致率 100%、可追溯率 100%、身份命中 0）；B 有两项"未判定"。
2. **对齐**：A 的独立端点分歧中位数 0.1000 s、≤0.25s 比例 71.79%，优于 B 的 0.3100 s / 40.84%，且差异的聚类自助区间 [0.1159, 0.2183] 不跨 0。
3. **表示效用**：A 在 fused 分类指标上最优（Accuracy 0.5920、Macro-F1 0.4971），与 C 的差异未达显著（C−A 的 Accuracy 区间 [-0.1061, 0.0965]）。
4. **效率与交付**：A 的整题包 24.0157 MB 满足 50MB 约束、全流程 CPU 可复现；C 的特征需拼接两个来源，交付链更长。

未选择B的具体原因：fused Macro-F1 比 A 低 0.1005（区间 [-0.2299, 0.0425]，方向在 5/5 种子上一致），
视觉单模态 Pearson 为负（-0.1967，区间 [-0.3598, 0.0068]），端点分歧中位数 0.3100 s 且 ≤0.25s 比例仅 40.84%，
并在数字静音样本上产生 fallback 词边界（13/11），缺少 A 的静音分支与逐维掩码语义。
其文本容量优势（768 维）只在单模态文本上体现，未转化为 fused 分类增益。

未选择C的具体原因：C 的文本来自 B、声学与视觉来自 A，因此其对齐与时序证据完全继承 A，
**无法提供独立的对齐审计**（论文中不能把 C 的对齐质量当作对 A 的独立复核）；
C 的 MAE/Pearson 优势区间跨 0（MAE [-0.2851, 0.0393]、Pearson [-0.0905, 0.2685]），
且 C 不构成独立的交付包（需拼接两个来源），故作为容量增强备选保留，不替换主线。

## 11. 结论边界

- 人工真值规模：`0`（A 包 `human_boundary_evaluation.status = not_required`，`human_reviewed_boundaries = 0`）。
- 未覆盖样本/短语：3 个 clip-context 样本（2 个音频缺失 + 1 个语音对齐不可用）、22 个次级证据不可用短语、14 个次级证据不完整样本、6 个零人脸样本；`roadmap_all_100_phrase_aligned = false`（100 样本全部完成序列映射，但并非全部完成短语级对齐）。
- 不能从本实验推出：① 任何人工边界精度（MAE/IoU）；② 附件3/附件4 无标签测试样本上的泛化性能（协议禁止用于调参）；③ B/C 的冷启动成本与 GPU 成本（本 run 未测）；④ B 的掩码一致性与静音伪词判定的通过结论（未判定）；⑤ 把模型间分歧当作误差。
- 后续最小必要实验：① 对 ≥30 个样本采集独立人工边界标注，取得人工 AE/IoU；② 对 C 的文本与对齐来源做独立审计，消除与 A 的共享依赖；③ 如需给出 B 的完整硬门结论，补跑一次带逐维掩码与静音伪词判定器的 B 侧提取（新 run_id）。

## 12. 产物与摘要

| 文件 | SHA-256 |
|---|---|
| `representation_probe/probe_summary.json` | `b90dfcc117f096b340732f377736772ed96d403de123805deac93d6f03fffb00` |
| `representation_probe/probe_metrics.csv` | `f324c5bc0ff381c7a9a9d151fd4337965629986f4711fa9971ed0c31fcd64f17` |
| `representation_probe/probe_predictions.csv` | `3b31be1a841701ffd4c001f39174a6d0db8c20a9854a6aa83abefed234962ec7` |
| `version_comparison/version_comparison.json` | `9323f92129312073fb84be7ec734c463b87439f27f0377869b218a1a9d784200` |
| `preflight.json` | `d685397899ca1eda0e637e4b3bc387d77533220babc6b736128166530cc74677` |
| `validation.json` | `aa52d4be203c9f5c8a3734b0327fa15ac8a66c4bf5ce08c7e8ffdbd9cde9240f` |
| `resource_usage.csv` | `6e806ae05896a88e50be4c6340dc5023385c70fb9a88c800a5461e788794066d` |
| `run_manifest_sha256.txt` | `b814c54d41ea0dd4204a2848118bc95f2f37bd646fab1f201b3e5ada37498c6b` |
| `supplementary_pairwise/pairwise_statistics_supplementary.json` | `d4ee07157d55baf9c2e55aa7745b2ed761c18c3ec4b0272e05b9e4dcba2a13f4` |
| `supplementary_pairwise/pairwise_statistics_supplementary.csv` | `cb7488f7cd5ba4effddc9685ee55b87edcaecbe2aa2ccd8cc84da50137d15a35` |
| `supplementary_pairwise/confidence_intervals.csv` | `5abe5403776477cb8a1f6b88ffb037142553f367da498d67dec8a2717ca733c3` |
| `supplementary_pairwise/per_sample_wtl.csv` | `0e841656953982b9c1f4e423bfc9a98fb1ed3a447d4ad2e7095f698540f2e44a` |

说明：
- `comparison_metrics.csv`、`comparison_summary.json`、`pairwise_statistics.json` 属于候选 B 内部 23 方案筛选 run
  （`q1_compare_20260925_full`，1013 s），不在本 run 的产物集合内，本表不重复列出。
- `run_manifest_sha256.txt` 覆盖协议产出的 170 个文件；`supplementary_pairwise/` 为**附加目录**，不修改任何冻结产物，
  其脚本与输入摘要见该目录 `supplementary_run.json`（脚本 SHA-256 `f85aa478d5e4105597289d9f9b22887d0f43cf7d1155289cca734302a5cf41ef`，
  输入 `probe_predictions.csv` 已在上表列出）。本文件由占位文本改为正式报告，见 `result_analysis_fill_record.json`。