# 问题一 模型与优化算法系统对比 —— 服务器执行说明

执行日期：2026-09-25
协议：`q1_comparison_extension_v1_20260925` / `q1-comparison-1.0`

## 1. 本目录内容

| 路径 | 说明 |
|---|---|
| `q1_comparison_extension_v1_20260925/` | 从 GitHub `hurry678/smgs` 的 `问题一/q1_comparison_extension_v1_20260925` 拉取的扩展包（15个文件），SHA-256 清单校验 `CHECKED=14, BAD=0` |
| `benchmark_runs/q1_compare_20260925_full/` | **本次服务器正式运行的完整产物回传**（含 100 条缓存，共 118 个文件） |
| `benchmark_runs/q1_full_results.tar.gz` | 上述产物的服务器端打包（21 MB） |
| `benchmark_runs/q1_ab_frozen_20260925/` | **第二轮 A/B/C 冻结对比完整回传**（`full/`、`smoke/`、`supplementary_pairwise/`） |
| `benchmark_runs/q1_ab_frozen_20260925/full/result_analysis.md` | 第二轮正式结果分析报告（20,878 B） |

## 2. 服务器信息

- SSH 别名：`myserver`（非交互登录）
- 工作目录：`/data2/hy/cts/e_problem`
- 协议包部署位置：`/data2/hy/cts/e_problem/q1_comparison_extension_v1_20260925`
- 运行环境：`/data2/hy/anaconda3/envs/secgpt-vllm/bin/python`（Python 3.10.20）
- 数据根：`/data2/hy/test_5/E题/E题数据`（含 `E题数据/附件1—4`）
- 硬件：Intel Xeon Gold 6330 @ 2.00GHz，92 逻辑核，251 GB RAM，**纯 CPU**
- **未安装任何新依赖**；`numpy 1.26.4 / scipy 1.15.3 / scikit-learn 1.7.2 / OpenCV 4.13.0 / ffmpeg 4.4.2` 均为环境自带

## 3. 实际执行内容

按 `docs/AI服务器执行说明.md` 的阶段顺序执行：

| 阶段 | 命令要点 | 结果 |
|---|---|---|
| 预检 | `--profile smoke --stage preflight` | ✅ `preflight.json.passed=true`（18/18 检查通过） |
| 冒烟 | `--profile smoke --stage git-screening` + `--stage validate` | ✅ 6样本×23候选全部 `ok`，验证通过 |
| **全量筛选** | `--profile full --stage git-screening` | ✅ 100样本×23候选全部 `ok`，wall 1013.3 s，`failures.jsonl` 为空 |
| **结果验证** | `--profile full --stage validate` | ✅ `validation.json.passed=true`（9/9 检查通过） |
| 清单校验 | `sha256sum -c run_manifest_sha256.txt` | ✅ 116/116 OK（服务器与本地各一次） |

### 全量筛选覆盖的候选族（23个）

- **文本**：`bert`(768维) vs `hash`(768维负对照)
- **对齐**：`uniform`、`length_weighted`、`energy_dp_20`、`full_dp_10/20/40`
- **声学子集**：`full_40`、`mfcc_delta_23`、`spectral_mel_31`、`core_prosody_9`、`energy_voiced_f0_3`
- **视觉子集**：`full_90`、`no_hog_54`、`compact_38`
- **视觉帧率**：2.5 / 5.0 / 10.0 fps
- **平滑**：`sigma` = 0.0 / 0.5 / 0.75 / 1.0

每个候选执行 **按 `video_id` 分组、10 次重复**的诊断探针（`probe_repeats=10`），
并输出候选相对固定参考的配对统计与 Holm 校正（`pairwise_statistics.*`）。

### 选择结果（协议冻结规则自动产出）

| 组件族 | 选中 | 综合分 |
|---|---|---:|
| alignment | `length_weighted` | 0.4200 |
| smoothing | `sigma_1p0` | 0.4044 |
| text | `bert` | 0.4181 |
| audio_features | `energy_voiced_f0_3` | 0.3680 |
| vision_features | `compact_38` | 0.3088 |
| vision_fps | `fps_10p0` | 0.2722 |

## 4. 第一轮未能执行的阶段及原因（重要；第二轮已补齐，见 §9）

协议中的 **A/B/C 跨版本对比**未执行，原因是**候选 A 在服务器上客观不存在**：

| 协议要求（候选A必须包含） | 服务器实测 |
|---|---|
| `outputs/alignment.jsonl` | ❌ 全盘（`/data2/hy/cts`）无此文件 |
| `outputs/features/*.npz`（100个） | ❌ 无 |
| `reports/boundary_audit.json` | ❌ 无 |
| `reports/validation.json` | ❌ 仅本次 run 自产的验证文件 |
| MFA 3.4.1 环境 | ❌ `which mfa` 无结果，`secgpt-vllm` 为唯一可用环境 |

服务器现有的第一问正式产物 `artifacts/q1/final_100` 经核验属**候选 B**
（`bert-base-uncased` 768维 / OpenCV Haar 90维 / `energy_pause_dp_v1` / `sigma=0.75`），
**不能冒充候选 A**。

因此以下命令未执行（执行会以 `ValueError: --current-q1-dir is required for version-comparison` 中止）：

- `--stage version-comparison`（A/B 共同边界比较）
- `--stage representation-probe`（A/B/C 统一样本级嵌套探针）

**这不影响**本次已完成的 B 内部系统对比（23候选 × 100样本 × 10重复）的有效性，
但论文中**不得**宣称"已完成 A/B/C 三方案系统比较"。

## 5. 关键结果（供论文使用）

1. **文本容量是唯一强显著组件**：`hash` 相对 `bert`，Macro-F1 **10/10 全负**（-0.0627，Holm p=0.0001），
   Pearson **10/10 全负**（-0.2498，Holm p<0.0001）→ BERT 768维必要性证据最强。
2. **视觉去冗余有效但方向不一致**：`compact_38` 相对 `full_90` 在 Macro-F1 上 **9胜1负**（+0.0310，Holm p=0.0207），
   但 Pearson 更低（-0.0328，区间跨0）→ 必须双指标并报。
3. **对齐方式在样本级探针上不可分辨**：5个候选 Macro-F1 差异全部跨0、Holm p 全为1；
   选中 `length_weighted` 是**约束驱动**（duration_fit 0.9702 vs 其余 0.76—0.86），不是性能驱动。
4. **声学低维权衡**：`energy_voiced_f0_3`（7维）MAE 最低 0.6269、准确率最高 0.51，但 Pearson 最低 0.1303。
5. **平滑**：Macro-F1 差异不显著；`sigma_1p0` 的 Pearson 显著低于 `sigma_0p75`（Holm p=0.0205），但效应量 <0.001，实务可忽略。

> 全部数值为**无监督代理指标**，题给附件1不含词级人工边界真值，**不得**解释为真实边界误差。

## 6. 复现命令

```bash
export PROTOCOL_ROOT=/data2/hy/cts/e_problem/q1_comparison_extension_v1_20260925
export E_ROOT=/data2/hy/test_5/E题/E题数据
export RUN_DIR=/data2/hy/cts/e_problem/benchmark_runs/q1_compare_20260925_full/full
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=20260924 TOKENIZERS_PARALLELISM=false

python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" --work-dir "$RUN_DIR" --profile full --stage preflight
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" --work-dir "$RUN_DIR" --profile full --stage git-screening
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" --work-dir "$RUN_DIR" --profile full --stage validate
```

## 7. 服务器侧只读挂载（本次创建，未改动任何正式产物）

为满足协议脚本对路径的硬编码要求，创建了两个**只读软链接**（未复制、未覆盖）：

```
/data2/hy/cts/e_problem/artifacts/q1/server_final_100/final_100
    -> /data2/hy/cts/e_problem/artifacts/q1/final_100
/data2/hy/cts/e_problem/artifacts/q1/data_cleaning/qualified_samples.csv
    -> /data2/hy/cts/e_problem/data/cleaning/qualified_samples.csv
```

## 8. 后续最小必要步骤（若要补齐 A/B/C）

1. 取得候选 A 的审计产物（`outputs/alignment.jsonl`、100个 `outputs/features/*.npz`、`reports/boundary_audit.json`、`reports/validation.json`）；
2. 在具备 MFA 3.4.1 的环境重放 A；
3. 用同一 `RUN_DIR` 追加 `--current-q1-dir "<A目录>"` 执行 `version-comparison` 与 `representation-probe`；
4. 重新执行 `validate` 并重算 `run_manifest_sha256.txt`。

## 9. 第二轮：A/B/C 冻结对比（2026-09-25，材料补齐后）

### 9.1 远端材料与冻结校验

- 拉取仓库：`C:\work\数模\smgs_repo`，远端 `origin/main` 为 `4522c02`（“Enable frozen Q1 comparison without MFA or raw data”）。
- 候选A冻结包：`ds/q1_comparison_extension/candidates/A_current_audited_q1_2_1.zip`，大小 1,073,211 B，SHA-256 `653dc1e5e5fe6b1b67720f6afdad705331a996d5f0f284f473a750d0b2935058`，147 个条目。
- 扩展包 `MANIFEST.sha256` 校验：服务器侧 18/18 OK。
- 本包已包含候选A的 100 个特征文件、对齐文件、边界审计和验证文件，不再需要 MFA 或附件原始视频参与本轮 A/B/C 冻结对比。

### 9.2 服务器部署

- SSH 别名：`myserver`；工作目录：`/data2/hy/cts/e_problem`
- 第二轮运行目录：`/data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925`
- **未覆盖**第一轮运行目录 `q1_compare_20260925_full`
- 解释器：`/data2/hy/anaconda3/envs/secgpt-vllm/bin/python`（Python 3.10.20；numpy 1.26.4 / scipy 1.15.3 / scikit-learn 1.7.2）
- 硬件：Intel Xeon Gold 6330 @ 2.00GHz，92 逻辑核，251 GB RAM；本轮全流程为纯 CPU。

### 9.3 执行与结果

| 项目 | 结果 |
|---|---|
| 冒烟运行 | `preflight` 与 `validate` 均通过；共享端点 546 |
| 全量运行 | `--profile full --stage frozen-all`；exit 0；wall 48.64 s；CPU 609%；峰值 RSS 148.8 MB |
| 预检 | 9/9 passed |
| 结果验证 | 7/7 passed |
| 失败记录 | `failures.jsonl` 0 行 |
| 探针设计 | 5 外层折 × 3 内层折 × 5 种子；按 `video_id` 分组；标准化仅在折内训练集拟合 |

**fused 视图主结果（均值 ± 标准差，5 种子）**

| 候选 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| A_current_audited | **0.5920 ± 0.0295** | **0.4971 ± 0.0299** | 0.6373 ± 0.0586 | 0.2460 ± 0.0617 |
| B_git_frozen | 0.5500 ± 0.0187 | 0.3966 ± 0.0120 | 0.5555 ± 0.0155 | 0.3193 ± 0.0718 |
| C_hybrid_capacity | 0.5860 ± 0.0219 | 0.4531 ± 0.0138 | **0.5441 ± 0.0252** | **0.3801 ± 0.0921** |

- A/B 边界一致性（模型间一致性，不是人工真值）：A 端点分歧中位数 0.1000 s、p90 0.5472 s、≤0.25 s 比例 71.79%；B 分别为 0.3100 s、0.8300 s、40.84%；A 胜 358 / B 胜 171 / 平 17；差值 bootstrap 95% CI `[0.1159, 0.2183]`。
- 结论应写成“A 的准确率与 Macro-F1 点估计最高、C 的 MAE 与 Pearson 点估计最好；三者差异在配对区间内均跨 0，不能据此宣称统计显著”。

### 9.4 附加配对统计

- 脚本：`supplementary_pairwise/compute_pairwise_supplement.py`，SHA-256 `f85aa478d5e4105597289d9f9b22887d0f43cf7d1155289cca734302a5cf41ef`
- 服务器运行：10,000 次按视频聚类的 bootstrap；exit 0；wall 55.03 s；CPU 128%；峰值 RSS 97,664 KB
- 结果：**所有配对的 95% 区间均跨 0**；MAE 方向一致且效应量大（B、C 在 5/5 种子优于 A，d_z ≈ -1.13 ~ -1.27）；Accuracy / Macro-F1 方向偏向 A。
- 一致性校验：重算点估计与 `probe_metrics.csv` 的最大绝对差为 `2.78e-16`；`target_consistency` 全为 True。
- `supplementary_pairwise/` 是附加分析，不属于冻结的 170 文件清单，不修改 `run_manifest_sha256.txt`。

### 9.5 报告与回传

- 正式报告：`full/result_analysis.md`，20,878 B，SHA-256 `7d5953371ccfdc91914b3d6f69fb6439e3c6137aed7bd953b959bc813d54ba8f`
- 填充记录：`full/result_analysis_fill_record.json`，SHA-256 `93b0c2fffce238cd6b8f9519b6042df97d2706f2e3c7ace6ba900e84042109bd`
- 本地与服务器双向 SHA-256 校验一致；报告不含单位、队员、队伍编号或本机绝对路径。
- 全量目录 170 条清单校验：169 OK / 1 BAD；唯一 BAD 为设计内更新后的 `result_analysis.md`（原占位文本先入清单，报告随后正式填充）。

### 9.6 口径边界（不得误写）

1. A/B 的端点比较是**模型间一致性证据**，不是人工边界误差；A 包 `human_boundary_evaluation.status = not_required`，`human_reviewed_boundaries = 0`。
2. 模板 §4.1 的四项人工边界评估在本轮均为“不可得”，不得用模型间一致性替代。
3. B 的“掩码一致率”和“静音伪词边界数”在本轮为**未判定**，不得写成通过。
4. 附件3/4 的无标签样本不得用于调参或模型选择。
5. 本轮未修改候选B正式产物，未伪造 `boundary_audit.json` 或 `validation.json`。
6. 主结果比较的是 A/B/C 三个候选的冻结表示与探针表现，不等同于真实业务标签上的最终泛化性能。

### 9.7 复现命令（服务器）

```bash
export PROTOCOL_ROOT=/data2/hy/cts/e_problem/ds/q1_comparison_extension
export RUN_DIR=/data2/hy/cts/e_problem/benchmark_runs/q1_ab_frozen_20260925
export PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=20260925 TOKENIZERS_PARALLELISM=false

# 冒烟
"$PY" "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/smoke" --profile smoke --stage frozen-all

# 全量（A/B/C 冻结对比 + 验证 + 清单）
"$PY" "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/full" --profile full --stage frozen-all
```

> 复现时若需要从冻结包重新物化候选A，可追加 `--force-materialize`；脚本会自动读取 `candidates/A_current_audited_q1_2_1.json` 与对应 zip。
