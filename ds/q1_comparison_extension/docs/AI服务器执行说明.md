# AI服务器执行说明：问题一系统对比

本说明供自动化代理在Linux/CPU或GPU服务器上执行。机器协议为[配置JSON](../configs/q1_comparison_protocol.json)，统一入口为[run_q1_comparison_protocol.py](../scripts/run_q1_comparison_protocol.py)。

## 0. 服务器反馈修正

此前服务器只有候选B，缺少候选A的`alignment.jsonl`、100个NPZ、`boundary_audit.json`和`validation.json`，因此此前没有执行A/B/C跨版本比较。候选B不能冒充A，Git历史中的B内部组件筛选也不能替代A/B/C同条件比较。

本目录现提供候选A冻结比较包和自动校验/解包脚本。它解决的是“服务器缺少A输入”，不自动把历史运行变成已完成实验。必须重新执行本说明的`preflight`、`version-comparison`、`representation-probe`和`validate`；在服务器结果被保留前，论文不得宣称“已完成A/B/C三方案系统比较”。

## 1. 执行约束

1. 不修改题给数据、当前正式输出或Git历史产物。
2. 每次实验写入新的`${RUN_DIR}`，不得覆盖另一run_id。
3. 先执行`preflight`，失败时停止；不能通过减少样本绕过错误。
4. `smoke`只检查流程，不能用于论文结论；`full`才是正式比较。
5. 附件3、4不用于调参。
6. 人工参考不存在时，误差字段写`null`；模型间分歧不得改名为accuracy/error。
7. GPU可选；CPU基线必须保留。

## 2. 环境准备

最低依赖：

```bash
export PROTOCOL_ROOT="$REPO/ds/q1_comparison_extension"
python -m pip install -r "$PROTOCOL_ROOT/requirements-comparison.txt"
ffmpeg -version
ffprobe -version
```

该文件限制兼容版本区间；每次运行仍须把实际精确版本写入`environment.json`。候选A和B的冻结比较特征已随仓库提供，因此执行`version-comparison`和`representation-probe`不需要下载BERT、Whisper或安装MFA。只有重新提取候选A才需要其完整依赖、模型资源和MFA 3.4.1。

建议环境变量：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=20260924
export TOKENIZERS_PARALLELISM=false
```

## 3. 输入变量

```bash
export REPO="/path/to/smgs"
export PROTOCOL_ROOT="$REPO/ds/q1_comparison_extension"
export RUN_ID="q1_compare_$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="/path/to/benchmark_runs/${RUN_ID}"
```

冻结A/B/C比较不需要`${E_ROOT}`。脚本默认校验`candidates/A_current_audited_q1_2_1.zip`，并解包到`${RUN_DIR}/materialized/A_current_audited/`。该冻结包包含：

```text
outputs/alignment.jsonl
outputs/features/*.npz
reports/boundary_audit.json
reports/validation.json
```

若服务器已有另一份经过校验的候选A，可显式传入`--current-q1-dir`覆盖默认包；不得把候选B目录传入该参数。

只有重跑Git候选B的原视频组件筛选时才设置：

```bash
export E_ROOT="/path/to/包含E题数据的目录"
```

## 4. 阶段命令

### 4.1 预检和计划

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR" \
  --profile smoke \
  --stage preflight
```

成功标准：返回码0，`preflight.json.passed=true`。预检前会验证候选A压缩包的字节数、SHA-256、ZIP路径安全、100条ID覆盖、验收绑定的100个NPZ摘要及严格发布状态。脚本同时生成：

- `resolved_execution_plan.json`：结构化命令；
- `run_resolved_plan.sh`：可审阅命令，不自动执行；
- `run_manifest_sha256.txt`：当前输出摘要。

### 4.2 冻结比较冒烟

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/smoke" \
  --profile smoke \
  --stage frozen-all
```

该模式使用全部100条冻结A/B产物，但只运行1个探针种子；检查解包、字段、静音分支、跨版本映射和失败列表。通过后才能运行五种子全量探针。Git原视频组件的6条冒烟另用`--e-root ... --profile smoke --stage git-screening`。

### 4.3 Git组件全量筛选

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --e-root "$E_ROOT" \
  --work-dir "$RUN_DIR/full" \
  --profile full \
  --stage git-screening
```

该阶段复用Git正式BERT词特征，从全部100条原视频重建比较缓存，运行：

- 文本：BERT vs Hash；
- 对齐：uniform、length-weighted、energy DP、full DP不同工作频率；
- 声学子集；
- 视觉子集；
- 2.5/5/10fps；
- 多个平滑sigma；
- 10次按`video_id`分组探针；
- 候选相对固定参考的配对统计与Holm校正。

历史Git对比结果可用于核对，但正式运行必须写新目录。

### 4.4 当前版与Git版共享边界比较

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/full" \
  --profile full \
  --stage version-comparison
```

比较器只使用当前版无原文提示ASR中“整短语原词全部匹配且区间正长”的共同子集。Git端从相同字符范围取首/末词时间。输出是模型一致性证据，不是人工准确率。

### 4.5 结果验证

先构建A/B/C统一样本级向量并运行嵌套分组探针：

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/full" \
  --profile full \
  --stage representation-probe
```

候选C在该脚本中使用Git的BERT-base文本向量与当前版声学/视觉向量，不重新估计时间边界，因此可以隔离“文本容量增大”带来的变化。每个模态按逐维有效值计算加权均值、标准差及有效率：文本按实际词元/词项计权，声学和视觉按时序项持续时间计权，不把缺失零计入统计。

随后执行结果验证：

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/full" \
  --profile full \
  --stage validate
```

成功标准：`validation.json.passed=true`。最后重新生成摘要：

```bash
find "$RUN_DIR/full" -type f ! -name run_manifest_sha256.txt -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 > "$RUN_DIR/full/run_manifest_sha256.txt"
```

### 4.6 服务器推荐的一键冻结比较

服务器已有仓库后，直接执行：

```bash
python "$PROTOCOL_ROOT/scripts/run_q1_comparison_protocol.py" \
  --work-dir "$RUN_DIR/full" \
  --profile full \
  --stage frozen-all
```

该命令依次完成候选A校验/解包、冻结输入预检、A/B共享边界比较、A/B/C五种子嵌套探针和输出验证。它不调用MFA、不读取原视频，也不重建候选A或B的底层特征。

## 5. A/B/C正式探针执行要求

Git现有脚本能完成B内部组件筛选；[build_q1_probe_vectors.py](../scripts/build_q1_probe_vectors.py)把A/B/C三套冻结特征转换到共同样本级接口，[evaluate_q1_candidate_vectors.py](../scripts/evaluate_q1_candidate_vectors.py)执行分组嵌套验证：

```text
sample_id: string
video_id: string
y_class: Negative|Neutral|Positive
y_reg: float in [-3,3]
x_text: float[D_t]
x_audio: float[D_a]
x_vision: float[D_v]
text_valid_ratio: float
audio_valid_ratio: float
vision_valid_ratio: float
```

每个模态从时序项计算“逐维有效加权均值＋逐维有效加权标准差＋有效率”。文本按词元/词项计权，声学和视觉按持续时间计权；禁止先补零再把补零计入均值。三模态拼接前可在训练折内标准化。

固定外层5折、内层3折、5个种子。候选超参数仅限：

- 逻辑回归`C ∈ {0.01,0.1,1,10}`；
- Ridge `alpha ∈ {0.1,1,10,100}`；
- C方案若做投影，维度`{128,256}`且投影只在训练折拟合。

AI不得自行扩展搜索空间。任何改动先更新协议版本。

## 6. 资源记录

每个重任务用GNU time或等价工具包裹：

```bash
/usr/bin/time -v -o "$RUN_DIR/resource.txt" \
  python ...command...
```

GPU任务同时记录：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
  --format=csv -l 1 > "$RUN_DIR/gpu_usage.csv"
```

统一写入`resource_usage.csv`：

```text
run_id,candidate,stage,device,cold_or_warm,wall_s,cpu_s,peak_rss_mb,peak_vram_mb,output_bytes,status
```

冷运行先清候选自身特征缓存，不清系统页缓存；热运行复用合法缓存。两者分开报告。

## 7. 输出格式

标准目录：

```text
${RUN_DIR}/
  preflight.json
  resolved_execution_plan.json
  environment.json
  logs/
  cache/
  git_screening/
    comparison_metrics.csv
    comparison_summary.json
    comparison_report.md
    pairwise_statistics.csv
    pairwise_statistics.json
  version_comparison/
    version_comparison.csv
    version_sample_summary.csv
    version_comparison.json
  candidate_vectors/
    A_current_audited.npz
    B_git_frozen.npz
    C_hybrid_capacity.npz
    candidate_vector_manifest.json
  representation_probe/
    probe_predictions.csv
    probe_metrics.csv
    probe_summary.json
  resource_usage.csv
  failures.jsonl
  result_analysis.md
  validation.json
  run_manifest_sha256.txt
```

`failures.jsonl`每行字段：

```json
{
  "run_id": "string",
  "candidate": "A|B|C",
  "sample_id": "string|null",
  "stage": "string",
  "exception_type": "string",
  "message": "string",
  "traceback_tail": "string",
  "retry_count": 0,
  "resolution": "failed|fixed_and_rerun|documented_fallback"
}
```

## 8. 异常处理

| 异常 | 固定动作 |
|---|---|
| 输入缺失/摘要变化 | 停止，不自动下载或替换题给数据 |
| 某候选单样本失败 | 保留失败状态，不从共同分母静默删除 |
| GPU OOM | 降批大小；仍失败则CPU重跑并记录 |
| BERT缓存缺失 | 停止并记录模型revision/SHA后再准备，不切换随机权重 |
| 静音音轨 | 不生成可解释为发音事件的词边界 |
| ASR不匹配 | 保留未匹配；不把其他文本强行映射 |
| 人脸检测冲突 | 导出原帧复核；检测框不自动等于有效人脸 |
| 指标非有限 | 标记运行失败，定位标准化/常量预测，不用0替代 |
| 输出摘要不一致 | 视为产物变更，重新验证和生成报告 |

## 9. AI完成判据

AI仅在以下条件全部成立时宣告完成：

1. 预检、全量运行和输出验证均通过；
2. 所有候选覆盖相同100条或明确报告失败；
3. 结果包含每折、每样本和资源数据，不只有汇总均值；
4. 人工真值与自动证据被明确区分；
5. 论文主表、适用性矩阵、失败案例和选择理由均已填充；
6. `run_manifest_sha256.txt`与实际文件一致；
7. 报告不包含单位、队员、队伍编号或本机绝对路径。
