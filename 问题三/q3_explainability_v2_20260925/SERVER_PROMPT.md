# 服务器执行任务：问题三可解释性 v2

严格按顺序执行。不得修改 `ds/`、`优化/q2yhv2_1/`，不得重新训练问题二模型，
不得用 official test、附件3或附件4选择解释参数。附件4只能在
`freeze_parameters.json` 成功生成后首次读取。

## 0. 固定路径

```bash
set -euo pipefail

REPO=/data2/hy/smgs
EXT="$REPO/问题三/q3_explainability_v2_20260925"
LEGACY=/data2/hy/cts/e_problem
RUN=/data2/hy/q3_runs/q3_explainability_v2_20260925
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
DEVICE=cuda:2

RAW="/data2/hy/test_5/E题/E题数据/E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件"
Q1_MAT=/data2/hy/q1_runs/q1_ab_compare_full_20260925/full/materialized/A_current_audited
MFA="$Q1_MAT/.mfa/bin/mfa"
MFA_DICT="$Q1_MAT/resources/models/mfa/english_mfa_dictionary_v3.1.0.dict"
MFA_ACOUSTIC="$Q1_MAT/resources/models/mfa/english_mfa_acoustic_v3.1.0.zip"
MFA_G2P="$Q1_MAT/resources/models/mfa/english_us_mfa_g2p_v3.0.0.zip"

cd "$REPO"
git pull --ff-only
mkdir -p "$RUN/logs" "$RUN/validation"
```

如果仓库不在 `/data2/hy/smgs`，只修改 `REPO`。其余路径来自已验证的 Q1/Q2
服务器环境；路径不存在时停止并报告，不得下载替代模型或跳过哈希。

九个冻结 checkpoint：

```bash
CHECKPOINTS=(
  "$LEGACY/artifacts/q2/final/M0_seed42.pt"
  "$LEGACY/artifacts/q2/final/M0_seed43.pt"
  "$LEGACY/artifacts/q2/final/M0_seed44.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add/M0_seed45.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add/M0_seed46.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add/M0_seed47.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add2/M0_seed48.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add2/M0_seed49.pt"
  "$LEGACY/artifacts/q2/v2_m0_ensemble_add2/M0_seed50.pt"
)
```

## 1. 静态测试与验证阶段预检

此阶段禁止访问 `$RAW`。

```bash
$PY -m unittest discover -s "$EXT/tests" -v 2>&1 | tee "$RUN/logs/unittest.log"
$PY -m py_compile "$EXT"/scripts/*.py "$EXT"/tests/*.py
$PY -m ruff check "$EXT" 2>&1 | tee "$RUN/logs/ruff.log"

$PY "$EXT/scripts/q3_preflight_v2.py" \
  --scope validation \
  --protocol "$EXT/configs/protocol.json" \
  --repo-root "$REPO" \
  --legacy-root "$LEGACY" \
  --out "$RUN/preflight_validation.json" \
  2>&1 | tee "$RUN/logs/preflight_validation.log"
```

必须确认 `preflight_validation.json.status == "PASS"`。

## 2. 只生成验证集 tokenizer 偏移

```bash
$PY "$EXT/scripts/q3_prepare_offsets_v2.py" \
  --scope q2 \
  --q2-data "$LEGACY/data/q2/q2_data.npz" \
  --q2-splits valid \
  --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
  --out "$RUN/q2_valid_offsets.npz" \
  --audit-json "$RUN/q2_valid_offsets_audit.json" \
  2>&1 | tee "$RUN/logs/q2_valid_offsets.log"
```

必须确认 valid 的 tokenizer 匹配率为 1.0。

## 3. 完整验证集三尺度实验

三次运行必须使用相同 728 条验证样本、相同冻结权重和随机种子：

```bash
for W in 3 5 10; do
  mkdir -p "$RUN/validation/w$W"
  /usr/bin/time -v "$PY" "$EXT/scripts/q3_explain_core_v2.py" \
    --mode validation \
    --data "$LEGACY/data/q2/q2_data.npz" \
    --offsets "$RUN/q2_valid_offsets.npz" \
    --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --freeze-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
    --out-dir "$RUN/validation/w$W" \
    --split valid --n 728 \
    --window-size "$W" --stride "$W" \
    --point-n 128 --top-k 2 --batch-size 64 --seed 2026 \
    --device "$DEVICE" \
    2>&1 | tee "$RUN/logs/validation_w$W.log"
done
```

不得根据运行中间结果新增窗口、修改权重或更换样本。

## 4. 冻结解释参数

```bash
$PY "$EXT/scripts/q3_select_freeze_v2.py" \
  --report "3=$RUN/validation/w3/q3_validation_report.json" \
  --report "5=$RUN/validation/w5/q3_validation_report.json" \
  --report "10=$RUN/validation/w10/q3_validation_report.json" \
  --out "$RUN/freeze_parameters.json" \
  --expected-n 728 --fallback-window 5 \
  --point-scan-n 128 --top-k 2 --batch-size 64 --seed 2026 \
  --q2-freeze-manifest "$REPO/优化/q2yhv2_1/server_results_20260925/freeze_manifest.json" \
  --q2-version-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
  2>&1 | tee "$RUN/logs/freeze_parameters.log"

W=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_window_size"])' "$RUN/freeze_parameters.json")
test "$W" = 3 -o "$W" = 5 -o "$W" = 10
SELECTED_REPORT="$RUN/validation/w$W/q3_validation_report.json"
```

到此为止才能读取附件4。

## 5. 附件4预检与数据转换

```bash
$PY "$EXT/scripts/q3_preflight_v2.py" \
  --scope attachment4 \
  --protocol "$EXT/configs/protocol.json" \
  --repo-root "$REPO" \
  --legacy-root "$LEGACY" \
  --raw-dir "$RAW" \
  --mfa-executable "$MFA" \
  --dictionary "$MFA_DICT" \
  --acoustic-model "$MFA_ACOUSTIC" \
  --g2p-model "$MFA_G2P" \
  --out "$RUN/preflight_attachment4.json" \
  2>&1 | tee "$RUN/logs/preflight_attachment4.log"

$PY "$EXT/scripts/prepare_q3_data_v2.py" \
  --raw-dir "$RAW" \
  --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
  --out-dir "$RUN/attachment4_data" \
  2>&1 | tee "$RUN/logs/prepare_attachment4.log"

$PY "$EXT/scripts/q3_prepare_offsets_v2.py" \
  --scope attachment4 \
  --attachment4 "$RUN/attachment4_data/attachment4_aligned.npz" \
  --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
  --out "$RUN/attachment4_offsets.npz" \
  --audit-json "$RUN/attachment4_offsets_audit.json" \
  2>&1 | tee "$RUN/logs/attachment4_offsets.log"
```

## 6. MFA 时轴映射

```bash
$PY "$EXT/scripts/q3_prepare_alignment_v2.py" \
  --raw-dir "$RAW" \
  --attachment4 "$RUN/attachment4_data/attachment4_aligned.npz" \
  --offsets "$RUN/attachment4_offsets.npz" \
  --base-audit "$RUN/attachment4_data/attachment4_audit.json" \
  --out-audit "$RUN/alignment_audit.json" \
  --work-dir "$RUN/mfa_work" \
  --mfa-executable "$MFA" \
  --mfa-root-dir "$RUN/mfa_cache" \
  --dictionary "$MFA_DICT" \
  --acoustic-model "$MFA_ACOUSTIC" \
  --g2p-model "$MFA_G2P" \
  --expected-mfa-version 3.4.1 \
  --num-jobs 1 --min-mfa-sample-rate 0.8 \
  2>&1 | tee "$RUN/logs/alignment.log"
```

禁止使用 `--allow-low-coverage` 生成正式结果。必须检查：

- `alignment_audit.json.status == "PASS"`；
- MFA 资源 SHA 与协议一致；
- `mfa_sample_rate >= 0.8`；
- 回退样本逐条记录错误原因。

## 7. 附件4冻结推理

```bash
mkdir -p "$RUN/final_attachment4"
/usr/bin/time -v "$PY" "$EXT/scripts/q3_explain_core_v2.py" \
  --mode attachment4 \
  --data "$LEGACY/data/q2/q2_data.npz" \
  --attachment4 "$RUN/attachment4_data/attachment4_aligned.npz" \
  --offsets "$RUN/attachment4_offsets.npz" \
  --audit "$RUN/alignment_audit.json" \
  --model-dir "$LEGACY/models/all-MiniLM-L6-v2" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --freeze-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
  --out-dir "$RUN/final_attachment4" \
  --window-size "$W" --stride "$W" \
  --point-n 20 --top-k 2 --batch-size 64 --seed 2026 \
  --device "$DEVICE" \
  2>&1 | tee "$RUN/logs/attachment4_inference.log"
```

附件4结果不能反向改变 `W`、排序规则、阈值或预测器。

## 8. 打包、独立校验与审计归档

```bash
mkdir -p "$RUN/submission"
$PY "$EXT/scripts/q3_build_outputs_v2.py" \
  --raw-json "$RUN/final_attachment4/q3_attachment4_raw.json" \
  --out-dir "$RUN/submission" \
  --attachment4 "$RUN/attachment4_data/attachment4_aligned.npz" \
  --model-manifest "$REPO/ds/artifacts/q2/v3_m3_ensemble9/version_manifest.json" \
  --explain-script "$EXT/scripts/q3_explain_core_v2.py" \
  --alignment-script "$EXT/scripts/q3_prepare_alignment_v2.py" \
  --math-script "$EXT/scripts/q3v2_math.py" \
  --data "$LEGACY/data/q2/q2_data.npz" \
  --offsets "$RUN/attachment4_offsets.npz" \
  --audit "$RUN/alignment_audit.json" \
  --validation-report "$SELECTED_REPORT" \
  --explanation-parameters "$RUN/freeze_parameters.json" \
  2>&1 | tee "$RUN/logs/package.log"

$PY "$EXT/scripts/verify_q3v2.py" \
  --repo-root "$REPO" \
  --extension-root "$EXT" \
  --run-dir "$RUN" \
  --out "$RUN/verification.json" \
  2>&1 | tee "$RUN/logs/verify.log"

$PY "$EXT/scripts/export_q3v2_audit.py" \
  --extension-root "$EXT" \
  --run-dir "$RUN" \
  --export-dir "$RUN/audit_export" \
  2>&1 | tee "$RUN/logs/export_audit.log"

cd "$RUN/audit_export"
sha256sum -c SHA256SUMS.txt
```

## 9. 回传要求

回传 `$RUN/audit_export/` 的三个文件，并给出：

- 选中的窗口与选择模式；
- 三个窗口的单位预算 top-support 指标和两组聚类 95% CI；
- 完整验证集 Accuracy、Macro-F1、MAE、Pearson；
- 修正后主模态计数、两两交互摘要；
- 附件4三类预测计数与强度范围；
- MFA 成功数、回退样本 ID、音视频证据映射状态计数；
- 20 条输出覆盖结果、提交大小与最终 SHA-256；
- 三次验证和附件4推理的耗时、峰值内存/显存；
- 明确说明未读取 test/附件3做选择，附件4在冻结后才首次读取。
