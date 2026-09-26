# 服务器增量执行任务：问题三可解释性 v2.2

本版本只修复 v2 的 top-support 单边择优偏差。不得训练或替换问题二模型，
不得修改 `ds/`、`优化/q2yhv2_1/`、v2 源码与 v2 服务器结果。所有新输出只
写入 `/data2/hy/q3_runs/q3_explainability_v2_2_20260926`。

## 0. 固定路径

```bash
set -euo pipefail

REPO=/data2/hy/smgs
EXT="$REPO/问题三/q3_explainability_v2_2_20260926"
LEGACY=/data2/hy/cts/e_problem
OLD=/data2/hy/q3_runs/q3_explainability_v2_20260925
RUN=/data2/hy/q3_runs/q3_explainability_v2_2_20260926
PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
DEVICE=cuda:2

export PYTHONPATH="$OLD/python_deps"
RAW="/data2/hy/test_5/E题/E题数据/E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件"
MFA="$OLD/.mfa/bin/mfa"
MFA_DICT="$OLD/resources/mfa/english_mfa_dictionary_v3.1.0.dict"
MFA_ACOUSTIC="$OLD/resources/mfa/english_mfa_acoustic_v3.1.0.zip"
MFA_G2P="$OLD/resources/mfa/english_us_mfa_g2p_v3.0.0.zip"

cd "$REPO"
mkdir -p "$RUN/logs" "$RUN/validation"
```

若服务器仍不是 Git 工作区，按 v2 已验证方式将新扩展同步到 `$EXT`，随后
执行 `MANIFEST_SHA256.txt` 全量校验。不得把 v2 的验证报告复制成 v2.2 报告。

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

## 1. 静态测试与验证预检

```bash
cd "$EXT"
sha256sum -c MANIFEST_SHA256.txt
$PY -m unittest discover -s tests -v 2>&1 | tee "$RUN/logs/unittest.log"
$PY -m py_compile scripts/*.py tests/*.py
$PY -m ruff check . 2>&1 | tee "$RUN/logs/ruff.log"

$PY scripts/q3_preflight_v2.py \
  --scope validation \
  --protocol configs/protocol.json \
  --repo-root "$REPO" \
  --legacy-root "$LEGACY" \
  --out "$RUN/preflight_validation.json" \
  2>&1 | tee "$RUN/logs/preflight_validation.log"
```

必须确认单元测试包含以下反例并通过：

```text
continuous candidates = [0, 1]
control candidates    = [1, 0]
symmetric top-vs-top difference = 0
```

## 2. 验证集偏移

可复用 v2 的验证集偏移，但必须按 SHA 复制并记录；建议直接重建：

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

## 3. 重跑完整验证集三尺度

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

每份报告的
`local_occlusion.top_support_fidelity.comparison_schema` 必须等于
`q3v2.2-symmetric-top-support-v1`，且
`control_selection_symmetric == true`。

## 4. 重新冻结解释参数

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
case "$W" in 3|5|10) ;; *) echo "INVALID W=$W"; exit 1;; esac
SELECTED_REPORT="$RUN/validation/w$W/q3_validation_report.json"
```

若没有尺度同时通过两类对称控制门槛，选择器必须自动回退 `W=5`。

## 5. 冻结后复用已验证附件4产物

只有第 4 步成功后才能执行本节。以下三个 v2 产物不受窗口统计修复影响，可
按协议 SHA 复用：

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

mkdir -p "$RUN/attachment4_data"
cp "$OLD/attachment4_data/attachment4_aligned.npz" "$RUN/attachment4_data/"
cp "$OLD/attachment4_offsets.npz" "$RUN/"
cp "$OLD/alignment_audit.json" "$RUN/"

echo "7572f8bd6fdfb0e40ff0f7446894a6ace44d63a90a17c7ee2a6376d5b3b7567b  $RUN/attachment4_data/attachment4_aligned.npz" | sha256sum -c -
echo "36b8ec63d87e063c9598590e114d40e7ee4ea2ed45e9d4fd97f75be1d6630c5b  $RUN/attachment4_offsets.npz" | sha256sum -c -
echo "2002a77e72809677da74151534e3f8b44f51036b258952a06efd83a0048f886d  $RUN/alignment_audit.json" | sha256sum -c -

$PY - <<PY
import json
from pathlib import Path
d=json.loads(Path("$RUN/alignment_audit.json").read_text())
assert d["status"] == "PASS"
assert d["mfa_aligned_samples"] == 20
assert d["fallback_samples"] == 0
print({"status":"PASS","mfa_aligned_samples":20,"fallback_samples":0})
PY
```

## 6. 按新冻结窗口重跑附件4解释

即使 `W` 仍为 3，也必须重跑，以确保报告含 v2.2 对称统计 schema。

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

## 7. 打包与验证

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

$PY "$EXT/scripts/compare_q3v2_2_results.py" \
  --old-raw "$OLD/final_attachment4/q3_attachment4_raw.json" \
  --new-raw "$RUN/final_attachment4/q3_attachment4_raw.json" \
  --out "$RUN/result_comparison.json" \
  2>&1 | tee "$RUN/logs/result_comparison.log"

$PY "$EXT/scripts/verify_q3v2.py" \
  --repo-root "$REPO" \
  --extension-root "$EXT" \
  --run-dir "$RUN" \
  --out "$RUN/verification.json" \
  2>&1 | tee "$RUN/logs/verify.log"
```

若服务器同步目录不是 Git 工作区，`verify_q3v2.py` 的 protected baseline 检查
会失败；只能在确认 `ds/` 与 `优化/q2yhv2_1/` 的哈希清单均未变化后，使用
完整 Git 工作区执行验证，不得伪造 `protected_baseline_diff`。

## 8. 审计归档与回传

依赖环境仍位于旧 `$OLD`，新 `$RUN` 不含 2.8 GB 第三方环境，可直接导出：

```bash
$PY "$EXT/scripts/export_q3v2_audit.py" \
  --extension-root "$EXT" \
  --run-dir "$RUN" \
  --export-dir "$RUN/audit_export" \
  2>&1 | tee "$RUN/logs/export_audit.log"

cd "$RUN/audit_export"
sha256sum -c SHA256SUMS.txt
```

回传时必须列出：

- 三个尺度下 continuous/random/scatter 各自的 top 单位预算均值；
- 两组**对称 top-vs-top** 聚类 95% CI；
- 最终窗口与是否触发 W=5 回退；
- 相对 v2 的窗口、局部证据、预测和 Shapley 变化；
- 复用的三个附件4产物 SHA；
- `verification.json` 与审计 ZIP SHA/CRC。
