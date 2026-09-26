# 问题三 v2 时轴映射器（MFA token -> 原文/视频）健壮性修复记录

- 记录生成时间：2026-09-26 09:42 (CST)
- 运行目录：`/data2/hy/q3_runs/q3_explainability_v2_20260925`
- 冻结扩展根：`/data2/hy/smgs/问题三/q3_explainability_v2_20260925`
- 影响范围：**仅附件4 MFA 词级 token 到原文/视频时轴的映射器**。

## 1. 修复性质声明

本次修复属于**映射器实现健壮性修复（parser robustness fix）**，不是模型变更、
不是实验协议变更、不是参数变更：

- 未改动任何 checkpoint、权重、集成方式或冻结清单；
- 未改动 `freeze_parameters.json`（该文件于 2026-09-26T09:19:18+08:00 生成，早于本修复）；
- 未改动窗口 `W=3`、`stride=3`、`point_n`、`top_k`、`batch_size`、`seed` 等任何冻结参数；
- 未改动协议 `configs/protocol.json`、判定规则、Shapley/遮挡/点扫描数学定义；
- 未改动提交文件生成脚本（`q3_build_outputs_v2.py`）与校验脚本（`verify_q3v2.py`）。

修复只影响“MFA 输出 token 如何在原文中定位并消费”这一纯文本解析步骤，
使原本因排版字符（圆括号 vs 方括号、孤立标点）无法定位的样本能够正常映射，
而不改变任何数值计算路径。

## 2. 脚本与版本

| 文件 | 位置 | SHA-256 |
|---|---|---|
| `q3_prepare_alignment_v2.py`（原始 v2，失败版本） | `$EXT/scripts/` | `ef7e8728765791468c03735746037edac79654d081e619fbf86e279d2c6f8d73` |
| `q3_prepare_alignment_v2_1.py`（修复版 v2_1，正式使用） | `$RUN/scripts/` | `d335e728c9fb3b061251d9de93cccd6d759f0550aeeefad9145250ece006fd9f` |

说明：v2_1 修复脚本**只放在 `$RUN/scripts/`**，未写入冻结扩展根 `$EXT`。
冻结扩展根保持 23 个文件的 source manifest 不变，避免破坏 Q3 v2 的源码清单校验。

## 3. 失败现象（首次 v2 运行）

```
{
  "status": "FAIL",
  "mfa_aligned_samples": 15,
  "fallback_samples": 5,
  "mfa_sample_rate": 0.75
}
```

- 回退样本 ID：`04, 09, 10, 14, 20`（5/20，`mfa_sample_rate = 0.75`）
- 覆盖率门槛 `--min-mfa-sample-rate 0.8` 未通过，流程按协议**立即停止**，
  未生成 `alignment_audit.json` 的正式版本，也未进入附件4冻结推理。
- 未使用 `--allow-low-coverage` 绕过门槛。

### 3.1 逐样本根因

| 样本 ID | `alignment_route` | `alignment_error` |
|---|---|---|
| 04 | `approximate_proportional_fallback` | `ValueError: cannot map MFA token 18: ']'` |
| 09 | `approximate_proportional_fallback` | `ValueError: MFA emitted [bracketed] without matching source text` |
| 10 | `approximate_proportional_fallback` | `ValueError: MFA emitted [bracketed] without matching source text` |
| 14 | `approximate_proportional_fallback` | `ValueError: MFA emitted [bracketed] without matching source text` |
| 20 | `approximate_proportional_fallback` | `ValueError: MFA emitted [bracketed] without matching source text` |

根因归纳（均为解析规则过窄，非数据缺陷）：

1. **样本 04**：原文含 `"GO BLUE" goodbye?]`，MFA 把结尾的 `]` 作为独立 token 输出。
   v2 规则把“归一化后为空的 token”一律视为无法映射，直接抛错。
2. **样本 09/10/14/20**：原文中非语音事件使用**圆括号**排版（如 `(uhh)`、`(laughs)`），
   而 MFA 统一输出 `[bracketed]` 占位标签。v2 只按方括号 `\[[^\]]+\]` 回查原文，
   在圆括号排版下匹配失败。

## 4. 修复规则（最小改动）

对应 diff（`$EXT/scripts/q3_prepare_alignment_v2.py` -> `$RUN/scripts/q3_prepare_alignment_v2_1.py`）：

```diff
         if label == "[bracketed]":
-            match = re.search(r"\[[^\]]+\]", text[raw_cursor:])
+            match = re.search(r"\([^)]+\)|\[[^\]]+\]", text[raw_cursor:])
             if match is None:
                 raise ValueError("MFA emitted [bracketed] without matching source text")
...
             cursor += len(normalized(text[char_start:char_end]))
+        elif not target:
+            char_start = text.find(label, raw_cursor)
+            if char_start < 0:
+                raise ValueError(f"cannot map MFA token {word_index}: {label!r}")
+            if normalized(text[raw_cursor:char_start]):
+                raise ValueError(f"MFA skipped source text before {label!r}")
+            char_end = char_start + len(label)
         else:
             found = normalized_text.find(target, cursor)
-            if not target or found < cursor:
+            if found < cursor:
                 raise ValueError(f"cannot map MFA token {word_index}: {label!r}")
```

三条规则：

1. `[bracketed]` 占位标签同时识别**圆括号**与**方括号**两种排版：
   `\([^)]+\)|\[[^\]]+\]`。
2. 归一化后为空的 token（纯标点，如 `]`）按**原文字面量**在 `raw_cursor` 之后查找，
   找到即消费其字符区间；若中间跳过了非空白原文则仍抛错。
3. 保留原有 **cursor 单调前进**约束与 **最终全覆盖断言**（映射结束后必须消费完整原文），
   因此放宽匹配不会掩盖真实的错位。

## 5. 修复后验证

- 先在 5 个失败样本上做 dry-run，随后对 20 个样本全量重跑第 6 步。
- 重跑命令与首次完全一致（同一 `--min-mfa-sample-rate 0.8`、同一 MFA 3.4.1 资源、
  同一 `num_jobs=1`），仅替换映射器脚本。
- 结果：

```
{
  "status": "PASS",
  "mfa_aligned_samples": 20,
  "fallback_samples": 0,
  "mfa_sample_rate": 1.0,
  "audit": "/data2/hy/q3_runs/q3_explainability_v2_20260925/alignment_audit.json"
}
alignment status: PASS sample_rate: 1.0
[2026-09-26 09:35:36] STEP6 v2_1 done
```

- 20/20 样本 `alignment_route = mfa_forced_alignment`，无回退样本（`fallback_sample_ids = []`）。
- MFA 资源 SHA 与协议逐项一致：
  - dictionary `975bf9c7791535c5aec57995e0bd2b77eb7ca364d1d974c2134e07bb0f16b079`
  - acoustic `2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068`
  - g2p `9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02`

## 6. 失败证据与审计文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `$RUN/logs/alignment_failed_v2_first.json` | `619a8c56b6c43428ecd773ec7cffe7af8749b5a495e2d9578f1189bf1036ea5a` |
| `$RUN/logs/alignment_failed_v2_first.log` | `91fadc5e043d68879fd0533222a2f9b0ea94e70a393e81f71b1914e2a2096d82` |
| `$RUN/alignment_audit.json`（修复后正式版） | `2002a77e72809677da74151534e3f8b44f51036b258952a06efd83a0048f886d` |

## 7. 合规性

- 本修复发生在 `freeze_parameters.json` 生成**之后**、附件4冻结推理**之前**，
  未使用附件4的任何标签（附件4本身无标签）或预测结果反向调整参数。
- 附件4在此之前仅用于第 5 步的数据转换与 tokenizer offsets 预检；
  解释参数 `W=3` 的冻结依据完全来自验证集三尺度实验。
- 未修改 `ds/` 与 `优化/q2yhv2_1/`，未重新训练问题二模型，仅使用 `cuda:2`。