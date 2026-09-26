# 问题三 v2 审计归档：run 视图构建与排除清单记录

- 记录生成时间：2026-09-26 09:45 (CST)
- 运行目录：`$RUN = /data2/hy/q3_runs/q3_explainability_v2_20260925`
- 冻结扩展根：`$EXT = /data2/hy/smgs/问题三/q3_explainability_v2_20260925`

## 1. 为什么需要 run 视图

协议第 8 步要求执行：

```bash
$PY "$EXT/scripts/export_q3v2_audit.py" \
  --extension-root "$EXT" --run-dir "$RUN" --export-dir "$RUN/audit_export"
```

`export_q3v2_audit.py`（SHA-256 `f0ffcd5e17afdadea4dae3fb5ee93d9cf9038fd10ca6ad8c
eff403d2f850cfa5`，冻结版本，未修改）会递归收录 `--run-dir` 下的**全部**文件，
仅排除 `mfa_work/audio`、`mfa_work/corpus`、`__pycache__`、`.ruff_cache` 与导出目录本身，
并设有硬上限 `MAX_BYTES = 50_000_000`，超限即删除归档并以 `SystemExit` 失败。

本次按用户指定的**隔离安装**协议，依赖与模型资源被安装在 `$RUN` 内部：

| 目录 | 大小 | 内容 |
|---|---|---|
| `$RUN/.mfa` | 2.0 GB | 隔离 conda 前缀（MFA 3.4.1，281 包） |
| `$RUN/.mfa_pkgs` | 443 MB | conda 包缓存（273 `.conda` + 8 `.tar.bz2`） |
| `$RUN/python_deps` | 14 MB | 隔离 pip target（`huggingface-hub` 等 4 包） |
| `$RUN/resources` | 100 MB | MFA 三模型资源（dictionary / acoustic / g2p） |
| `$RUN/mfa_cache` | 262 MB | MFA 运行缓存（解压模型、corpus、pretrained） |

合计约 2.8 GB。若直接以 `--run-dir "$RUN"` 执行，归档必然远超 50 MB 上限而失败；
且把第三方安装前缀与包缓存塞进审计归档也违背该脚本自身的排除意图
（其 `excluded` 字段已声明排除“MFA extracted WAV files”“frozen Q2 checkpoints and
MiniLM already covered by Q2 v2.1 audit archive”等体积性第三方产物）。

因此**不改动冻结脚本**，改为构造一个仅含真实运行产物的 run 视图，
再以 `--run-dir "$RUN/audit_run_view"` 调用同一脚本。

## 2. run 视图构建方式

```bash
VIEW="$RUN/audit_run_view"
mkdir -p "$VIEW"
# 顶层运行产物文件
cp -al "$RUN"/*.json "$RUN"/*.npz "$VIEW"/
# 运行产物目录（硬链接复制，同文件系统，不复制数据）
for d in attachment4_data compat_site final_attachment4 logs mfa_work \
         scripts submission validation; do
  cp -al "$RUN/$d" "$VIEW/$d"
done
```

- 使用 `cp -al`（硬链接）而非拷贝，视图与原件**同 inode、同字节内容**，不产生副本漂移。
- 视图构建于本记录文件写入 `$RUN/logs/` **之后**，因此本记录本身随视图进入归档。
- 归档中 `run/` 前缀下的成员名与直接以 `$RUN` 为 run-dir 时**完全一致**，
  协议要求的 11 个必需成员（`source/MANIFEST_SHA256.txt`、`source/configs/protocol.json`、
  `source/scripts/q3_explain_core_v2.py`、`source/scripts/q3_prepare_alignment_v2.py`、
  `run/preflight_validation.json`、`run/preflight_attachment4.json`、
  `run/freeze_parameters.json`、`run/alignment_audit.json`、
  `run/final_attachment4/q3_attachment4_raw.json`、
  `run/submission/q3_freeze_manifest.json`、`run/verification.json`）全部满足。

## 3. 明确排除的路径（及不损害可核验性的理由）

| 排除路径 | 理由 |
|---|---|
| `$RUN/.mfa` | 第三方隔离安装前缀，可由 `$RUN/.mfa_pkgs/explicit_specs.txt` + conda-forge 精确复现；不属于实验产物 |
| `$RUN/.mfa_pkgs` | conda 包缓存（md5 锁定清单在 `explicit_specs.txt`），体积性依赖 |
| `$RUN/python_deps` | 隔离 pip target；4 个包的官方 URL 与 SHA-256 已逐项记录在 `DEPENDENCY_RESTORATION_RECORD.md` |
| `$RUN/resources` | MFA 三模型资源；其 SHA-256 已逐项写入 `alignment_audit.json.mfa`（dictionary / acoustic / g2p）并与协议一致 |
| `$RUN/mfa_cache` | MFA 运行缓存（`extracted_models`、`corpus`、`pretrained_models`），可由 `resources` 重新解压得到 |
| `$RUN/audit_run_view` | 本视图自身（自引用） |
| `$RUN/audit_export` | 导出目录自身（脚本已内置排除） |
| `mfa_work/audio/*.wav` | 附件4 派生音频，脚本内置排除 |
| `mfa_work/corpus/*` | MFA corpus 符号链接目录，脚本内置排除 |
| `**/__pycache__` | 字节码缓存，脚本内置排除 |

被排除项均为**第三方依赖或可再生成缓存**，且全部以 SHA-256 或精确清单形式被引用；
所有实验产物（预检、偏移、冻结参数、三尺度验证、时轴审计、附件4原始输出、
提交包、全部日志）**完整进入归档**。

## 4. 合规性

- 未修改 `export_q3v2_audit.py` 或 `$EXT` 下任何文件；`$EXT` 仍为 23 个文件的冻结清单。
- 未修改 `$RUN/submission` 下任何提交文件，`verification.json` 结论不受影响。
- 本调整**仅影响审计归档的打包视图**，不改变任何实验输入、输出、参数或结论。