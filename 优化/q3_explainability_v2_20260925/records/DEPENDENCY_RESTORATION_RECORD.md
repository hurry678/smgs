# 问题三 v2 —— 恢复协议指定依赖与 MFA 资源（隔离安装记录）

- 执行日期：2026-09-26（CST）
- 服务器：`hdu-Super-Server`（Ubuntu 22.04，glibc 2.35）
- 运行目录：`/data2/hy/q3_runs/q3_explainability_v2_20260925`（记作 `$RUN`）
- 授权范围：**恢复协议指定依赖**，不构成模型或实验协议变更
- 隔离原则：**不升级、不污染共享环境 `secgpt-vllm`**

---

## 1. Python 依赖（隔离安装，经 `PYTHONPATH` 使用）

安装方式：`pip install --no-index --no-deps --target "$RUN/python_deps"`
使用方式：`PYTHONPATH="$RUN/python_deps" $PY ...`（`$PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python`，Python 3.10.20）

| 包 | 版本 | 官方下载 URL | SHA-256 | 说明 |
|---|---|---|---|---|
| `huggingface-hub` | 0.36.2 | `https://files.pythonhosted.org/packages/a8/af/48ac8483240de756d2438c380746e7130d1c6f75802ef22f3c6d49982787/huggingface_hub-0.36.2-py3-none-any.whl` | `48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270` | 满足协议 `>=0.23,<1.0` |
| `fsspec` | 2026.9.0 | `https://files.pythonhosted.org/packages/6c/c0/a98505f18594f1bce828bb159cec0fcf9860562f1a2c85913409fc8f3d9e/fsspec-2026.9.0-py3-none-any.whl` | `8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f` | `huggingface-hub` 的硬依赖（共享环境缺失） |
| `regex` | 2024.11.6 | `https://files.pythonhosted.org/packages/f2/98/26d3830875b53071f1f0ae6d547f1d98e964dd29ad35cbf94439120bb67a/regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `997d6a487ff00807ba810e0f8332c18b4eb8d29463cfb7c820dc4b6e7562d0cf` | `transformers==4.41.2` 的硬依赖（共享环境缺失） |
| `safetensors` | 0.4.5 | `https://files.pythonhosted.org/packages/b9/df/6f766b56690709d22e83836e4067a1109a7d84ea152a6deb5692743a2805/safetensors-0.4.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `c6d156bdb26732feada84f9388a9f135528c1ef5b05fae153da365ad4319c4c5` | `transformers` 要求的 `>=0.4.1`（共享环境缺失） |

SHA 交叉核验：上表 SHA 与 PyPI JSON API（`https://pypi.org/pypi/<pkg>/<version>/json`）公布的 `digests.sha256` **逐项一致**；服务器端 `sha256sum` 复核亦一致。

安装日志（服务器）：
- `$RUN/python_deps/install_logs/pip_download_hub.log`
- `$RUN/python_deps/install_logs/pip_download_fsspec.log`
- `$RUN/python_deps/install_logs/pip_download_regex_safetensors.log`
- `$RUN/python_deps/install_logs/pip_install.log`
- `$RUN/python_deps/install_logs/pip_install_root.log`
- `$RUN/python_deps/install_logs/pip_install_regex_safetensors.log`

导入验证（带 `PYTHONPATH`）：

```
hub 0.36.2 / fsspec 2026.9.0 / regex(metadata) 2024.11.6 / safetensors 0.4.5
transformers 4.41.2 / torch 2.3.0+cu121 (cuda True, 4 devices)
```

---

## 2. MFA 3.4.1（conda-forge，离线隔离安装）

- 来源通道：`conda-forge`（官方渠道）
- 包集合：**281** 个归档（273 `.conda` + 8 `.tar.bz2`，合计 441.8 MB），平台 `linux-64` / `noarch`
- 精确清单（含 md5 锁定）：`$RUN/.mfa_pkgs/explicit_specs.txt`
- 安装前缀：`$RUN/.mfa`（独立 conda 前缀，未写入任何共享环境）
- 安装命令：`conda create -p "$RUN/.mfa" --offline --yes --file "$RUN/.mfa_pkgs/explicit_specs.txt"`
- 安装日志：`$RUN/logs/mfa_install.log`
- 版本验证：`$RUN/.mfa/bin/mfa version` → **3.4.1**

> 说明：初次使用 `--file <绝对路径清单>` 时，conda 的 MatchSpec 解析器无法处理 `x264-1!164.3095` 中的 epoch 分隔符 `!`；
> 改用 conda 官方 explicit 清单格式（`@EXPLICIT` + `file://` + `#md5`）后安装成功。该调整仅涉及安装机制，不改变包集合。

---

## 3. MFA 模型资源（官方现行渠道：GitHub Releases）

**渠道变更说明**：协议原文给出的 `https://mfa-models.s3.amazonaws.com/...` 三个地址当前均返回
`HTTP 404 NoSuchBucket`（该 S3 桶已不存在）。MFA 3.4.1 源码中 `ModelManager.base_url` 已指向
`https://api.github.com/repos/MontrealCorpusTools/mfa-models/releases`，即官方现行渠道为
**GitHub Releases（`MontrealCorpusTools/mfa-models`）**。因此按官方现行渠道获取，并**逐项核对协议 SHA**。

| 协议目标文件 | Release tag | 官方资产名 | 下载 URL | 大小 (B) | 协议 SHA-256 | 实得 SHA-256 | 匹配 |
|---|---|---|---|---|---|---|---|
| `english_mfa_dictionary_v3.1.0.dict` | `dictionary-english_mfa-v3.1.0` | `english_mfa.dict` | `https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_mfa-v3.1.0/english_mfa.dict` | 1,078,195 | `975bf9c7791535c5aec57995e0bd2b77eb7ca364d1d974c2134e07bb0f16b079` | `975bf9c7791535c5aec57995e0bd2b77eb7ca364d1d974c2134e07bb0f16b079` | ✅ |
| `english_mfa_acoustic_v3.1.0.zip` | `acoustic-english_mfa-v3.1.0` | `english_mfa.zip` | `https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip` | 92,170,811 | `2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068` | `2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068` | ✅ |
| `english_us_mfa_g2p_v3.0.0.zip` | `g2p-english_us_mfa-v3.0.0` | `english_us_mfa.zip` | `https://github.com/MontrealCorpusTools/mfa-models/releases/download/g2p-english_us_mfa-v3.0.0/english_us_mfa.zip` | 10,740,410 | `9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02` | `9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02` | ✅ |

- 服务器落盘位置：`$RUN/resources/mfa/`（使用协议要求的版本化文件名）
- 任一不一致即停止；本次三项全部一致，未发生替换或降级。

### 资源就绪校验（`$RUN/logs/mfa_resource_check.json`，status = PASS）

- MFA 版本：3.4.1
- 三文件 SHA-256 逐项匹配协议
- 三模型均可被 MFA 3.4.1 正常加载：
  - `DictionaryModel` → `english_mfa_dictionary_v3.1.0`
  - `AcousticModel` → `english_mfa_acoustic_v3.1.0`（version 3.1.0）
  - `G2PModel` → `english_us_mfa_g2p_v3.0.0`

---

## 4. 隔离性证据（共享环境未被污染）

- 共享环境 `secgpt-vllm/lib/python3.10/site-packages` 中**不存在** `huggingface_hub`、`fsspec`、`regex`、`safetensors` 任何一项
- 不带 `PYTHONPATH` 时：`import huggingface_hub` → `ModuleNotFoundError`；`import fsspec` → `ModuleNotFoundError`
- 共享环境 site-packages 最新修改时间仍为 2026-03-21（本次未写入）
- MFA 安装到独立前缀 `$RUN/.mfa`（2.0 GB），未调用任何 `conda env update` / 全局安装

---

## 5. 合规声明

- 本次仅**恢复协议指定依赖与官方模型资源**，未更换模型、未修改实验协议、未调整任何超参数
- 附件4（`$RAW`）在 `freeze_parameters.json` 生成前**未被读取**
- 最终 `verification.json` 通过前**不推送**任何结果