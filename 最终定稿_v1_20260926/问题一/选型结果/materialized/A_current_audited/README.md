# 第一问：多模态特征提取与时序对齐

固定路线：Google小BERT（128维）＋librosa声学量（16维）＋MediaPipe人脸几何量（6维）。Montreal Forced Aligner（MFA）估计给定文本主边界，stable-ts与Silero VAD提供自动一致性证据；另以Whisper base.en自由识别全部非数字静音音轨作独立模型核验。输入是题给附件1的全部100条视频，不使用情感标签训练提取器。完整结果、异常和验收状态见[解题报告](reports/第一问解题报告.md)。

当前100条均已形成确定性的最终序列并通过严格发布门：97条为MFA短语级结果，`-NFrJFQijFE$_$1`因MFA失败且VAD无语音区而使用整段`clip_context`，2条数字静音样本使用整段`clip_context`并将16维音频掩码全部置假。三类结果统一存为`P×150`，人工复核不再是自动结果的发布前置条件。

## 1. 查阅入口

| 文件/目录 | 内容 |
|---|---|
| [实施计划](实施计划.md) | 任务分解、依赖、目录、验收标准、固定回退规则 |
| [解题报告](reports/第一问解题报告.md) | 解题路线、数学模型、实现、实测结果、方法评估和限制 |
| [特征字典](reports/特征字典.md) | 每一维的定义、单位、时间坐标及掩码约定 |
| [100条汇总](outputs/q1_summary.csv) | 每条样本的时长、维度、短语数、覆盖率、异常 |
| `outputs/features/` | 每条样本一个压缩NPZ，使用`allow_pickle=False`读取 |
| `outputs/alignment.jsonl` | 每条样本的完整原文、词/短语时间、子词和原帧映射 |
| `outputs/figures/` | 真实波形、基频、几何特征时间图及原帧示例 |
| [自动验证结果](reports/validation.json) | 全量覆盖、数值、掩码、PTS和独立聚合重算结果 |
| [逐样本验收](outputs/acceptance.jsonl) | 当前输出的正式验收状态与NPZ摘要；验证、打包共用 |
| [典型样本与全量表](reports/典型样本与全量结果.md) | 100条正文表、原文/音频/原帧/特征一一对应；同样嵌入解题报告 |
| [独立模型核验](reports/边界独立核验与修正说明.md) | 全量自由识别、异常案例与第二端点退化处理 |
| [音频缺失证据](reports/audio_missing_evidence.json) | 两条数字静音原音轨的逐声道解码与源文件摘要 |
| [自动边界协议与最终结果](reports/自动边界协议与100条最终结果_2026-09-24.md) | 统一协议、三类分支、100条验收统计 |
| `annotations/` | 可选人工审计与修订接口 |
| `resources/model_sources.json` | 模型来源、固定版本、许可证和文件摘要 |
| `resources/resource_lock.json` | 独立保存的预期SHA-256；download只能校验，不能自动重写 |
| `intermediate/mfa/run.json` | 本次批量及逐条恢复命令、返回码、样本ID、日志摘要和输出摘要 |
| `reports/last_run.json` | 包含音频准备及MFA的提取墙钟时间、逐样本结果和运行指纹 |
| `outputs/audit_audio/` | 既定先导、高分歧、VAD冲突及自动合并样本的真实音轨副本 |
| `logs/events.jsonl` | 本地成功/失败事件、耗时和错误堆栈；诊断包不含该原始日志 |

目录中原视频、MFA环境、提取模型和中间缓存用于本地复现，不应整个目录压缩提交。正式提交入口为`submission/第一问交付包.zip`。包体实测见`reports/package_check.json`，且第一问包体通过不等于三问总附件通过50MB。

## 2. 当前工作区直接运行

以下命令在项目根目录执行，路径中有中文，必须保留引号。

```bash
# 全量提取：已有相同配置输出会跳过
"第一问/.venv/bin/python" "第一问/run.py" extract

# 单独生成/检查MFA主边界
"第一问/.venv/bin/python" "第一问/run.py" align-mfa

# 人工边界修改后，核验原生缓存并重新聚合全部样本
"第一问/.venv/bin/python" "第一问/run.py" reaggregate

# 系统数据审计：检查4个附件；统计结果保存在reports/data_quality/
"第一问/.venv/bin/python" "第一问/src/dataset_audit.py"

# 严格验证与正式打包
"第一问/.venv/bin/python" "第一问/run.py" audit-boundaries
"第一问/.venv/bin/python" "第一问/run.py" validate
"第一问/.venv/bin/python" "第一问/run.py" summarize
"第一问/.venv/bin/python" "第一问/run.py" report
"第一问/.venv/bin/python" "第一问/run.py" package

# 三条先导样本及其验证
"第一问/.venv/bin/python" "第一问/run.py" extract --pilot
"第一问/.venv/bin/python" "第一问/run.py" validate --pilot

# 重新生成图
"第一问/.venv/bin/python" "第一问/run.py" figures
```

程序以脚本位置定位资源，与运行命令时的工作目录无关。参数在`configs/q1.json`中；不要根据情感标签修改提取规则。提取指纹包含转写清单、实际配置对象、源文件、核心源码、模型、依赖锁和人工标注。`reaggregate`另外核对原生数组摘要、提取配置、工具版本及时间坐标，缓存不兼容时拒绝复用，应运行`extract`。`--force`用于明确要求完整重算的场景。先导验证和全量验证均使用同一张主复核表；`pilot_boundary_review.csv`只作导出视图。修改复核表后应重新全量验证，再打包。

## 3. 在新环境复现

要求Python 3.11、系统ffmpeg/ffprobe，以及题给原视频目录。macOS安装`ffmpeg`可使用`brew install ffmpeg`；其他平台从FFmpeg官方发行渠道安装。MediaPipe wheel的平台支持需按实际环境核对，当前已验证的是macOS arm64、Python 3.11.16、CPU。

在解压后的第一问目录执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt

# MFA使用独立conda环境；micromamba也可替换为conda
micromamba create -y -p "$PWD/.mfa" -c conda-forge montreal-forced-aligner=3.4.1

# input-dir指向包含label-100.xlsx及37个视频子目录的原始目录
.venv/bin/python run.py prepare --input-dir "/实际路径/MOSEI数据集部分原始视频-100条"

# 下载按来源清单固定的提取资源，保存在resources/models/
.venv/bin/python run.py download

# 完整重算，避免复用包内已有结果
.venv/bin/python run.py extract --force
.venv/bin/python run.py audit-boundaries
.venv/bin/python run.py validate
.venv/bin/python run.py summarize
.venv/bin/python run.py figures
.venv/bin/python run.py report
.venv/bin/python run.py package
```

新环境的`prepare`会复制原样本到`resources/input/`并检查源/副本摘要，保留原标签表。提交包不包含原视频、Whisper权重、Face Landmarker权重和环境，因此全流程重算需要题给素材及下载通道，不能声称全流程离线复现。包内的小BERT可供后续整题共用，最终合包时应去重。

所有资源先与`resource_lock.json`比较，符合预期后才写本次实际来源清单。发现已有文件损坏时程序拒绝登记，用户明确移走损坏文件后可重新下载；有意升级模型须显式审核并修改版本与预期摘要。参考文档若上游内容变化也会拒绝静默覆盖。`audit-boundaries`首次运行另下载官方Whisper base.en（仅审计、不进入正式150维特征）；其权重不随包分发，逐词识别结果随包保存。

`check-text`是与第二问接口一致性的诊断入口，另需项目同级`E题数据/附件2-数据集特征文件/aligned_50.pkl`。它只核对train/valid词元与原文，不训练情感模型，不评价test；不是读取第一问已生成特征的必要步骤。

## 4. 读取特征

```python
from pathlib import Path
import json
import numpy as np

root = Path("第一问")  # 在解压目录内部运行时改为Path(".")
records = [
    json.loads(line)
    for line in (root / "outputs/alignment.jsonl").read_text().splitlines()
]
sample = records[0]
with np.load(root / sample["feature_file"], allow_pickle=False) as data:
    features = data["features"]     # P × 150
    valid = data["valid"]           # P × 150，真表示对应维有效
    intervals = data["intervals"]   # P × 2，原片段相对时间
    text = features[:, :128]
    audio = features[:, 128:144]
    vision = features[:, 144:150]
    print(sample["sample_id"], features.shape, intervals[0])
    print(sample["phrases"][0]["text"])
    print(sample["phrases"][0]["frame_indices"])
```

`sample_id`含`$_$`，在shell中须使用单引号，防止变量替换：

```bash
"第一问/.venv/bin/python" "第一问/run.py" extract '--sample=-wny0OAz3g8$_$0' --force
```

缺失数值置0，并保存独立掩码；不得将视觉0值解释为中性表情，也不能把无声处的0Hz视为真实基频。时间覆盖率和置信度都不是边界准确率。

当前版本为`q1-2.1`，验收政策为`q1-acceptance-4`。全量验证结果见`reports/validation.json`，回归测试结果见`logs/tests_q1_2_1.log`。验证器从保留的原生数组独立重建全部150维；逐短语`visual_observation_state`区分采样未覆盖、检出失败、几何拒绝及主体歧义。

## 5. 自动协议与可选人工修订

自动协议固定为：

1. 非数字静音样本先执行MFA 3.4.1；词级字符必须无损映射回题给原文。
2. 按标点、停顿、词数和时长分组；极短或异常词速的小组自动并入时间上最近的相邻组，并标记`automatic_quality_merge`。
3. stable-ts给出同字符范围的第二组端点，Silero VAD给出语音区；非正长或缺失的第二区间不计入覆盖和分歧，明确记为证据缺失并标C级。发布门核对主边界和每短语证据状态，一致性等级不作为人工真值。
4. MFA不可用时输出整段`clip_context`。数字静音令音频16维值为0且掩码全假；非静音但无可靠语音边界时保留整段可观测音频聚合。

`annotations/boundary_review.csv`保留为可选审计接口，不是自动结果的发布门。若实际导入`manual_phrases.jsonl`修改边界，则该样本必须在复核表中明确批准后才能再次发布；自动结果不会冒充人工参考。

需要修改短语时间时，按[特征字典](reports/特征字典.md)新增`annotations/manual_phrases.jsonl`，运行`reaggregate`和全量诊断验证，再明确批准新输出。导入修订不自动批准，也不生成自我评价参考。样本原文、时间、来源或NPZ变化后，旧批准变为`stale`，已填值保留供核对，旧行另存`.history.jsonl`。

`reference_source=correction`用于修订流程；`independent`用于独立评价，另需`reference_id`说明依据，且人工修订者不能同时充当该修订的独立评价者。误差统计仅使用独立参考，修订与当前输出之差另外报告。主表和输出变化后，旧验证会被打包器拒绝。

人工标注不允许改变原转写、标签或样本数。程序先检查所有非空白字符完整覆盖、真实区间合法且有序，再按新短语区间聚合三模态。没有人工参考的边界误差必须保持未知，不能用自动输出自评出“0秒误差”。

当前状态是100条最终序列均自动接受，其中97条有词/短语级语音对齐，3条为显式`clip_context`。这满足统一模型的100条输出要求；它不等于声称3条`clip_context`具有不可观测的逐词语音边界。人工真值边界仍为0，因此跨对齐器分歧不能写成“准确率”或“真实误差”。
