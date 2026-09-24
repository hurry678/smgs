# 问题一特征字典与处理规则

## 总体接口

- 样本单位：附件1中的每个原始视频；样本级 NPZ 保留变长序列，不使用跨样本 padding。
- 词级公共索引：英文转写经 `[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*` 分词；每个词都有 `word_start_sec`、`word_end_sec`、`word_confidence` 和 `word_fallback`。
- 所有模态缺失均由 `*_valid` 或 `word_*_valid` 显式标记，零向量不等于有效观测。
- 弱平滑只在连续且全维有效的帧段内执行，不跨 padding、缺失边界或视频边界。

## 文本模态

- 后端：`precomputed`；BERT 模型：`bert-base-uncased`。
- 维度：768。
- 输出：`text_features`、`text_valid`、`word_tokens`。
- 处理：`bert-base-uncased` 子词 offset 映射到词级均值；无子词映射时使用确定性哈希回退。
- 未做文本改写、标签训练或跨模态监督。

## 语音模态

- 采样率：16000 Hz；窗长：400 点（25 ms）；步长：160 点（10 ms）。
- 维度：40；Mel 滤波器：20；MFCC：13。
- 组成：能量、ZCR、频谱质心、带宽、85% 滚降、平坦度、频谱通量、有声概率、归一化基频、13 维 MFCC、10 维 MFCC 差分、8 维 log-Mel。
- 弱平滑：连续全维有效帧内 `gaussian_filter1d(sigma=0.75)`；mask 不改变。
- 词级聚合：按时间重叠加权平均；无可观测帧时 `word_audio_valid=0`。

## 视觉模态

- 后端：OpenCV Haar `haarcascade_frontalface_default.xml`。
- 采样率：5.0 fps；人脸 ROI：48×48。
- 维度：90。
- 组成：人脸框几何 8 维、Hu 矩 7 维、ROI 3×3 均值/标准差 18 维、灰度直方图 16 维、HOG 2×2×9=36 维、Laplacian 纹理 5 维。
- 检测失败：零向量且 `vision_valid=0`；不推断为中性表情。
- 弱平滑：仅连续检测成功帧内 `gaussian_filter1d(sigma=0.75)`；mask 不改变。
- 词级聚合：词区间内有效采样帧均值；无有效帧时 `word_vision_valid=0`。

## 词级对齐

- 后端：`energy_pause_dp_v1`，20 fps 工作分辨率。
- 方法：语音能量、有声概率、频谱通量估计活动与停顿；词长权重、单调边界、非重叠约束和停顿奖励组成动态规划。
- 边界规则：严格单调、非重叠；词区间限制在有效音频时长内。
- 置信度：活动占比、边界停顿质量和时长适配度的加权组合。
- 边界说明：这是确定性可复现的基线，不宣称等价于神经强制对齐器。

## NPZ 字段

- `text_features`：`[W, 768]`
- `text_valid`：`[W, 768]`
- `word_tokens`：`[W]`
- `audio_features`：`[T, 40]`，`audio_times` 为帧中心秒数
- `audio_valid`：`[T, 40]`
- `vision_features`：`[V, 90]`，`vision_times` 为采样时间秒数
- `vision_valid`：`[V]`
- `word_audio_features`：`[W, 40]`，`word_audio_valid`：`[W]`
- `word_vision_features`：`[W, 90]`，`word_vision_valid`：`[W]`
- `word_start_sec`、`word_end_sec`、`word_confidence`、`word_fallback`：`[W]`
