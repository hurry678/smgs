"""Render Q1 required tables and evidence directly from the current artifacts."""
import csv
import json

import numpy as np

from prepare import ROOT


def code(value):
    return "`" + str(value).replace("|", "\\|").replace("\n", " ") + "`"


def number(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def excerpt(value, limit=180):
    value = " ".join(value.split())
    if not value:
        return "（空输出）"
    return code(value[:limit] + ("…" if len(value) > limit else ""))


def main():
    rows = list(csv.DictReader((ROOT / "outputs/q1_summary.csv").open(encoding="utf-8-sig")))
    metas = {r["sample_id"]: r for r in map(json.loads,
             (ROOT / "outputs/alignment.jsonl").read_text().splitlines())}
    audit = json.loads((ROOT / "reports/boundary_audit.json").read_text())
    asr = {r["sample_id"]: r for r in audit["samples"]}
    validation = json.loads((ROOT / "reports/validation.json").read_text())
    assert len(rows) == len(metas) == 100 and validation["strict_release_passed"]
    speech_phrases = sum(len(m["phrases"]) for m in metas.values() if m["speech_alignment_available"])
    unmatched = [r for r in audit["samples"]
                 if r["status"] == "decoded_without_transcript_prompt" and r["exact_matched_words"] == 0]
    body = [
        "## 典型样本与全部100条结果",
        "",
        "本节全部数值由当前CSV、JSONL和NPZ生成。样本ID、序列行、字符范围、声学窗和视频帧之间可双向核验。T/A/V分别表示文本、语音、视觉；每个序列项的维度均为128/16/6，共150维。有效长度P是实际序列项数，文件中不填充；后续组成batch时另设位置掩码。",
        "",
        f"额外核验对{audit['decoded_samples']}条非数字静音音轨执行无原文提示的Whisper base.en识别，另{audit['digital_silence_skipped']}条数字静音明确跳过。{speech_phrases}个MFA短语中，{audit['matched_phrase_count']}个原词全部精确匹配且ASR时间为正长，得到{audit['comparable_endpoint_count']}个可比端点；分歧中位数{number(audit['absolute_difference_s']['median'])}秒，P90为{number(audit['absolute_difference_s']['p90'])}秒，最大{number(audit['absolute_difference_s']['maximum'])}秒。识别文本相对题给文本的词编辑比为{audit['transcript_word_error_ratio_decoded_samples']:.3%}。这些是词匹配子集上的模型差异，不能外推为全体边界误差；人工真值边界数仍为0。",
        "",
        f"仍有{speech_phrases - audit['matched_phrase_count']}个MFA短语未满足独立比较条件，{len(unmatched)}条非数字静音样本无原词精确匹配；部分识别输出还出现重复句。可能原因包括识别失败、非语音以及题给转写与音轨不一致，单靠该模型不能归因。MFA时间应理解为在题给转写条件下的估计；结构验收通过不消除其语义对应不确定性。原文、源素材和所有未匹配结果完整保留。",
        "",
        "[完整核验设计、异常案例及来源说明](边界独立核验与修正说明.md)；全部逐词原始识别随包保留。",
        "",
        "### 典型样本：三模态均有观测的既定短先导",
        "",
    ]
    sid = "-wny0OAz3g8$_$0"
    meta = metas[sid]
    body += [
        f"样本{code(sid)}，实测时长{meta['duration_s']:.6f}秒。原始转写为：",
        "", "> " + meta["text"], "",
        "下表列出每个短语的原文、实际音频区间、声学窗索引及参与视觉均值的原帧范围。帧区间是最早至最晚采样帧，并非期间所有帧均被聚合；精确列表保留在JSONL。区间均左闭右开，原帧序号从0开始。",
        "",
        "| 行 | 原文字符范围与片段 | 音频区间/s | 25ms声学窗索引 | 采样原帧首—末 | 原始PTS首—末/s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for p in meta["phrases"]:
        frames, pts, windows = p["frame_indices"], p["frame_pts_s"], p["audio_window_indices"]
        body.append(
            f"| P{p['phrase_index']} | [{p['char_start']},{p['char_end']}) {code(p['text'])} | "
            f"[{p['start_s']:.3f},{p['end_s']:.3f}) | "
            f"{windows[0]}—{windows[-1]} | {frames[0]}—{frames[-1]} | {pts[0]:.6f}—{pts[-1]:.6f} |")
    body += [
        "", "三模态代表值如下。BERT第0维仅用于展示数值对应，不赋予单维情绪含义；全部128维语义、16维声学和6维视觉数值见同一NPZ行。掩码列为各模态有效维数。",
        "", "| 行 | 子词索引 | BERT[0] | MFCC[0] | log-RMS | ZCR | F0/Hz | 嘴开合 | 有效维T/A/V |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    with np.load(ROOT / meta["feature_file"], allow_pickle=False) as z:
        for p, x, valid in zip(meta["phrases"], z["features"], z["valid"]):
            body.append(
                f"| P{p['phrase_index']} | {p['token_indices'][0]}—{p['token_indices'][-1]} | "
                + " | ".join(f"{float(x[k]):.5f}" if valid[k] else "缺失" for k in (0, 128, 141, 142, 143, 146))
                + f" | {valid[:128].sum()}/{valid[128:144].sum()}/{valid[144:].sum()} |")
    body += [
        "", f"![真实波形、基频、嘴开合与短语](../outputs/figures/{sid}_alignment.png)",
        "", f"![对应原视频帧及原文](../outputs/figures/{sid}_frames.png)",
        "", f"[播放该样本的源音轨副本](../outputs/audit_audio/{sid}.m4a)。该音频只供回看，没有被登记为人工标注。",
        "", f"无原文提示的Whisper base.en识别结果：{code(asr[sid]['asr_text'])}。",
        "", "| 行 | MFA区间/s | 独立ASR同词片段/s | 两端点分歧/s |",
        "|---:|---|---|---|",
    ]
    for p in asr[sid]["phrases"]:
        body.append(
            f"| P{p['phrase_index']} | {code([round(x, 3) for x in p['primary_s']])} | "
            f"{code([round(x, 3) for x in p['asr_s']]) if p['asr_s'] else '词未全部匹配，不比较'} | "
            f"{code([round(x, 3) for x in p['absolute_difference_s']]) if p['asr_s'] else '—'} |")
    body += [
        "", "这一展示验证三模态记录的对应关系，并提供另一模型的语音证据；模型之间接近或分歧都不能直接当作人工测得的真实边界误差。",
        "",
        "中等时长先导样本`-HwX2H8Z4hY$_$9`则在全部采样帧中未获得可靠人脸关键点，三个短语的视觉6维均缺失。原帧可见人物出现在画面内的小电视屏幕中；这是检测可用性限制，不等于物理上无人。下图保留其真实帧及相同的原文/时间对应，特征以0加假掩码记录。",
        "",
        "![视觉未检出的既定中等先导](../outputs/figures/-HwX2H8Z4hY$_$9_frames.png)",
        "", "### 全量结果表",
        "",
        "全部原片段均完整保留、未截断，表中原始有效时长取原容器实测时长，保留六位小数；它描述可回溯的素材时间范围，不代表整个范围均有语音或人脸观测。VAD有效语音区另存JSONL。维度T/A/V为每行维数；缺失维仍占据对应列，靠掩码区分。异常栏记录整段上下文、第二端点缺失、自动合并、视觉缺失和主体歧义；完整细项见JSONL。",
        "",
        "“ASR可比/P”是满足原词全部匹配且区间正长的短语数/全部MFA短语数；整段上下文为“—”。该列记录独立证据覆盖，不表示人工准确率。",
        "",
        "| # | 样本ID | 模态 | 实测时长/s | 维度T/A/V | P | 对齐粒度 | 一致性等级 | ASR可比/P | 观测/处理状态 |",
        "|---:|---|---|---:|---|---:|---|---|---|---|",
    ]
    for i, row in enumerate(rows, 1):
        meta = metas[row["sample_id"]]
        notes = []
        if not meta["audio_observed"]:
            notes.append("音频缺失")
        elif not meta["speech_alignment_available"]:
            notes.append("语音时间不可用")
        if meta["automatic_boundary_evidence"].get("secondary_unavailable_phrase_indices"):
            notes.append("第二端点缺失")
        if any("automatic_quality_merge" in p["flags"] for p in meta["phrases"]):
            notes.append("自动合并")
        states = {p["visual_observation_state"] for p in meta["phrases"]}
        if "observed_with_subject_ambiguity" in states:
            notes.append("主体歧义")
        if states - {"observed", "observed_with_subject_ambiguity"}:
            all_missing = not states & {"observed", "observed_with_subject_ambiguity"}
            notes.append("全部视觉缺失" if all_missing else "部分视觉缺失")
        matched_phrases = sum(p["asr_s"] is not None for p in asr[row["sample_id"]]["phrases"])
        asr_coverage = f"{matched_phrases}/{len(meta['phrases'])}" if meta["speech_alignment_available"] else "—"
        body.append(
            f"| {i} | {code(row['sample_id'])} | T/A/V | {float(row['duration_s']):.6f} | "
            f"128/16/6 | {row['phrases']} | {row['granularity']} | "
            f"{row['automatic_confidence_grade']} | {asr_coverage} | {'；'.join(notes) or '已生成'} |")
    body += ["", "全部100行均通过当前结构验证和自动验收。等级与有效掩码描述不同的对象：等级反映对齐器一致性，有效掩码描述对应特征是否有可用观测。"]
    (ROOT / "reports/典型样本与全量结果.md").write_text("\n".join(body) + "\n")
    evidence = validation["automatic_boundary_evaluation"]
    report = [
        "# 边界独立核验与修正说明", "",
        "## 核验设计与统计口径", "",
        "对附件1全部100条按同一规则处理：98条非数字静音音轨输入OpenAI Whisper base.en，自由识别时不输入题给转写、关键词或MFA时间；2条数字静音仅保留明确例外状态。该模型与stable-ts使用的tiny.en权重不同，未根据情感标签调整或训练。固定解码温度0、beam size 5、英文、word_timestamps=True、无前句提示；版本和官方摘要见audit_model_source.json。",
        "",
        "识别完成后，按忽略大小写、保留英文缩合词的单词编辑距离将识别词与题给原文单调匹配；仅短语全部原词精确匹配且ASR时段为正长时比较端点。所有未匹配短语保留并计入覆盖分母，不以模型输出回填原文或人工表。模型时间是额外算法证据，不是人工参考；匹配子集的分歧不能外推至剩余短语。",
        "",
        "| 指标 | 实测值 |", "|---|---:|",
        f"| 音频自由识别/数字静音跳过 | {audit['decoded_samples']} / {audit['digital_silence_skipped']} |",
        f"| 词全部匹配且ASR区间正长的MFA短语 | {audit['matched_phrase_count']} / {speech_phrases} |",
        f"| 可比较端点 | {audit['comparable_endpoint_count']} |",
        f"| MFA与独立ASR端点分歧中位数/s | {number(audit['absolute_difference_s']['median'])} |",
        f"| MFA与独立ASR端点分歧P90/s | {number(audit['absolute_difference_s']['p90'])} |",
        f"| MFA与独立ASR最大端点分歧/s | {number(audit['absolute_difference_s']['maximum'])} |",
        f"| 98条识别文本相对题给文本的词编辑比 | {audit['transcript_word_error_ratio_decoded_samples']:.3%} |",
        f"| ASR核验循环实测时间/s | {audit['wall_seconds']:.3f} |",
        "| 人工真值边界 | 0 |",
        "",
        "词编辑比同时受识别错误、非语音和题给转写差异影响，不用于断言源转写错误，也不是情感模型评价指标。具体原因需要听辨原音轨；程序没有自动填写人工审核者或参考边界。上述核验循环时间不含首次模型下载和最后导出回放音轨。",
        "",
        "## 第二对齐器退化区间修正", "",
        f"当前stable-ts有{evidence['secondary_unavailable_phrases']}个短语映射区间缺失或非正长，涉及{evidence['secondary_incomplete_samples']}条样本。这些短语明确标记secondary_status=missing_or_nonpositive_span，两个端点均不计入有效覆盖或分歧。剩余{evidence['cross_aligner_endpoint_count']}端点的分歧中位数为{evidence['cross_aligner_absolute_difference_s']['median']:.3f}秒，P90为{evidence['cross_aligner_absolute_difference_s']['p90']:.3f}秒。",
        "",
        "MFA负责主时间估计，stable-ts和VAD作为审计证据。发布规则分别记录主边界质量、第二端点有效覆盖及每短语证据是否有明确状态；缺失第二端点使一致性等级为C，不能将其宣传为第二模型已确认。该规则保留可用的MFA结果，不凭一项辅助证据的失败删除原样本。",
        "",
        "## 预定异常范围的逐条核验", "",
        "范围为三条既定先导、stable-ts最大分歧前三条、VAD为空的全部样本、发生自动质量合并的全部样本。全部100条已识别，这里只展开上述可解释的案例；所列范围不影响总体统计。",
        "",
        "| 样本ID | 分层原因 | 精确匹配词/题给词 | 自由识别文本（最多180字符） |", "|---|---|---:|---|",
    ]
    pilots = set(json.loads((ROOT / "configs/pilot.json").read_text()))
    high = {m["sample_id"] for m in sorted(metas.values(), key=lambda m: (
        m["automatic_boundary_evidence"]["maximum_absolute_difference_s"] or 0), reverse=True)[:3]}
    for sid, meta in metas.items():
        reasons = []
        if sid in pilots:
            reasons.append("既定先导")
        if sid in high:
            reasons.append("高分歧")
        if not meta["automatic_boundary_evidence"]["vad_intervals_s"]:
            reasons.append("VAD空")
        if any("automatic_quality_merge" in p["flags"] for p in meta["phrases"]):
            reasons.append("自动合并")
        if reasons:
            a = asr[sid]
            transcript = ("数字静音，未执行识别" if a["status"] == "digital_silence_not_decoded"
                          else excerpt(a["asr_text"]))
            report.append(f"| {code(sid)} | {'、'.join(reasons)} | {a['exact_matched_words']}/{a['reference_word_count']} | {transcript} |")
    report += [
        "", "## 核验后发现的词不匹配与识别退化", "",
        f"以下{len(unmatched)}条非数字静音样本未匹配到任何原词。其中MFA已失败的样本保持整段上下文，其余保留题给转写条件下的MFA估计及其一致性/缺失证据；不能把强制对齐成功当作原文与音轨语义相符。此表是全量识别后的异常描述，不是预先确定的统计子集。表内文本最多展示180字符，完整文本在JSON中。",
        "",
        "| 样本ID | 题给原文 | 无提示识别文本 |", "|---|---|---|",
    ]
    for a in unmatched:
        report.append(f"| {code(a['sample_id'])} | {excerpt(a['text'])} | {excerpt(a['asr_text'])} |")
    pilot_matches = "；".join(
        f"{code(sid)}为{asr[sid]['exact_matched_words']}/{asr[sid]['reference_word_count']}"
        for sid in json.loads((ROOT / "configs/pilot.json").read_text()))
    report += [
        "",
        "例如`-NFrJFQijFE$_$2`的题给原文谈技术领域协调，识别输出涉及动物感官；`-iRBcNs9oI8$_$8`的题给原文谈肌肉与力量，识别输出涉及邀请他人。它们提示转写/音轨对应关系需要听辨核实；当前证据不足以决定修改哪一方，因此不自动改写或替换输入。`-yRb-Jum7EQ$_$1/5/6`与`-tPCytz4rww$_$11`还出现重复语句，说明ASR自身也会退化。",
        "",
        f"三条先导的精确匹配词/题给词：{pilot_matches}。高分歧样本既包括零词匹配，也包括ASR重复，故不能只根据某一模型或某个总体分位数判断真实边界。词编辑比允许大于100%（插入词超过参考长度），全部样本保留在总体统计中。",
        "", "原音轨回放文件及源摘要见outputs/audit_audio/manifest.json；逐词识别、概率、时间戳和解码参数见intermediate/boundary_audit/。完整比较见reports/boundary_audit.json。",
        "",
        "## 可复现记录与资源基准", "",
        "MFA新运行记录包含唯一run_id、输入/模型/执行代码指纹、逐次命令与返回码、日志摘要、恢复样本ID以及输出SHA-256。last_run.json包含音频准备、MFA调用和全部样本提取的墙钟时间，并列明缓存命中与失败；不把逐样本耗时之和称为全流程耗时。",
        "",
        "resource_lock.json保存本次审计冻结的预期SHA-256；已有文件与新下载文件必须先比对预期值，匹配后才写本次实际来源清单。download不会修改基准。Whisper摘要来自官方模型URL；其他基准是已审计官方来源文件的复现锁定值，不声称有发布者数字签名。文档参考页若上游发生变化，会因摘要不符拒绝覆盖，需要显式审核版本变更。",
    ]
    (ROOT / "reports/边界独立核验与修正说明.md").write_text("\n".join(report) + "\n")
    target = ROOT / "reports/第一问解题报告.md"
    text = target.read_text()
    start, end = "<!-- GENERATED_Q1_RESULTS_START -->", "<!-- GENERATED_Q1_RESULTS_END -->"
    embedded = "\n".join(body).replace(
        "## 典型样本与全部100条结果", "### 5.1 独立模型核验").replace(
        "### 典型样本：三模态均有观测的既定短先导",
        "### 5.2 典型样本：三模态均有观测的既定短先导").replace(
        "### 全量结果表", "### 5.3 全量结果表")
    generated = start + "\n\n" + embedded + "\n\n" + end
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + generated + text[b:]
    else:
        text = text.replace("## 6. 验证边界与限制", generated + "\n\n## 6. 验证边界与限制")
    target.write_text(text)
    print("Rendered current 100-row table, three-modality example and independent-model audit.")


if __name__ == "__main__":
    main()
