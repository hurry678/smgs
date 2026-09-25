"""Build auditable statistics figures and a Chinese report from persisted audit outputs."""
import csv
import json
import numpy as np
from prepare import ROOT, dump, sha256
from dataset_audit import OUT, write_csv


def load_rows(name):
    with (OUT/name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(x):
    return float(x) if x not in (None, "", "None") else np.nan


def format_number(x, digits=3):
    return f"{x:.{digits}f}" if x is not None and np.isfinite(float(x)) else "—"


def figures(samples, q1, q2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","text.parse_math":False})
    colors=["#31567C","#459F99","#DF9554"]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    durations=np.array([number(r["duration_s"]) for r in samples])
    labels=np.array([number(r["label"]) for r in samples])
    axes[0,0].hist(durations,bins=np.arange(0,32,2),color=colors[0],edgecolor="white")
    axes[0,0].set(title="Clip duration (100 clips)",xlabel="Seconds",ylabel="Clips")
    axes[0,1].bar(["Negative","Neutral","Positive"],[sum(labels<0),sum(labels==0),sum(labels>0)],color=colors)
    axes[0,1].set(title="Sentiment labels (descriptive only)",ylabel="Clips")
    axes[1,0].scatter(durations,[number(r["text_words"]) for r in samples],c=colors[1],alpha=.75)
    axes[1,0].set(title="Duration and transcript length",xlabel="Duration (s)",ylabel="Whitespace word count")
    states=q1["visual_phrase_states"]
    keys=["observed","observed_with_subject_ambiguity","no_reliable_face_detected",
          "sampling_no_coverage","detected_geometry_rejected"]
    values=[states.get(k,0) for k in keys]
    axes[1,1].barh(["Observed","Observed, subject uncertain","No reliable face detected","No sampled frame","Geometry rejected"],
                  values,color=[colors[0],colors[2],colors[2],colors[2],colors[2]])
    for i,v in enumerate(values):axes[1,1].text(v+2,i,str(v),va="center",fontsize=9)
    axes[1,1].set(title="Visual observations in 345 phrases",xlabel="Phrases",xlim=(0,max(values)*1.13))
    axes[1,1].invert_yaxis()
    fig.suptitle("Q1 data quality and distributions | 2026-09-24",fontsize=16)
    for ext in ("svg","png"):fig.savefig(OUT/f"q1_distributions.{ext}",dpi=150)
    plt.close(fig)
    names=["duration_s","text_words","phrase_count","phrase_time_coverage","face_valid_rate","voiced_rate",
           "f0_mean_hz","native_log_rms_mean","native_mouth_open_mean","label"]
    labels=["Duration","Words","Phrases","Time coverage","Face validity","Voicing","Pitch","Log RMS","Mouth opening","Label"]
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    correlations=load_rows("q1_correlations.csv")
    for ax,method in zip(axes,["pearson_r","spearman_rho"]):
        matrix=np.eye(len(names))
        for row in correlations:
            if row["x"] in names and row["y"] in names:
                a,b=names.index(row["x"]),names.index(row["y"])
                matrix[a,b]=matrix[b,a]=number(row[method])
        image=ax.imshow(matrix,vmin=-1,vmax=1,cmap="RdBu_r")
        ax.set_xticks(range(len(names)),labels,rotation=50,ha="right",fontsize=9)
        ax.set_yticks(range(len(names)),labels,fontsize=9)
        for a in range(len(names)):
            for b in range(len(names)):
                ax.text(b,a,f"{matrix[a,b]:.2f}",ha="center",va="center",fontsize=7,
                        color="white" if abs(matrix[a,b])>.7 else "#202020")
        ax.set_title("Pearson r" if method=="pearson_r" else "Spearman rho")
    fig.colorbar(image,ax=axes,shrink=.8)
    fig.suptitle("Clip-level associations | pairwise n=94–100 | descriptive, not causal",fontsize=14)
    for ext in ("svg","png"):fig.savefig(OUT/f"q1_correlations.{ext}",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    splits=q2["versions"]["aligned_50"]["splits"]
    x=np.arange(3)
    for j,split in enumerate(("train","valid")):
        s=splits[split]
        axes[0].bar(x+(j-.5)*.34,[int(s["class_counts"][str(k)])/s["n"] for k in range(3)],
                    width=.34,label=split,color=colors[j])
    axes[0].set_xticks(x,["Negative","Neutral","Positive"])
    axes[0].set(title="Attachment 2 class proportions",ylabel="Proportion")
    axes[0].legend()
    vals=[splits[s]["modalities"][m]["zero_valid_positions"]/splits[s]["modalities"][m]["valid_positions"]
          for s in ("train","valid") for m in ("audio","vision")]
    axes[1].bar(["Train audio","Train vision","Valid audio","Valid vision"],np.array(vals)*100,
                color=[colors[0],colors[1],colors[0],colors[1]])
    axes[1].set(title="Native zero rows within semantic positions",ylabel="Percent (padding excluded)")
    axes[1].tick_params(axis="x",rotation=25)
    for ext in ("svg","png"):fig.savefig(OUT/f"attachment2_distributions.{ext}",dpi=150)
    plt.close(fig)


def main():
    q1=json.loads((OUT/"q1_audit.json").read_text())
    q2=json.loads((OUT/"q2_audit.json").read_text())
    special=json.loads((OUT/"special_audit.json").read_text())
    validation=json.loads((ROOT/"reports/validation.json").read_text())
    rerun=json.loads((OUT/"q1_rerun_comparison.json").read_text())
    special_rows = special["files"]
    q4_zero = [r for r in special_rows if r["attachment"]=="附件4" and r["version"]=="aligned"
               and r.get("vision_zero_semantic_positions",0)]
    q4_suffix = [r for r in special_rows if r["attachment"]=="附件4" and r["version"]=="unaligned"
                 and r.get("vision_nonzero_after_length",0)]
    samples=load_rows("q1_samples.csv")
    correlations=load_rows("q1_correlations.csv")
    grouped=load_rows("q1_video_group_correlations.csv")
    figures(samples,q1,q2)
    allrows=[]
    for row in samples:
        decision=next(r for r in validation["acceptance"] if r["sample_id"]==row["sample_id"])
        allrows.append({**row,"acceptance_status":decision["status"]})
    write_csv(OUT/"q1_samples.csv",allrows)
    anomalies=[]
    for row in allrows:
        sid=row["sample_id"]
        if row["digital_silence"]=="True":
            anomalies.append({"sample_id":sid,"type":"source_digital_silence","level":"source_exception",
                              "action":"保留原始样本及原生特征，不生成虚假语音时间"})
        if row["alignment_hard_passed"]=="False" and int(row["phrase_count"])>0:
            anomalies.append({"sample_id":sid,"type":"alignment_quality_failed","level":"review_required",
                              "action":"实际听看并回标后重聚合；不以重跑代替人工准确性验证"})
        if number(row["face_valid_rate"])==0:
            anomalies.append({"sample_id":sid,"type":"no_valid_face","level":"observability",
                              "action":"保留视觉缺失掩码；无法由未检出确定物理原因"})
    for sid in q1["duration_outside_stated_range"]:
        anomalies.append({"sample_id":sid,"type":"duration_below_statement_range","level":"documentation",
                          "action":"使用实际容器时长并披露与题面范围的差异"})
    write_csv(OUT/"issues.csv",anomalies)
    lines=[
        "# 数据集系统分析与第一问重新执行报告",
        "",
        "日期：2026-09-24。范围：附件1全部100条原视频及现有第一问产物；附件2两版本各4850条；附件3两版本各30份；附件4两版本各20份及视频配对。原始样本和标签未修改，未训练模型或生成专项预测。",
        "",
        "**结论：文件完整性整体通过，但三模态可观测性和对齐准确性尚不满足最终验收条件；已按要求从原视频重新执行第一问。重跑完成不等于静音得到修复或人工边界验收完成。**",
        "",
        "## 1. 完整性、格式与逻辑检查",
        "",
        "| 检查对象 | 实测结果 | 判断 |",
        "|---|---|---|",
        f"| 附件1编号、视频与标签 | 37个video_id、100条唯一记录、100条视频；缺文件/额外文件/重复ID均为0 | 配对完整 |",
        "| 附件1标签表 | 无空单元格、公式误读、重复整行或重复规范化转写；标签均在[-3,3]且文字极性与符号一致 | 标签结构通过；不单独证明主观标注无噪声 |",
        "| 原件、副本与摘要 | 100条视频原件与工作副本、输入清单摘要一致；标签表副本一致 | 未发现替换或意外改动 |",
        f"| 完整解码与时间 | {q1['decode_checked']}条、{q1['decoded_frame_count']:,}帧可解码；逐帧PTS递增并匹配ffprobe记录 | 无已检出解码损坏或时间轴矛盾 |",
        "| Excel数字表示 | 两份表的连续标签均以文本存储，可转换为有限浮点数；ID保留字符串 | 是接口转换要求，不直接判为脏数据 |",
        "| 附件1历史修复说明 | 说明写6列，当前实际5列；历史64/10/26划分不能对应当前附件2 | 报告采用实际工作表和ID核验 |",
        f"| 附件1与当前附件2重合 | train/valid/test={q2['q1_current_overlap']['train']}/{q2['q1_current_overlap']['valid']}/{q2['q1_current_overlap']['test']}，另{len(q2['q1_ids_absent_from_attachment2'])}条不在当前附件2 | 不能以历史说明替代当前划分 |",
        "| 附件2两版本 | 样本数3395/728/727；同ID顺序、原文、text/text_bert和标签字段一致；矩阵未检出NaN/Inf | 版本身份一致 |",
        "| 附件2划分交叉 | train/valid/test之间同sample_id、同video_id交叉均为0 | 未检出此两类泄漏 |",
        "| 附件3/4文件接口 | 60＋40份全部可读，检查字段数值合法；附件4两版本20对原文/词元和视频文件一致 | 版本配对通过；不等于已人工确认转写对应音轨 |",
        "",
        f"标识符只按题面要求检查可作为ID的字符串和一致性，附件2有{q2['spreadsheet']['numeric_video_id_rows']}条数字型video_id记录，不能强加“必须11位”的规则。附件2标签表没有重复ID或重复整行，有{len(q2['spreadsheet']['duplicate_normalized_text'])}组规范化后重复原文；不同素材可能说出同一句话，不能仅据此删除。附件2保留test仅做结构、范围、符号、ID及零值质量检查，没有使用其标签分布或预测表现选模。",
        "",
        "## 2. 异常与缺失不是同一类问题",
        "",
        "两条原始音轨（`-mJ2ud6oKI8$_$1`、`-mJ2ud6oKI8$_$2`）逐声道解码后全为数字0。它们是源数据不可用例外，不能从现有音轨恢复声学事件，也不能均匀分配时间冒充对齐。所有100条均保留，静音样本保存未对齐原生特征。",
        "",
        f"345个短语中视觉状态为：可靠几何观测{q1['visual_phrase_states'].get('observed',0)}、有观测但主体歧义{q1['visual_phrase_states'].get('observed_with_subject_ambiguity',0)}、没有可靠人脸检出{q1['visual_phrase_states'].get('no_reliable_face_detected',0)}、采样未覆盖{q1['visual_phrase_states'].get('sampling_no_coverage',0)}、几何拒绝{q1['visual_phrase_states'].get('detected_geometry_rejected',0)}。后三类合计{q1['visual_missing_phrases']}个视觉缺失短语；另{q1['pitch_missing_phrases']}个短语无有效基频。无声区基频缺失是正常定义，不能按有效0Hz参与统计。",
        "",
        "新增`visual_observation_state`记录上述可观测原因；保持缺失掩码，不把“检测不到”直接解释为原画面必然无人。4条样本的71个有效歧义帧仍需主体核验，现有值只描述所选可见人脸。自动对齐质量失败44条，属于候选时间需要核验的问题，不能靠重复计算消除。",
        "",
        "实际时长2.257—29.288秒，2条短于题面2.648秒下界，保留实测时长。约1秒一次的缩略帧筛查在3条视频中发现近黑画面候选；未发现相邻采样缩略帧完全重复。没有样本超过预设0.1%的近满幅音频采样比例。近黑画面和近满幅仅为筛查，不证明损坏、卡帧或削波不存在，也不对样本作自动删除。",
        "",
        f"IQR规则共标记{q1['iqr_candidate_count']}个“样本—指标”组合，涉及{q1['iqr_candidate_samples']}条样本；其中时长11条位于上侧长尾。IQR是Q1−1.5IQR/Q3＋1.5IQR的描述性筛查，不是错误标准。较长文本、极性较强但合法的标签都可能被标记，未截尾、改值或删样本。",
        "",
        "## 3. 关键指标分布",
        "",
        "| 指标 | 有效n | 均值 | 标准差 | 中位数 | Q25—Q75 | 最小—最大 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for d in q1["distributions"]:
        lines.append(f"| {d['metric']} | {d['n_valid']} | {format_number(d['mean'])} | {format_number(d['std'])} | {format_number(d['median'])} | {format_number(d['q25'])}—{format_number(d['q75'])} | {format_number(d['min'])}—{format_number(d['max'])} |")
    lines += [
        "",
        "附件1负向18%、中性25%、正向57%，明显偏向正向；这不违反标签规范，但不能把本100条当作类别均衡样本。附件2train三类为967/758/1670（28.48%/22.33%/49.19%），valid为206/184/338（28.30%/25.27%/46.43%）。第一问只做描述统计；这些标签未用于调整提取参数或映射质量门。",
        "",
        "每条视频是主统计单位，37个video_id各含1—9条片段。视觉均值只用有效几何帧，基频均值只用有声有效窗；缺失置为统计NA，不能把输出中占位0当观测。150维逐维分布采用345个短语，统计有效n、缺失n、均值、标准差、分位数与IQR候选；这是短语加权口径，不等于样本等权或视频时长加权。",
        "",
        "![第一问分布与观测质量](data_quality/q1_distributions.svg)",
        "",
        "## 4. 相关性与解释边界",
        "",
        "同时报告Pearson与Spearman，逐对剔除缺失并保留有效n。以下是指定结构关系及标签关系的描述，不作多重比较后的显著性结论，也不据相关性选特征。包含无声样本的log-RMS使用真实静音声学窗值；基频则排除无有效有声窗样本。",
        "",
        "| 变量对 | n | Pearson r | Spearman ρ | 按video_id均值聚合后的r（n） |",
        "|---|---:|---:|---:|---|",
    ]
    pairs=[("duration_s","text_chars"),("text_words","token_count"),("native_corner_61_mean","native_corner_291_mean"),
           ("label","native_mouth_open_mean"),("label","f0_mean_hz"),("label","native_zcr_mean")]
    for a,b in pairs:
        r=next(r for r in correlations if {r["x"],r["y"]}=={a,b})
        g=next(r for r in grouped if {r["x"],r["y"]}=={a,b})
        lines.append(f"| {a} / {b} | {r['n']} | {number(r['pearson_r']):.3f} | {number(r['spearman_rho']):.3f} | {number(g['pearson_r']):.3f}（{g['n']}） |")
    lines += [
        "",
        "时长与文本长度的高相关符合较长片段往往有更多转写的事实；两侧嘴角指标约0.98的相关提示几何冗余，但不构成删除维度的充分理由。标签与嘴开合、基频的弱至中等相关可能受说话人、题材、拍摄、语速等混杂，不能推出嘴开合导致正向情绪，也不证明这150维已足以预测情感。",
        "",
        "按video_id取均值是对片段依赖和来源权重的敏感性描述，会改变统计对象，并未消除混杂或建立独立重复试验。附件2按有效位置RMS计算的标签相关较弱：对齐版audio RMS的Pearson在train约0.102、valid约0.134；text RMS约−0.111/−0.033。RMS只概括向量幅度，不能代表语义信息量或模态预测能力。",
        "",
        "![相关性矩阵](data_quality/q1_correlations.svg)",
        "",
        "## 5. 附件2—4的使用适合性",
        "",
        "对齐版train/valid共有76882/17172个有效语义位置：audio原生全零行65/46，vision原生全零行4249/980；整段视觉张量全零110/15条。这是原生不可用状态，不能因称为“完整条件”而忽略。text在非语义位置也可以非零，必须依赖attention及特殊词元掩码，不能按向量非零判断有效。数值有限和ID完整均不能单独说明所有模态有可靠观测。",
        "",
        "![附件2类别与原生零值](data_quality/attachment2_distributions.svg)",
        "",
        "| 未对齐视觉长度核查 | train | valid | test（质量计数） |",
        "|---|---:|---:|---:|",
    ]
    us=q2["versions"]["unaligned_50"]["splits"]
    lines.append("| 声明长度之后仍有非零视觉行的样本 | "+" | ".join(str(us[s]["modalities"]["vision"]["samples_with_nonzero_after_declared_length"]) for s in ("train","valid","test"))+" |")
    lines.append("| 上述非零行总数 | "+" | ".join(str(us[s]["modalities"]["vision"]["nonzero_outside_valid_positions"]) for s in ("train","valid","test"))+" |")
    lines += [
        "",
        "例如train样本`-HwX2H8Z4hY$_$3`的vision_lengths=1，却有14个非零行，最后非零位置索引84。全4850条的vision_lengths都恰等于“首个全零视觉行索引，最小取1”；这是本次核得的数值关系，不等于已取得原提取代码。因此，将该字段直接解释为可裁剪的有效前缀会丢失观测。当前未对齐版的前缀统计仅描述“声明前缀”，真实时间与缺失/填充边界仍需来源说明；不将其报告为已正确恢复的有效率。既定Q2/Q3继续使用对齐版，此处未修改原字段或混用版本。",
        "",
        "附件3对齐版提供浮点存储的3×50词元输入，已检查整数性。当前30份文件没有检出SEP之前的零词元或attention内孔；音频/视觉分别在27份文件中有语义位置全零行。适配器仍不能把零值一概当填充。未对齐版只有raw_text和音视频数组，缺少text_bert及有效长度字段，不能声称两版本接口相同。零值检查记录的是可见状态，不从30条专项文件反推训练缺失分布。",
        "",
        f"附件4两版本的20组原文/词元和视频哈希对应一致；对齐版{len(q4_zero)}份文件有合计{sum(r['vision_zero_semantic_positions'] for r in q4_zero)}个语义位置视觉全零行。未对齐版{len(q4_suffix)}份文件有合计{sum(r['vision_nonzero_after_length'] for r in q4_suffix)}个声明长度之后的非零视觉行，需同样澄清长度含义。“原始场景三模态完整”不意味着所有提取行都非零。配对检查不替代实际时间定位或人工对齐验收，未提前进行归因排序或专项预测。",
        "",
        "## 6. 第一问重新执行与复核",
        "",
        "本轮前快照保存于`intermediate/dataset_quality_20260924/before.zip`，271文件、28,667,806字节；旧波形另存`audio_before/`，原生缓存旧摘要保存为`native_before.json`。",
        "",
        "质量问题已触发完整重新执行：重新从原视频解码波形，使用固定参数重跑Whisper给定文本对齐、冻结BERT、声学量、人脸几何及短语聚合（`extract --force`）。未换样本、未按标签调参。新增视觉缺失原因字段，并修正重复聚合时自动生成的标记可能重复保留的问题。",
        "",
        f"重跑请求{rerun['requested_samples']}条，错误{rerun['run_errors']}条；与快照相比{rerun['identical_npz_arrays']}条NPZ的全部数组逐值一致，发生数组变化{len(rerun['changed_samples'])}条。完整比较记录见[data_quality/q1_rerun_comparison.json](data_quality/q1_rerun_comparison.json)。同参数重跑的稳定性是计算复现证据，不是边界准确性证据。",
        f"原生特征也重新计算：{rerun['identical_native_npz_hashes']}/{rerun['native_npz_files']}份文本、声学和视觉NPZ的文件摘要与重跑前一致。29项测试通过，包含有效0与统计缺失区分、逐对相关性有效n、填充排除、视觉原因分类、数字ID与标签逻辑，以及原有人工验收和重聚合集成测试。",
        "",
        f"全量验证：{validation['checked_samples']}条结构与独立聚合重算通过；最大绝对重算差{max(r['max_reconstruction_abs_error'] for r in validation['samples']):.12g}。当前{validation['strictly_accepted_samples']}条有映射且自动验收通过、{validation['human_boundary_evaluation']['required_samples']}条待实际复核、{validation['verified_source_exception_samples']}条静音源数据例外。人工独立参考仍为0，中位/P90误差不可得；严格发布仍失败。",
        "",
        "本轮报告修正了历史说明与实测混用、缺失原因混称、将有限值等同可用观测、将相关性等同情感有效性、将复算一致等同准确性等潜在误解。当前可用于方法展示、接口开发和有掩码的数据分析；尚不能认定为通过实际对齐验收的最终第一问。",
        "",
        "## 7. 全100条结果表",
        "",
        "各条均保存文本/声学/视觉三模态，维度128＋16＋6=150；下表P为真实短语数。待复核包含硬门失败和其他按政策必须复核的样本，不与硬门通过率混称。无映射的两条保留原生粒度，不能把P=0解读为删样本。",
        "",
        "| 样本ID | 时长(s) | 三模态维度 | 粒度 | P | 视觉有效帧率 | 验收状态 |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for r in allrows:
        state={"aligned_auto_accepted":"自动验收通过","verified_source_exception":"静音源例外",
               "aligned_awaiting_review":"待复核"}.get(r["acceptance_status"],r["acceptance_status"])
        lines.append(f"| {r['sample_id']} | {number(r['duration_s']):.3f} | 128/16/6 | {'短语' if int(r['phrase_count']) else '原生未对齐'} | {r['phrase_count']} | {number(r['face_valid_rate']):.1%} | {state} |")
    lines += [
        "",
        "## 8. 证据与复现入口",
        "",
        "- [逐样本指标](data_quality/q1_samples.csv)、[逐短语观测](data_quality/q1_phrases.csv)、[150维统计](data_quality/q1_feature_dimensions.csv)、[IQR候选清单](data_quality/q1_iqr_candidates.csv)。",
        "- [Pearson/Spearman及有效n](data_quality/q1_correlations.csv)、[按视频来源聚合的相关性](data_quality/q1_video_group_correlations.csv)。",
        "- [源视频解码记录](data_quality/media_integrity.json)、[附件1审计](data_quality/q1_audit.json)、[附件2审计](data_quality/q2_audit.json)、[附件3/4审计](data_quality/special_audit.json)。",
        "- [本轮复算对比](data_quality/q1_rerun_comparison.json)、[全量验证](validation.json)、[逐样本验收](../outputs/acceptance.jsonl)、[源数据例外](source_exceptions.json)。",
        "",
        "可从项目根目录运行`第一问/.venv/bin/python 第一问/src/dataset_audit.py`完成审计；`--skip-media`只复用本轮解码证据，`--only-q1`只刷新Q1统计。图表与报告由`src/report_data_quality.py`读取落盘JSON/CSV生成。完整重跑日志保存在本地`logs/dataset_quality_full_extract.log`；诊断包含核心审计代码、结构化证据及图表，原素材和本地缓存不重复打包。",
    ]
    (ROOT/"reports/数据集系统分析与第一问复核报告_2026-09-24.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!="evidence_manifest.json"]
    dump(OUT/"evidence_manifest.json",{"files":[{"path":str(p.relative_to(ROOT)),"sha256":sha256(p)} for p in sorted(files)],
                                      "report":"reports/数据集系统分析与第一问复核报告_2026-09-24.md"})
    print("Data quality report and figures generated.")


if __name__=="__main__":
    main()
