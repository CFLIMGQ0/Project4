#!/usr/bin/env python3
"""Build monolingual LaTeX sources from the bilingual main.tex.

The bilingual source remains the single source of truth.  This script only
selects the existing English or Chinese counterpart of each prose block.  For
the Chinese edition it also translates structural metadata that occurs only
once in the bilingual source (title, headings, captions, and table headers).
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "main.tex"
ENGLISH_OUTPUT = ROOT / "main_en.tex"
CHINESE_OUTPUT = ROOT / "main_zh.tex"

CHINESE_RE = re.compile(r"[\u3400-\u9fff]")
TRANSLATION_MARKER_RE = re.compile(
    r"(?:\\noindent)?\\textbf\{中文(?:摘要|翻译)：\}\s*"
)


SECTION_TITLES = {
    "Introduction": "引言",
    "Related Work": "相关工作",
    "Endoscopic Datasets and Examination-Level Multi-Image Learning": "内镜数据集与检查级多图像学习",
    "Multi-Label MIL and Higher-Order Label Reasoning": "多标签多实例学习与高阶标签推理",
    "Image--Report Multimodal Medical Learning": "图像--报告多模态医学学习",
    "Acquisition-Aware Position Modeling": "采集感知位置建模",
    "Materials and Methods": "材料与方法",
    "Dataset construction and preprocessing": "数据集构建与预处理",
    "Study design and source data": "研究设计与数据来源",
    "Data cleaning, standardization, and clinical review of target labels": "数据清洗、标准化与目标标签临床审核",
    "Examination-level multimodal sample construction": "检查级多模态样本构建",
    "Overview of AMEF-MIL": "AMEF-MIL总体结构",
    "Image Encoding": "图像编码",
    "Image encoding and APro-CoPE procedural context modeling": "图像编码与APro-CoPE流程上下文建模",
    "Multi-label attention MIL": "多标签注意力MIL",
    "Label hypergraph reasoning": "标签超图推理",
    "Text Encoding": "文本编码",
    "Finding-description selection and category masking": "所见描述选择与类别名称掩码",
    "TextCNN-based finding-description encoding": "基于TextCNN的所见描述编码",
    "Cross-Modal Fusion": "跨模态融合",
    "Label-query cross-attention and gated fusion": "标签查询交叉注意力与门控融合",
    "Prediction Branches": "预测分支",
    "Primary multimodal branch": "主要多模态分支",
    "Image-only branch and multimodal knowledge transfer": "纯图像分支与多模态知识迁移",
    "Experiments": "实验",
    "Experimental Settings": "实验设置",
    "Overall Performance Comparison and Label-wise Analysis": "总体性能比较与标签级分析",
    "Comparison with image-only, text-only, and multimodal models": "纯图像、纯文本与多模态模型比较",
    "Out-of-fold label-wise confusion analysis": "折外标签级混淆分析",
    "Analysis and Ablation of Model Components": "模型组件分析与消融实验",
    "Image budget and position encoding": "图像预算与位置编码",
    "Component analysis of APro-CoPE": "APro-CoPE组件分析",
    "Qualitative analysis of APro-CoPE under subsampling and full-sequence retention": "子采样与完整序列保留条件下的APro-CoPE定性分析",
    "Effect of APro-CoPE on examination-level prediction quality": "APro-CoPE对检查级预测质量的影响",
    "Ablation of multi-label attention MIL and label hypergraph reasoning": "多标签注意力MIL与标签超图推理消融",
    "Ablation of label-conditioned cross-modal retrieval and adaptive residual fusion": "标签条件跨模态检索与自适应残差融合消融",
    "Full-factorial ablation of the main architectural components": "主要架构组件的全因子消融",
    "Image-only Knowledge Transfer": "纯图像知识迁移",
    "Discussion": "讨论",
    "Conclusions": "结论",
    "Author contributions": "作者贡献",
    "ORCID iDs": "ORCID标识",
    "Statements and declarations": "声明",
    "Ethical considerations": "伦理考虑",
    "Consent for publication": "发表同意",
    "Data availability": "数据可用性",
}


CAPTIONS = {
    "fig:dataset_pipeline": "数据集构建与预处理流程。对原始图像--报告记录进行清洗和标准化，根据诊断结论生成初始检查级标签并由两名主任医师独立审核，对胃镜所见描述中的直接目标表述进行掩码，并按照患者分组构建用于多模态学习的数据子集。",
    "T1": "四个胃镜数据集的分折检查数量。每个单元格按照训练集/验证集/测试集格式表示。数量统计于训练集过采样之前；验证集和测试集均未进行重平衡。",
    "fig:label_cooccurrence": "四个胃镜数据集的原始与重平衡标签共现矩阵。各列依次展示WLE、染色胃镜、手术胃镜和EUS的分布。上排为原始检查分布，下排为每个数据集中五个独立重平衡训练集汇总后的累积分布；验证集和测试集均未重平衡，也未纳入汇总。每个单元格给出相应数量和百分比。对角线表示单标签阳性频率，非对角线表示两两共阳性频率。百分比以对应原始数据集或汇总训练曝光为基数计算。SMT表示食管黏膜下肿瘤，EML表示食管黏膜病变。",
    "fig:model_architecture": "AMEF-MIL总体结构。图像分支结合共享视觉编码、基于APro-CoPE的双向上下文建模、多标签注意力MIL和标签超图推理。文本分支采用多尺度TextCNN编码经过类别名称掩码的所见描述。标签级交叉注意力和门控残差融合生成主要多模态预测；经超图细化的视觉表征支持辅助纯图像预测，并在训练阶段接收来自预训练多模态教师的知识。",
    "fig:apro_cope": "APro-CoPE结构。原始采集索引用于定义归一化采集锚点；双向视觉转变用于估计有界上下文变形分数；质量守恒归一化在保留观测序列端点的同时构建单调上下文坐标。绝对位置路径增强实例特征，相对位置路径向双向Transformer提供注意力头特异性偏置。",
    "T2": "四个胃镜数据集上纯图像、纯文本与多模态方法的比较。数值为患者分组五折宏平均F1的均值$\\pm$标准差。img.和txt.分别表示内镜图像输入和经过类别名称掩码的胃镜所见描述输入。",
    "fig:oof_confusion_selected": "AMEF-MIL在五个患者分组测试折上汇总的折外一对其余混淆矩阵。各行依次对应WLE、染色胃镜、手术胃镜和EUS数据集，各列对应三个目标标签。每个单元格给出汇总数量和行归一化百分比，每次检查仅出现在一个测试折中。",
    "T3": "四个胃镜数据集上采样图像预算与位置编码的影响。在每种图像预算下，使用相同上下文编码器分别评价无位置编码、采样槽位位置编码和APro-CoPE。数值为患者分组五折宏平均F1的均值$\\pm$标准差。PE表示位置编码。",
    "T3_apro": "四个胃镜数据集上APro-CoPE的组件分析，每次检查采样64张图像。所有变体均采用相同的两层双向Transformer，仅改变位置机制。数值为患者分组五折宏平均F1的均值$\\pm$标准差。w/o表示移除，PE表示位置编码，Trans.表示转变机制，CG表示内容门控，PW表示以成对转变替代PT，PT表示持续转变，AA表示采集锚点，MC表示质量守恒，Abs.表示绝对位置路径，Rel.表示相对位置路径。",
    "fig:apro_cope_length_groups": "子采样与完整序列保留条件下的APro-CoPE定性可视化。最大图像预算为$T=64$时，上组在$N>64$时均匀采样64张图像，下组在$N\\leq64$时保留完整序列。",
    "fig:apro_cope_prediction_distributions": "四个胃镜数据集上APro-CoPE相对于采样槽位位置编码对检查级预测质量的影响。各行依次对应WLE、染色胃镜、手术胃镜和EUS，各列展示标签级准确率、真实类别置信度和Brier分数。每个点表示在对应留出测试折上评价的一次检查。图像数量超过64张时，在覆盖原始序列的采集顺序分层样本上对指标取平均，采样轮次最多为5次。小提琴形状表示分布，箱线表示四分位距与中位数，菱形表示算术均值，$\\bar{x}$给出相应均值。标签级准确率和真实类别置信度越高越好，Brier分数越低越好。",
    "T_label_reasoning": "四个胃镜数据集上多标签注意力MIL与标签超图推理的针对性消融，每次检查采样64张图像。各配置均保留采集感知视觉上下文建模、标签查询交叉注意力和门控残差融合。数值为患者分组五折宏平均F1的均值$\\pm$标准差。",
    "fig:label_evidence_overlap": "不同MIL架构的标签特异性图像证据分离情况。对折1留出测试集中的每次检查，使用同一组最多64张均匀采样图像，分别按照三个标签进行排序，并计算三组Top-5证据集合之间的平均两两Jaccard重叠率。每个点表示一次检查，箱线表示四分位距与中位数，百分比表示算术均值。重叠率越低表示标签特异性支持证据的分离越强。",
    "fig:label_targeted_deletion": "四个胃镜数据集上的标签定向删除分析。对每个被删除证据的标签（行），在标签超图推理前移除其Top-5注意力图像，并将由此产生的决策影响分配至三个输出标签（列），随后在每行内归一化为100\\%。上排为共享注意力对照，下排为所提出模型。红色边框标记被删除标签与受影响标签一致的位置；对角线越集中表示标签特异性决策选择性越强。",
    "fig:label_hypergraph_copositive_confidence": "四个胃镜数据集上不同标签集合复杂度下的阳性标签置信度。依据阳性目标标签数量将检查分为单阳性组和共阳性组。柱形表示模型赋予阳性标签的平均置信度，误差线表示95\\%置信区间。比较配置均采用多标签注意力，仅标签推理机制不同。",
    "T_crossmodal_fusion": "四个胃镜数据集上标签查询交叉注意力与门控残差融合的$2\\times2$消融，每次检查采样64张图像。各配置均保留基于APro-CoPE的采集感知上下文建模和标签超图推理。数值为患者分组五折宏平均F1的均值$\\pm$标准差。",
    "fig:label_query_retrieval_specificity": "四个胃镜数据集上的标签特异性文本检索分析。正确条件保留对应标签查询所检索的文本表征，跨标签条件在不同标签之间重新分配检索表征，共享池化条件以统一所见描述表征进行替换。折线表示合并四个数据集后模型赋予阳性标签的平均置信度。",
    "T_factorial": "四个胃镜数据集上四个主要架构组件的全因子消融，每次检查采样64张图像。数值为患者分组五折宏平均F1的均值$\\pm$标准差。M1表示基于APro-CoPE的采集感知Transformer上下文建模，M2表示标签超图推理，M3表示标签查询交叉注意力，M4表示门控残差融合。M1内部的位置设计选择见表~\\ref{T3_apro}。",
    "T_image_transfer": "不同训练阶段多模态知识迁移策略下辅助纯图像分支的五折性能。MM sup.表示可训练多模态分支的直接监督，KD表示来自预训练且冻结多模态教师的知识蒸馏。第一行仅使用纯图像监督，不含多模态引导。所有配置在验证和测试阶段均只使用图像。数值为患者分组五折宏平均F1的均值$\\pm$标准差。",
}


TABLE_HEADER_REPLACEMENTS = {
    "Fold & WLE & Chromoendoscopy & Surgical gastroscopy & EUS \\\\": "折 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
    "Model & img. & txt. & WLE & Chromoscopic & Surgical & EUS \\\\": "模型 & 图像 & 文本 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
    "Images & Method & WLE & Chromoscopic & Surgical & EUS \\\\": "图像数 & 方法 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
    "Position variant & Trans. & AA & MC & Abs. & Rel. & WLE & Chromoscopic & Surgical & EUS \\\\": "位置变体 & 转变 & AA & MC & 绝对 & 相对 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
    "MIL aggregation & Label reasoner & WLE & Chromoscopic & Surgical & EUS \\\\": "MIL聚合 & 标签推理器 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
    "Text retrieval & Residual fusion & WLE & Chromoscopic & Surgical & EUS \\\\": "文本检索 & 残差融合 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
    "Model & KD & MM sup. & WLE & Chromoscopic & Surgical & EUS \\\\": "模型 & KD & 多模态监督 & WLE & 染色胃镜 & 手术胃镜 & EUS \\\\",
}


def normalize_structure(text: str) -> str:
    """Ensure structural commands after translated prose form their own chunks."""
    return re.sub(
        r"(?<!\n)\n(?=\\(?:section|subsection|subsubsection)\*?\{)",
        "\n\n",
        text,
    )


def split_chunks(text: str) -> list[str]:
    return re.split(r"\n\s*\n", text.strip())


def is_translation_chunk(chunk: str) -> bool:
    return "中文摘要" in chunk or "中文翻译" in chunk


def translation_groups(chunks: list[str]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(chunks):
        if not is_translation_chunk(chunks[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(chunks) and CHINESE_RE.search(chunks[i]):
            i += 1
        groups.append((start, i))
    return groups


def is_shared_boundary(chunk: str) -> bool:
    stripped = chunk.lstrip()
    prefixes = (
        "\\documentclass",
        "\\usepackage",
        "\\ifXeTeX",
        "\\newcommand",
        "\\def",
        "\\begin{document}",
        "\\runninghead",
        "\\title",
        "\\author",
        "\\affiliation",
        "\\corrauth",
        "\\email",
        "\\begin{abstract}",
        "\\keywords",
        "\\maketitle",
        "\\medskip",
        "\\section",
        "\\subsection",
        "\\subsubsection",
        "\\begin{figure",
        "\\begin{table",
        "\\FloatBarrier",
        "\\begin{thebibliography}",
        "\\end{thebibliography}",
        "\\end{document}",
    )
    return stripped.startswith(prefixes)


def build_english(source: str) -> str:
    chunks = split_chunks(normalize_structure(source))
    drop: set[int] = set()
    for start, end in translation_groups(chunks):
        drop.update(range(start, end))
    output = [chunk for index, chunk in enumerate(chunks) if index not in drop]
    text = "\n\n".join(output) + "\n"
    text = text.replace("\n\\medskip\n\n\\section{Introduction}", "\n\\section{Introduction}")
    return text


def strip_translation_marker(chunk: str) -> str:
    return TRANSLATION_MARKER_RE.sub("", chunk, count=1).strip()


def extract_top_translation(source: str, kind: str) -> str:
    if kind == "abstract":
        pattern = r"\\noindent\\textbf\{中文摘要：\}\s*(.*?)(?=\n\s*\n)"
    else:
        pattern = r"\\noindent\\textbf\{中文翻译：\}\s*关键词：(.*?)(?=\n\s*\n)"
    match = re.search(pattern, source, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not locate Chinese {kind}")
    return match.group(1).strip()


def replace_metadata(text: str, chinese_abstract: str, chinese_keywords: str) -> str:
    text = re.sub(r"\\runninghead\{.*?\}", r"\\runninghead{林子建等}", text, count=1)
    text = re.sub(
        r"\\title\{.*?\}",
        r"\\title{基于交叉注意力与超图推理的采集感知多模态多实例学习用于胃镜检查级多标签分类：一项回顾性多中心研究}",
        text,
        count=1,
    )
    text = re.sub(
        r"^\\author\{.*$",
        r"\\author{林子建\\affilnum{1,2}，蔡成泽\\affilnum{3}，黄晨曦\\affilnum{2}，高华\\affilnum{1}，何杰\\affilnum{1,4}}",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"\\affiliation\{.*?\n\s*\\affilnum\{4\}.*?\}",
        "\\\\affiliation{\\\\affilnum{1}复旦大学附属中山医院厦门医院内镜中心，中国福建厦门 361015\\\\\\\\\n"
        "\\\\affilnum{2}厦门大学信息学院，中国福建厦门 361102\\\\\\\\\n"
        "\\\\affilnum{3}福建农林大学计算机与信息学院，中国福建福州 350002\\\\\\\\\n"
        "\\\\affilnum{4}厦门市肿瘤治疗临床研究中心，中国福建厦门 361015}",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"\\corrauth\{.*?\}",
        r"\\corrauth{高华和何杰，复旦大学附属中山医院厦门医院内镜中心，中国福建厦门市湖里区金湖路668号，361015。}",
        text,
        count=1,
    )
    text = re.sub(
        r"\\begin\{abstract\}.*?\\end\{abstract\}",
        "\\\\begin{abstract}\n" + chinese_abstract + "\n\\\\end{abstract}",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"\\keywords\{.*?\}",
        "\\\\keywords{" + chinese_keywords + "}",
        text,
        count=1,
    )
    return text


def translate_structure(text: str) -> str:
    for english, chinese in SECTION_TITLES.items():
        for command in ("section", "subsection", "subsubsection"):
            text = text.replace(
                f"\\{command}{{{english}}}", f"\\{command}{{{chinese}}}"
            )
            text = text.replace(
                f"\\{command}*{{{english}}}", f"\\{command}*{{{chinese}}}"
            )

    translated_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("\\caption{"):
            for label, caption in CAPTIONS.items():
                if f"\\label{{{label}}}" in line:
                    indent = line[: len(line) - len(line.lstrip())]
                    line = indent + f"\\caption{{{caption}\\label{{{label}}}}}"
                    break
        line = TABLE_HEADER_REPLACEMENTS.get(line, line)
        line = re.sub(r"^Fold ([1-5]) &", r"第\1折 &", line)
        translated_lines.append(line)
    return "\n".join(translated_lines) + "\n"


def build_chinese(source: str) -> str:
    normalized = normalize_structure(source)
    chunks = split_chunks(normalized)
    groups = translation_groups(chunks)
    chinese_abstract = extract_top_translation(source, "abstract")
    chinese_keywords = extract_top_translation(source, "keywords")

    drop: set[int] = set()
    for start, end in groups:
        group_text = "\n\n".join(chunks[start:end])
        if "中文摘要" in group_text or "关键词：" in group_text:
            drop.update(range(start, end))
            continue
        if "作者 ORCID 如上所示" in group_text:
            drop.update(range(start, end))
            continue

        paired_english: list[int] = []
        previous = start - 1
        while previous >= 0:
            candidate = chunks[previous]
            if is_translation_chunk(candidate) or is_shared_boundary(candidate):
                break
            drop.add(previous)
            paired_english.append(previous)
            previous -= 1

        paired_english.reverse()

        # The qualitative APro-CoPE analysis places its figure between the two
        # English paragraphs and their two Chinese counterparts.
        if group_text.startswith(
            "\\textbf{中文翻译：} 为定性展示APro-CoPE如何"
        ):
            for index, chunk in enumerate(chunks):
                if chunk.startswith(
                    "To qualitatively illustrate how APro-CoPE constructs acquisition-anchored"
                ):
                    drop.add(index)
                    paired_english.insert(0, index)
                    break

        # Existing Chinese translations intentionally focus on readable prose
        # and often omit inline citation commands.  Carry the citations from
        # each English source paragraph into its Chinese counterpart so that
        # the monolingual edition retains the paper's complete attribution.
        chinese_indices = list(range(start, end))
        if len(paired_english) == len(chinese_indices):
            citation_pairs = zip(paired_english, chinese_indices)
            for english_index, chinese_index in citation_pairs:
                citations = re.findall(r"\\cite\{[^}]+\}", chunks[english_index])
                missing = [c for c in citations if c not in chunks[chinese_index]]
                if missing:
                    chunks[chinese_index] = chunks[chinese_index].rstrip() + " " + " ".join(missing)
        elif paired_english:
            citations = [
                citation
                for english_index in paired_english
                for citation in re.findall(r"\\cite\{[^}]+\}", chunks[english_index])
            ]
            missing = [c for c in citations if c not in "\n".join(chunks[start:end])]
            if missing:
                chunks[end - 1] = chunks[end - 1].rstrip() + " " + " ".join(missing)

    output: list[str] = []
    for index, chunk in enumerate(chunks):
        if index in drop:
            continue
        if is_translation_chunk(chunk):
            chunk = strip_translation_marker(chunk)
        output.append(chunk)

    text = "\n\n".join(output) + "\n"
    text = replace_metadata(text, chinese_abstract, chinese_keywords)
    text = text.replace("\n\\medskip\n", "\n")

    localization = r"""
\makeatletter
\def\abstract{\lrbox\absbox\minipage{\textwidth}%
  \sagesf\normalsize%
  \section*{\normalsize 摘要}\vskip -1.5mm%
  }
\def\endabstract{\endminipage\endlrbox}
\def\keywords#1{%
  \gdef\@keywords{\begin{minipage}{\textwidth}{\normalsize\sagesf \textbf{关键词}}\\ \parbox[t]{\textwidth}{#1}\end{minipage}}}
\def\corrauth#1{\gdef\@corrauth{%
  \footnotetext[0]{\par\vskip-3pt\sagesf\noindent\textbf{通讯作者：}\\ #1}}}
\def\email#1{%
  \gdef\@email{\footnotetext[0]{\sagesf 电子邮箱：#1}}}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\makeatother
""".strip()
    text = text.replace(
        "\\begin{document}", localization + "\n\n\\begin{document}", 1
    )

    # The SAGE environments generate English-only headings.  Use equivalent
    # Chinese headings while retaining the same declaration content.
    text = text.replace(
        "作者声明不存在与本文研究、作者身份和/或发表相关的潜在利益冲突。",
        "\\subsection*{利益冲突声明}\n\n作者声明不存在与本文研究、作者身份和/或发表相关的潜在利益冲突。",
    )
    text = text.replace(
        "本研究受到福建省自然科学基金资助（No. 2025J011511）。",
        "\\subsection*{基金项目}\n\n本研究受到福建省自然科学基金资助（No. 2025J011511）。",
    )
    text = text.replace(
        "作者感谢两家参与医院的两名主任医师对检查级标签进行独立临床审核。",
        "\\section*{致谢}\n\n作者感谢两家参与医院的两名主任医师对检查级标签进行独立临床审核。",
    )

    text = text.replace(
        "Zijian Lin \\href{https://orcid.org/0009-0007-7260-4064}",
        "林子建 \\href{https://orcid.org/0009-0007-7260-4064}",
    )
    text = text.replace(
        "Chengze Cai \\href{https://orcid.org/0009-0000-8851-1005}",
        "蔡成泽 \\href{https://orcid.org/0009-0000-8851-1005}",
    )
    text = text.replace(
        "Chenxi Huang \\href{https://orcid.org/0000-0002-6695-753X}",
        "黄晨曦 \\href{https://orcid.org/0000-0002-6695-753X}",
    )
    text = text.replace(
        "Hua Gao \\href{https://orcid.org/0009-0007-9844-998X}",
        "高华 \\href{https://orcid.org/0009-0007-9844-998X}",
    )
    text = text.replace(
        "Jie He \\href{https://orcid.org/0009-0000-6423-3928}",
        "何杰 \\href{https://orcid.org/0009-0000-6423-3928}",
    )
    text = text.replace(
        "\\begin{thebibliography}{99}",
        "\\renewcommand{\\refname}{参考文献}\n\\begin{thebibliography}{99}",
        1,
    )
    return translate_structure(text)


def validate(english: str, chinese: str) -> None:
    for name, text in (("English", english), ("Chinese", chinese)):
        if "中文翻译" in text or "中文摘要" in text:
            raise RuntimeError(f"{name} output still contains bilingual markers")
        if text.count("\\begin{document}") != 1 or text.count("\\end{document}") != 1:
            raise RuntimeError(f"{name} output has an invalid document boundary")
        if text.count("\\begin{equation}") != text.count("\\end{equation}"):
            raise RuntimeError(f"{name} output has unmatched equation environments")
        if text.count("\\begin{figure") != text.count("\\end{figure"):
            raise RuntimeError(f"{name} output has unmatched figure environments")
        if text.count("\\begin{table") != text.count("\\end{table"):
            raise RuntimeError(f"{name} output has unmatched table environments")


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    english = build_english(source)
    chinese = build_chinese(source)
    validate(english, chinese)
    ENGLISH_OUTPUT.write_text(english, encoding="utf-8")
    CHINESE_OUTPUT.write_text(chinese, encoding="utf-8")
    print(f"Wrote {ENGLISH_OUTPUT.name} ({english.count(chr(10))} lines)")
    print(f"Wrote {CHINESE_OUTPUT.name} ({chinese.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
