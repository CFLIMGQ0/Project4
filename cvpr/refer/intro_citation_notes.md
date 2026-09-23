# Introduction citation audit

检索和元数据核验日期：2026-09-22。新版引言的引用只支持相关工作和通用方法描述，不支持 ALM-MIL、ACPE、LCCF 的原创性，也不支持尚未提供的实验结果。第一段的临床任务动机、局限性分析和四条贡献保留为作者论述，没有添加不直接支撑的外部文献。

## Citation map

| 引用键 | 新版引言位置 | 支撑内容 | 核验来源 |
| --- | --- | --- | --- |
| `yu2025comrope` | 位置建模段 | 强支撑：通过可学习的可交换角度矩阵改进 RoPE | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_ComRoPE_Scalable_and_Robust_Rotary_Position_Embedding_Parameterized_by_Trainable_CVPR_2025_paper.html), pp. 4508--4517 |
| `wei2025videorope` | 位置建模段 | 强支撑：时间维度分配和可调时间间距，用于表达时空结构 | [ICML 2025 / PMLR](https://proceedings.mlr.press/v267/wei25h.html), pp. 66118--66136 |
| `yang2025path` | 位置建模段 | 强支撑：由输入决定的 Householder 变换累积形成内容相关位置编码 | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/59c27bf8d56d3d50c7aeaf7535dee975-Abstract-Conference.html) |
| `zheng2025dapev2` | 位置建模段 | 强支撑：跨相邻位置和注意力头处理 attention scores；不直接学习采集坐标 | [ACL Anthology 2025](https://aclanthology.org/2025.acl-long.522/), pp. 10628--10666 |
| `zhang2025saif` | 图文融合段 | 强支撑：学习图文语义映射、筛选相关特征并进行自适应融合 | [Pattern Recognition 167 (2025)](https://www.sciencedirect.com/science/article/pii/S0031320325003759), article 111715 |
| `zhong2026mmtf` | 图文融合段 | 强支撑：token exchange、多尺度交叉注意力和图文表示整合 | [Biomedical Signal Processing and Control 111 (2026)](https://www.sciencedirect.com/science/article/pii/S1746809425008298), article 108318 |
| `zhu2025mambaml` | 图文融合段 | 强支撑：将视觉信息聚合到标签嵌入，并建模标签依赖和交叉关系 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_MambaML_Exploring_State_Space_Models_for_Multi-Label_Image_Classification_ICCV_2025_paper.pdf), pp. 4743--4753 |
| `wang2025granularity` | 图文融合段 | 强支撑：利用结构化类别描述丰富类别语义，并在共享图文空间对齐 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Language-Driven_Multi-Label_Zero-Shot_Learning_with_Semantic_Granularity_ICCV_2025_paper.html), pp. 1968--1978 |

## Placement decisions

- 新版第一段定义检查级任务并说明临床动机；没有添加方法引用，避免用一般背景论文替代任务定义的作者表述。
- 第二段只引用点名的四种位置建模方法；英文和中文使用同一 citation key。
- 第三段只引用点名的 SAIF、MMTF、MambaML 和 Wang et al.；新版不再讨论旧版的 Liu et al. 方法。
- 第四段是基于前述方法的综合分析，不把这些论文的实验结果误写成对本文任务局限性的直接证明，因此不在段末重复堆叠引文。
- ALM-MIL、ACPE、LCCF 及四条贡献描述是本文自身内容，暂不添加外部引用。

## Boundary notes

- ComRoPE 和 VideoRoPE 分别讨论可学习 RoPE 变换和视频时空位置结构；它们只能作为相关位置建模工作的例子，不能直接证明内镜图像的采集位置关系。
- DAPE V2 处理 attention scores 的邻域与跨头结构。它支持新版中的机制描述，但不应被表述为直接重建真实采集坐标。
- MambaML 同时建模视觉上下文、标签依赖以及标签与图像特征之间的交互。新版因此只陈述其单图多标签表示学习能力，不将其作为“只在某一阶段引入标签信息”的证据。
- SAIF 的 BibTeX 保留完整题名；出版社摘要支持图文映射、相关性掩码和简化注意力融合机制，但论文正式缩写是否为 SAIF 仍沿用作者当前称呼。
- “To the best of our knowledge, this is the first ...” 已从新版贡献列表移除，避免在尚未完成完整相关工作检索前作范围性首创声明。
- “extensive experiments”以及图像缺失鲁棒性、训练期多模态学习对纯图像推理的帮助，均属于待新主机结果确认的作者声明；本次不据此添加实验文献。
