# Related-work citation audit

核验和整理日期：2026-09-22。新版相关工作中的 `[1--27]` 已替换为 BibTeX citation keys；编号分组仍按作者提供的四个主题段落保持不变。部分位置建模和图像间距研究来自通用序列或 CT slice-localization 场景，引用用于说明方法路线，不表示这些工作直接解决离散内镜检查任务。

| 原编号组 | BibTeX keys | 主题 |
| --- | --- | --- |
| [1--4] | `vaswani2017attention`, `shaw2018relative`, `su2021roformer`, `press2022alibi` | 固定、相对、旋转位置编码与注意力偏置 |
| [5--8] | `golovneva2024cope`, `yang2025path`, `zheng2024dape`, `zheng2025dapev2` | 内容/数据自适应位置机制 |
| [9--10] | `su2025ctslice`, `tang2021bodypart` | 从图像估计 slice 位置或间距的 body-part regression 路线 |
| [11--15] | `ilse2018attention`, `lu2021clam`, `li2021dsmil`, `shao2021transmil`, `zhang2022dtfd` | 注意力池化、实例交互和层次化 MIL |
| [16--21] | `li2021albef`, `li2022blip`, `zhang2022convirt`, `huang2021gloria`, `zhou2022refers`, `zhong2026mmtf` | 图文对齐、全局—局部对应和医疗图文融合 |
| [22--23] | `li2022mallcnn`, `yang2018m3dn` | 视频多标签 MIL 与多模态多实例多标签学习 |
| [24--27] | `ridnik2021asymmetric`, `chen2019mlgcn`, `feng2019hypergraph`, `zhu2025mambaml` | 标签感知表示、标签依赖和高阶图推理 |

## Evidence boundaries

- `su2025ctslice` 和 `tang2021bodypart` 直接研究 CT slice 的位置分数、排序或跨体积校准，支持“图像驱动位置估计”这一背景，但不等同于本文的内镜视觉间距恢复。
- `li2022mallcnn` 明确结合视频帧级 MIL 与标签关系图；`yang2018m3dn` 建模多模态、多实例、多标签对象，二者支持相关任务范式的存在，不代表其输入与本文检查图像—报告配对完全相同。
- `zhu2025mambaml` 同时建模视觉上下文、标签依赖和标签—图像交互；本文只将其用于标签感知表示与标签关系建模的相关工作说明。
- 第三段最后两句是本文对 ACPE 与 LCCF 的定位，不使用外部文献证明原创性或实验效果。
