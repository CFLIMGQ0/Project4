# ALM-MIL manuscript

当前引言位于 `sec/1_intro.tex`，相关工作位于 `sec/2_related_work.tex`，均采用作者最新提供的中英文原文，按一段英文、一段中文翻译排列。`main.tex` 是纯英文论文入口，`draft.tex` 是用于阅读的中英对照入口；两者通过同一份 `main.tex`、`preamble.tex`、正文分节和参考文献共享全部内容与配置。模板摘要、示例图片和格式说明暂未启用。

使用 **XeLaTeX** 编译（Overleaf 中也请选择 XeLaTeX）：

纯英文论文：

```bash
cd cvpr
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

中英对照稿使用同样的流程，将入口文件和辅助文件名替换为 `draft`：

```bash
xelatex -interaction=nonstopmode -halt-on-error draft.tex
bibtex draft
xelatex -interaction=nonstopmode -halt-on-error draft.tex
xelatex -interaction=nonstopmode -halt-on-error draft.tex
```

如已安装 `latexmk`，可分别运行 `latexmk main.tex` 或 `latexmk draft.tex`。后续只需编辑共享的正文文件，不要在两个入口之间复制内容。

当前研究设定：训练与推理均输入一组有序图像及其配对的一份报告，预测多个共存标签。

| 固定术语 | 英文全称 | 当前引言中的作用 |
| --- | --- | --- |
| ALM-MIL | Acquisition-aware Label-centric Multimodal Multiple Instance Learning | 检查级多模态多标签框架 |
| ACPE | Acquisition-aware Contextual Position Encoding | 以原始采集坐标为锚，根据视觉转变构建内容自适应位置，并保持序列单调性 |
| LCCF | Label-Centric Coexistence Fusion | 标签特异性视觉聚合、共存标签超图推理、标签条件文本检索、标签级门控融合 |

本文内容以作者后续提供的新版本材料为依据。旧项目中的方法、数据口径和实验结果不自动沿用。引言使用 8 篇相关文献，相关工作按作者提供的 27 个编号组补充引用；使用模板自带的 `natbib` 数字引用及 `ieeenat_fullname.bst`，中英文共享编号。出处、引用位置和表述边界见 [引言引用核验说明](refer/intro_citation_notes.md) 和 [相关工作引用核验说明](refer/related_work_citation_notes.md)。首次提出声明及本研究的实验总结保留作者原文，仍需后续相关工作检索及最终实验依据支持。

下面保留原始模板说明。

# Official LaTeX template for CVPR/ICCV/3DV 

**Note:** as per PC decision, the Microsoft Word version of the template is no longer supported.
You may find the 2024 version [here](https://github.com/cvpr-org/author-kit/releases/tag/CVPR2024-v3(msword)).

### History (in reverse chronological order)

- updated for CVPR 2026 [Vladimir Pavlovic](mailto:vladimir@rutgers.edu)
- added styles for `subsubsection` and fixed the wrong PDF bookmarks by [Di Fang](https://github.com/fang-d)
- modernized for CVPR 2025 by [Christian Richardt](https://richardt.name/)
- fixed page centering for CVPR 2025 by [Stefan Roth](mailto:stefan.roth@NOSPAMtu-darmstadt.de)
- inline enumerations and `cvprblue` links for CVPR 2025 by [Ioannis Gkioulekas
](https://www.cs.cmu.edu/~igkioule/)
- added automated LaTeX build testing for CVPR 2025 by [Ahan Shabanov](https://ahanio.github.io)
- references in `cvprblue` for CVPR 2024 by [Klaus Greff](https://github.com/Qwlouse) 
- added natbib for CVPR 2024 by [Christian Richardt](https://richardt.name/)
- replaced buggy (review-mode) line numbering for 3DV 2024 by [Adín Ramírez Rivera
](https://openreview.net/profile?id=~Ad%C3%ADn_Ram%C3%ADrez_Rivera1)
- setup github repo of author-kit for 3DV 2025 by [Andrea Tagliasacchi](https://theialab.ca)
- modernized for CVPR 2022 by [Stefan Roth](mailto:stefan.roth@NOSPAMtu-darmstadt.de)
- created cvpr.sty file to unify review/rebuttal/final versions by [Ming-Ming Cheng](https://github.com/MCG-NKU/CVPR_Template)
- developed CVPR 2005 template  by [Paolo Ienne](mailto:Paolo.Ienne@di.epfl.ch) and [Andrew Fitzgibbon](mailto:awf@acm.org)
