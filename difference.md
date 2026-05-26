# TASK1 与 TASK2 当前模型差异

本文档对比当前 TASK1 主模型与 TASK2 选定模型 `exp6_long_mil_64_no_roi`。

## 当前选定模型

| 项目 | TASK1 | TASK2 |
|---|---|---|
| 任务名 | `task1` | `task2` |
| 模型名 | `gastro_label_graph_mil` | `exp6_long_mil_64_no_roi` |
| 底层模型类 | `GastroLabelGraphMIL` | `LongMILModel` |
| 主要目标 | 胃镜三标签 examination-level 多标签分类 | 胃镜三标签 examination-level 多标签分类 |
| 标签数 | 3 | 3 |
| backbone | ConvNeXt-Tiny | ConvNeXt-Tiny |
| feature dim | 512 | 512 |
| attention dim | 256 | 256 |
| label graph | 使用 | 使用 |

## 数据与标签规则差异

两个任务都是胃镜三标签多标签分类，标签名一致：

```text
label_esophageal_smt
label_esophageal_mucosal_or_tumor
label_gastritis
```

主要差异在数据来源和筛选规则：

| 项目 | TASK1 | TASK2 |
|---|---|---|
| 默认报告文件 | `valid_dicts_report.csv` | `valid_dicts_report_for task2.csv` |
| datalist 子目录 | `task_data/task1/` | `task_data/task2/` |
| 食管黏膜/肿物标签关键词 | 使用基础关键词 | 在 TASK1 基础上额外包含 `食管粘膜病变`、`sescc` |
| 数据切分 | 普通 examination-level 切分 | 启用 `group_by_patient: true`，按患者分组切分，减少同患者泄漏 |
| 类别平衡 | 当前未启用类别平衡 | 当前配置启用训练集多标签少数类过采样 |

## 模型结构差异

### TASK1: `gastro_label_graph_mil`

TASK1 当前主模型流程是：

```text
图像实例
→ ConvNeXt-Tiny instance encoder
→ label-wise gated attention pooling
→ label graph reasoner
→ per-label classifier
```

特点：

- 每个标签有独立的 attention head，得到 label-specific bag embedding。
- label graph reasoner 使用可学习 label tokens 构建标签关系图。
- 标签图对每个 label embedding 做一次关系传播后，再进入各标签分类头。
- 当前训练输入规模为 `train_max_instances=16`。

### TASK2: `exp6_long_mil_64_no_roi`

`exp6_long_mil_64_no_roi` 是自动实验条目名，底层模型是 `long_mil`，结构是：

```text
图像实例
→ ConvNeXt-Tiny instance encoder
→ 时间/顺序位置编码
→ Transformer long-context encoder
→ label-wise gated attention pooling
→ label graph reasoner
→ per-label classifier
```

特点：

- 在 MIL pooling 之前加入 Transformer encoder，先建模长 bag 内实例间上下文关系。
- 使用位置编码，让模型感知实例在 bag 中的相对顺序。
- 仍然使用 label-wise attention 和 label graph reasoner。
- `64_no_roi` 表示每个 examination 使用 64 张原图实例，不加入 ROI crop。
- 该配置显式关闭 ROI：`roi_enabled=false`，`roi_max_crops_per_bag=0`。

## 训练输入规模差异

| 项目 | TASK1 | TASK2 `exp6_long_mil_64_no_roi` |
|---|---:|---:|
| train max instances | 16 | 64 |
| eval max instances | 16 | 64 |
| train max batch instances | 192 | 128 |
| eval max batch instances | 192 | 128 |
| batch size | 12 | 1 |
| grad accumulation | 1 | 4 |
| random instance dropout | 0.25 | 0.0 |
| train sampling | random | uniform |
| eval sampling | uniform | uniform |

## 核心区别总结

TASK1 模型更像标准 label-aware MIL：

```text
短 bag 采样 + label-wise attention + label graph
```

TASK2 选定模型更像长序列 MIL：

```text
64 张原图长 bag + Transformer 上下文建模 + label-wise attention + label graph
```

因此两者最关键的区别不是 backbone 或标签图，而是：

1. TASK2 使用更多实例，`64` 对比 TASK1 的 `16`。
2. TASK2 在 pooling 前加入 Transformer long-context encoder。
3. TASK2 的 `no_roi` 版本不使用 ROI crop，只使用原图实例。
4. TASK2 数据切分更严格，按患者分组，并且当前配置启用训练集类别平衡。

