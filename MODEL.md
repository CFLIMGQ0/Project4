# 模型结构说明

## 当前重点模型

当前项目重点模型是 `gastro_label_graph_mil`，用于胃镜三标签多标签分类。

对应代码位置：

- `model/gastro_label_graph_mil/network.py`
- `model/gastro_label_graph_mil/modules.py`

## 整体结构

`gastro_label_graph_mil` 可以拆成 4 个部分：

1. 实例编码器 `InstanceEncoder`
2. 多标签注意力聚合 `MultiLabelAttentionMIL`
3. 标签关系建模模块 `LabelGraphReasoner`
4. 标签级分类头 `classifiers`

整体流程为：

`图像 -> 实例特征 -> 多标签 attention 聚合 -> 标签图传播 -> 标签级分类`

## 模块说明

### 1. `InstanceEncoder`

职责：

- 对 bag 内每张胃镜图像做共享视觉编码；
- 将 backbone 输出映射到统一的 `feature_dim`。

组成：

- `build_backbone(...)`
- 两层 `Linear + GELU + Dropout` 投影

输入输出：

- 输入：`images [B, N, C, H, W]`
- 输出：`instance_features [B, N, D]`

## 2. `MultiLabelAttentionMIL`

职责：

- 对 3 个标签分别做 attention 聚合；
- 让每个标签从同一组实例里选取自己最相关的图像证据。

输入输出：

- 输入：`instance_features [B, N, D]`
- 输出：
  - `bag_embeds [B, L, D]`
  - `attention [B, L, N]`

这里 `L = 3`，分别对应：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

## 3. `LabelGraphReasoner`

这是当前模型最核心的模块。

职责：

- 为每个标签维护一个可学习的 `label token`；
- 根据标签 token 的相似度构造标签图；
- 对标签级 bag 表征做一次图传播；
- 将传播后的上下文信息回注到每个标签表征中。

关键公式：

- `A = softmax(E E^T / sqrt(d))`
- `Z_prop = A @ Z`
- `Z_refined = Z + phi([Z ; Z_prop])`

含义：

- `E` 是标签 token；
- `A` 是标签之间的相关矩阵；
- `Z` 是原始标签级 bag 表征；
- `Z_refined` 是融合标签关系后的新表征。

这个设计的目的不是让 3 个标签完全独立预测，而是让它们在预测前先共享一部分结构化上下文。

## 4. 标签级分类头

职责：

- 对每个标签的 refined 表征分别输出 1 个 logit。

实现方式：

- `nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])`

最终输出：

- `logits [B, 3]`

## 模块化目录

当前 `model/` 目录只围绕这个重点模型展开：

```text
model/
├── common/
│   ├── backbones.py
│   └── pooling.py
└── gastro_label_graph_mil/
    ├── modules.py
    └── network.py
```

其中：

- `common/backbones.py` 负责 backbone 构建；
- `common/pooling.py` 负责通用 MIL 聚合；
- `gastro_label_graph_mil/modules.py` 放该模型专属子模块；
- `gastro_label_graph_mil/network.py` 放整网拼装逻辑。

## 配置说明

训练参数以 `configs/train.yaml` 为准；`configs/model.yaml` 仅记录 `gastro_label_graph_mil` 的结构参数。

每次运行 `train.py` 后，每个训练目录下会额外生成：

- `config.yaml`：记录该次训练使用的模型结构参数与训练参数；
- `checkpoints/best_macro_f1.ckpt`
- `checkpoints/best_micro_f1.ckpt`
- `checkpoints/best_val_loss.ckpt`

更完整的训练输出目录说明见 `README.md` 中的“训练输出目录约定（新增）”。

## Baseline模型

- `gastro_baseline`：位于 `baseline/gastro_baseline/`，作为胃镜三标签多标签分类的基础 MIL 对照模型。
- `colonoscopy_baseline`：位于 `baseline/colonocopy_baseline/`，作为肠镜二分类的基础 MIL 对照模型。
