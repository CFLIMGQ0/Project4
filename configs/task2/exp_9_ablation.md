# exp_9_ablation 实验说明

`exp_9_ablation` 以 `exp8_mm_watch_cross_attn` 为完整模型基准，围绕 watch 报告辅助分支做消融。当前 `configs/task2/train.yaml` 已开启 `auto_exp_9_ablation: true`，在 `src` 目录直接运行：

```bash
python train.py
```

会按顺序运行 17 个实验，结果默认写入：

```text
/home/Lim/Project4/outputs/train_runs/task2/exp_9_ablation/
```

## 完整模型基准

完整模型沿用 `exp8_mm_watch_cross_attn`：

- 图像分支：ConvNeXt-Tiny 编码单张图像，检查级别固定采样多张图像。
- 序列上下文：图像特征加入位置编码，再经过 Transformer context encoder。
- 标签关系：默认使用 `label_hypergraph`。
- watch 文本分支：将 watch 报告文本编码为 token embedding。
- 融合方式：用每个标签的图像表征作为 query，对 watch 文本 token 做 cross-attention，再用 gate 控制文本证据注入。
- 辅助约束：保留 image-only 辅助预测损失，默认 `image_aux_weight: 0.5`。

## 实验列表

### 1. 图像数量消融

保持完整模型结构不变，只改变一次检查最多采样的图像数量：

- `exp9_watch_instances_16`
- `exp9_watch_instances_32`
- `exp9_watch_instances_48`
- `exp9_watch_instances_64`
- `exp9_watch_instances_80`
- `exp9_watch_instances_96`

### 2. 去掉位置编码和 Transformer context

在相同图像数量设置下，去掉检查内位置编码和 Transformer context encoder，直接用原始图像实例特征进入 MIL 聚合：

- `exp9_watch_no_context_instances_16`
- `exp9_watch_no_context_instances_32`
- `exp9_watch_no_context_instances_48`
- `exp9_watch_no_context_instances_64`
- `exp9_watch_no_context_instances_80`
- `exp9_watch_no_context_instances_96`

### 3. 去掉 watch 文本

- `exp9_watch_no_text`

只保留图像长 MIL 分支，不输入 watch 文本，用于估计 watch 文本整体贡献。

### 4. label_graph 替换 hypergraph

- `exp9_watch_label_graph`

保留完整 watch cross-attention 融合，但将标签关系模块从 `label_hypergraph` 改为普通 `learnable` label graph。

### 5. 去掉 cross-attention

- `exp9_watch_no_cross_attn_pool_fusion`

去掉 label-wise cross-attention，改为先池化 watch 文本 embedding，再与每个标签的图像表征做 late fusion。

### 6. 去掉 gate

- `exp9_watch_cross_attn_no_gate`

保留 watch cross-attention，但去掉 gate，直接把文本表征加到对应标签的图像表征上。

### 7. 去掉 image_aux

- `exp9_watch_cross_attn_no_image_aux`

保留完整 watch cross-attention 结构，但将 `image_aux_weight` 置为 0，用于观察 image-only 辅助损失的贡献。

## 运行筛选

如果只想跑其中几个实验，可以使用 `--models` 指定逗号分隔的实验名，例如：

```bash
python train.py --models exp9_watch_instances_64,exp9_watch_cross_attn_no_gate
```
