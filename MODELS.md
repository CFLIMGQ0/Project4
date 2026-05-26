# 模型结构说明

## exp_8：train_007_exp8_mm_ablation_title_operation

### 基本定位

`train_007_exp8_mm_ablation_title_operation` 是 `exp_8` 当前默认采用的结构化短字段多模态模型配置。它使用 `exp8_structured_late_gate_mil` 作为基础模型，在 64 张原图 Long-MIL 图像分支上加入 `reportTitle` 和 `operationValue` 两个结构化字段，并通过 label-wise gated late fusion 融合。

该配置来自 `exp_8_mm_ablation` 字段消融实验中的第 7 组：

```text
图像 + reportTitle + operationValue
```

历史测试结果：

| 实验目录 | 输入 | best_epoch | macro_f1 | micro_f1 | macro_auc | macro_ap |
|---|---|---:|---:|---:|---:|---:|
| `train_007_exp8_mm_ablation_title_operation` | image + `reportTitle` + `operationValue` | 15 | 0.8841 | 0.8824 | 0.9402 | 0.9274 |

### 输入

| 输入类型 | 内容 | 说明 |
|---|---|---|
| 图像 | 每个检查最多 64 张原图 | 不启用 ROI |
| 结构化字段 | `reportTitle` | 报告标题/检查场景字段，类别 embedding |
| 结构化字段 | `operationValue` | 操作/检查类型字段，类别 embedding |
| 禁用字段 | `watchResult` | 标签来源字段，禁止作为输入 |
| 禁用字段 | `watch`、`suggest`、`specimen` | 本配置不使用长文本报告字段 |

### 图像分支

图像分支继承 Long-MIL 风格的检查级多实例学习流程：

```text
images [B, N, C, H, W]
  -> ConvNeXt-Tiny backbone
  -> instance features [B, N, 512]
  -> time position encoding
  -> Transformer context encoder
  -> MIL pooling
  -> label-wise image embeddings [B, L, 512]
  -> LabelHypergraphReasoner
```

关键参数：

| 参数 | 值 |
|---|---:|
| `backbone_name` | `convnext_tiny` |
| `feature_dim` | 512 |
| `attn_dim` | 256 |
| `hidden_dim` | 1024 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `encoder_chunk_size` | 16 |
| `label_graph_type` | `label_hypergraph` |
| `label_hypergraph_edges` | 2 |

### 结构化字段编码

结构化分支由 `StructuredFieldEncoder` 完成。当前配置只启用：

```yaml
structured_fields:
  - reportTitle
  - operationValue
```

两个字段均按类别字段处理：

```text
reportTitle      -> category embedding [64]
operationValue   -> category embedding [64]
field mask       -> 控制缺失字段和未启用字段
weighted average -> pooled structured field embedding
MLP fuse         -> structured embedding [B, 512]
```

结构化分支正则：

| 参数 | 值 | 作用 |
|---|---:|---|
| `structured_field_embed_dim` | 64 | 单个结构化字段 embedding 维度 |
| `structured_dropout` | 0.2 | 训练时随机丢弃部分结构化字段 |
| `modality_dropout` | 0.15 | 训练时随机置空整个结构化模态 |
| `structured_gate_l1_weight` | 0.001 | 抑制 gate 过大，避免模型过度依赖结构化字段 |

### Label-wise Gated Late Fusion

融合发生在图像分支得到每个标签的 label embedding 之后。结构化向量先投影为每个标签一份结构化 label embedding：

```text
structured_embed [B, 512]
  -> Linear(512, L * 512)
  -> structured_label_embeds [B, L, 512]
```

然后每个标签单独计算一个 gate：

```text
gate_l = sigmoid(MLP([image_label_embed_l, structured_label_embed_l]))
fused_l = image_label_embed_l + gate_l * structured_label_embed_l
logit_l = classifier_l(fused_l)
```

其中：

| 符号 | 含义 |
|---|---|
| `image_label_embed_l` | 图像分支为第 `l` 个标签得到的表示 |
| `structured_label_embed_l` | 结构化字段为第 `l` 个标签生成的表示 |
| `gate_l` | 第 `l` 个标签使用结构化信息的强度 |
| `fused_l` | 融合后的标签表示 |

`gate_l` 越接近 0，模型越依赖图像；越接近 1，模型越依赖结构化字段。

### 损失函数

主损失继承 TASK2 多标签分类设置：

```text
classification_loss = asymmetric loss
```

额外加入结构化 gate 的 L1 正则：

```text
structured_gate_l1 = mean(gates)
loss = classification_loss + 0.001 * structured_gate_l1
```

该正则的目的不是禁止结构化字段，而是防止 `reportTitle` 和 `operationValue` 这类流程相关字段完全压过图像分支。

### 当前默认训练配置

当前 `configs/task2/train.yaml` 已将普通 `python train.py` 默认切换为该配置：

```yaml
enabled_models:
  - exp8_structured_late_gate_mil

auto_exp_8_mm_ablation: false
```

普通训练会自动套用 `train_007` 的关键运行参数：

| 参数 | 值 |
|---|---:|
| `seed` | 2026 |
| `batch_size` | 1 |
| `eval_batch_size` | 1 |
| `grad_accum_steps` | 4 |
| `train_max_instances` | 64 |
| `eval_max_instances` | 64 |
| `train_max_batch_instances` | 128 |
| `eval_max_batch_instances` | 128 |
| `random_instance_dropout` | 0.0 |
| `roi_enabled` | false |

### 风险说明

`reportTitle` 和 `operationValue` 可以提升测试指标，但它们可能包含检查场景、术式路径或治疗流程信息。因此该模型适合作为当前 `exp_8` 的性能最优配置，但在论文或正式报告中应明确说明：

- 它不是纯图像模型。
- 它不是最低风险的结构化字段组合。
- `operationValue` 尤其可能存在检查路径代理风险。
- 需要结合置乱实验和去字段消融解释其收益来源。
