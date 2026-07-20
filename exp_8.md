# exp_8 多模态融合实验计划

`exp_8` 面向 TASK2 胃镜检查级三标签多标签分类，目标是在 `exp_7` 中的 `exp6_long_mil_64_no_roi` 基础上加入多模态信息，系统比较不同融合方式对 `macro_f1`、`macro_auc`、`subset accuracy` 和三类标签稳定性的影响。

本轮只研究多模态融合，不把 ROI 作为主变量。ROI 相关实验仍归入 `exp_7` 或后续 ROI 专项。

## 当前默认落地方案

当前 `configs/task2/train.yaml` 已开启 `auto_exp_8: true`。也就是说，直接在 `src` 目录运行：

```bash
python train.py
```

会一键批量运行 `exp_8` 五个主实验：

```text
exp8_mm_struct_late_gate
exp8_mm_label_proto_graph
exp8_mm_text_contrast_distill
exp8_mm_watch_cross_attn
exp8_mm_text_guided_top64_align
```

其中第一个结构化 late-gate 实验使用 `exp_8_mm_ablation` 中非置乱正式候选最优的字段组合：

```text
image + reportTitle + age + sex + operationValue
```

五个实验都使用 64 张原图作为图像底座，不把 ROI 作为本轮变量。当前默认不再批量运行 `exp_8_mm_ablation` 的 17 组字段消融；如需重跑字段消融，需要把 `auto_exp_8` 改为 `false`，再把 `auto_exp_8_mm_ablation` 改为 `true`。

`exp_8_mm_ablation` 中按 `macro_f1` 最高的是 `train_017_exp8_mm_ablation_shuffle_operation_train`，但它属于置乱审计实验，不适合作为正式 `exp_8` 默认输入。非置乱正式候选中，`train_011_exp8_mm_ablation_all_without_hp` 表现最好，因此当前将其固化为 `exp_8` 默认结构化 late-gate 配置。

| 来源实验 | 输入 | macro_f1 | micro_f1 | best_epoch | 说明 |
|---|---|---:|---:|---:|---|
| `train_001_exp8_mm_ablation_image_baseline` | image | 0.8719 | 0.8701 | 26 | 结构化字段消融的图像基线 |
| `train_011_exp8_mm_ablation_all_without_hp` | image + `reportTitle` + `age` + `sex` + `operationValue` | 0.8823 | 0.8800 | 10 | 当前 exp_8 结构化 late-gate 主实验复用的非置乱正式候选 |
| `train_017_exp8_mm_ablation_shuffle_operation_train` | image + 全字段，训练/验证/测试置乱 `operationValue` | 0.8829 | 0.8810 | 10 | 审计实验，不作为正式默认输入 |

注意：`reportTitle` 和 `operationValue` 仍属于流程相关字段，适合作为当前性能最优的 `exp_8` 主模型配置，但论文表述中必须说明其检查路径代理风险。若需要最保守的低风险可部署结论，应单独汇报 `image + age + sex`，不要把流程字段收益解释成纯图像证据收益。

## 一、基础设置

### 1.1 继承基线

| 项目 | 设置 |
|---|---|
| 基础实验 | `exp6_long_mil_64_no_roi` |
| 基础模型 | `long_mil` |
| 图像输入 | 每个检查最多 `64` 张原图 |
| ROI | 不启用 |
| 任务 | TASK2 三标签多标签分类 |
| 标签 | `label_esophageal_smt`、`label_esophageal_mucosal_or_tumor`、`label_gastritis` |
| 数据划分 | 继续使用 patient-level split |
| 主指标 | `macro_f1`，同时记录 `micro_f1`、`macro_auc`、`subset accuracy`、`hamming loss`、每标签 F1/AUC/PR-AUC |

建议输出目录：

```text
outputs/train_runs/task2/exp_8/
└── train_011_exp8_mm_ablation_all_without_hp/
    ├── config.yaml
    ├── log.csv
    ├── test_result.csv
    ├── test_report.csv
    ├── field_audit.csv
    ├── structured_metadata.json
    └── checkpoints/
```

该目录名固定复用原 `exp_8_mm_ablation` 第 11 组实验名，便于把当前 `exp_8` 主实验和非置乱正式候选最优字段组合对应起来。每个 `train_dir` 必须保存 `config.yaml`、`log.csv`、`test_result.csv`、`test_report.csv`、`checkpoints/` 和多模态字段审计文件。

### 1.2 可用模态与当前代码现状

源报告表 `/home/Lim/Project4/datasets/valid_dicts_report_for task2.csv` 中可用字段包括：

| 模态 | 字段示例 | 建议用途 | 泄漏级别 |
|---|---|---|---|
| 图像 | `exam_dir` 下的胃镜图片 | 主输入 | 无 |
| 基础结构化 | `age`、`sex`、`hp`、`score` | 可部署 late fusion 或缺失审计 | 低到中 |
| 报告标题 | `reportTitle` | 区分胃镜/无痛胃镜等检查类型 | 低 |
| 操作类型 | `operationValue` | 区分普通检查、超声内镜、内镜治疗等流程 | 低到中 |
| 检查所见 | `watch` | 报告辅助、多模态 teacher、训练期蒸馏 | 中到高 |
| 活检/标本 | `specimen` | 活检区域弱监督、报告辅助 | 中 |
| 建议 | `suggest` | 可能含诊断性提示，只做消融或上限 | 高 |
| 最终诊断 | `watchResult` | 标签来源，只做泄漏审计，禁止作为多模态输入 | 极高 |

当前多模态候选键限定为 `reportTitle`、`age`、`sex`、`hp`、`operationValue`、`specimen`、`score`、`suggest`、`watch`。当前 datalist 已保留 `reportTitle`、`hp`、`specimen`、`watch`，但 `age`、`sex`、`operationValue`、`score`、`suggest` 需要在 datalist 构建或 dataset 回连源报告表时补齐。`exp_8` 落地时需要新增：

- datalist 保留 `age`、`sex`、`operationValue`、`score`、`suggest` 等本轮有效字段。
- `MILBagDataset.__getitem__` 返回 `structured_features`、`structured_mask`、`text_inputs` 或离线 `text_embeddings`。
- `mil_collate_fn` 对结构化张量和文本 embedding 做 padding/stack。
- trainer 根据模型 `forward` 参数自动传入多模态字段。
- 所有离线文本 embedding、teacher 预测、字段扫描脚本都加 `tqdm` 进度条，并保存字段来源清单。

### 1.3 数据泄漏原则

`TASK2` 的标签由 `watchResult` 派生，因此多模态实验必须分开汇报：

| 类型 | 推理输入 | 是否可作为正式主结果 |
|---|---|---|
| 严格可部署 | 图像 + 诊断前结构化字段 + 报告标题 | 可以 |
| 固定语义增强 | 图像 + 固定标签文本原型；不输入个体报告诊断文字 | 可以，需单独说明 |
| 训练期蒸馏 | 训练用文本或 teacher，测试只输入图像 | 可以，归入 image-only student |
| 报告辅助 | 测试输入 `watch` 或 `specimen` | 可以作为“图像+报告所见”任务，不与 image-only 直接排名 |
| 高风险文本实验 | 测试输入 `watch` 原文或强诊断 `suggest` | 只能作为报告辅助或上限参考，不作为 image-only 结论 |

所有 `test_result.csv` 汇总必须新增字段：

```text
modality_level: strict_deploy | fixed_proto | train_time_distill | report_assist | upper_bound
modality_fields: none | reportTitle | age | sex | hp | operationValue | specimen | score | suggest | watch
inference_inputs: image | image+structured | image+text | image_only_student
leakage_note: 中文说明
```

## 二、实验总表

本轮做 6 个实验，均以 `exp6_long_mil_64_no_roi` 的 64 原图 Long-MIL 为图像底座。

| 序号 | 实验名 | 融合类型 | 推理输入 | 泄漏级别 | 目的 |
|---:|---|---|---|---|---|
| 1 | `exp8_mm_struct_late_gate` | 结构化信息 label-wise gated late fusion | 图像 + 结构化字段 | 低 | 建立最稳的可部署多模态基线 |
| 2 | `exp8_mm_label_proto_graph` | 标签文本原型 + label graph 语义约束 | 图像 + 固定标签原型 | 低 | 强化三标签语义和标签关系 |
| 3 | `exp8_mm_text_contrast_distill` | 图文对比预训练 + image-only 微调 | 测试只输入图像 | 训练期蒸馏 | 让图像分支吸收报告语义 |
| 4 | `exp8_mm_watch_cross_attn` | 检查所见文本与图像 token cross-attention | 图像 + `watch` | 中到高 | 评估报告所见与图像联合判别上限 |
| 5 | `exp8_mm_vlm_teacher_distill` | 医学 VLM/LLM teacher 软标签蒸馏 | 测试只输入图像 | 训练期蒸馏或上限 | 用外部多模态知识增强 image-only student |
| 6 | `exp8_mm_text_guided_top64_align` | 文本引导 Top-64 选图 + 图文实例/局部对齐 | 图像 + 分阶段多模态字段 | 中到高 | 用文本选择关键图像，并解释文本内容对应哪些图像证据 |

## 三、实验 1：结构化信息 label-wise gated late fusion

### 3.1 核心思路

在 `long_mil` 图像分支之外加入结构化短字段，使用 label-wise gate 控制每个标签对结构化信息的依赖。该实验分为保守低风险口径和当前性能候选口径：保守口径只使用 `age`、`sex` 等低风险字段；当前 `exp_8` 默认口径根据 `exp_8_mm_ablation` 结果加入 `reportTitle` 和 `operationValue`，但必须标注流程字段风险。

### 3.2 输入字段

保守低风险版使用：

| 字段 | 处理方式 |
|---|---|
| `age` | 连续变量标准化，缺失值置 0，同时加缺失 mask |
| `sex` | embedding，缺失单独一类 |

当前 `exp_8` 默认性能候选使用：

| 字段 | 处理方式 | 风险说明 |
|---|---|---|
| `reportTitle` | 简单文本类别 embedding，不做完整诊断文本解析 | 可能携带检查类型/路径先验 |
| `age` | 连续变量标准化，缺失值置 0，同时加缺失 mask | 低风险人口学字段 |
| `sex` | embedding，缺失单独一类 | 低风险人口学字段 |
| `operationValue` | 类别 embedding，低频类别合并 | 可能携带操作路径或术式选择代理信息 |

当前默认不使用 `hp`、`specimen`、`score`、`suggest`、`watch`、`watchResult`。其中 `watchResult` 是标签来源，始终禁止作为模型输入。

### 3.3 模型结构

```text
64 张原图
  -> Long-MIL
  -> label-wise image embeddings [B, 3, D]

结构化字段
  -> tabular encoder
  -> structured embedding [B, D]

融合：
gate_l = sigmoid(MLP([image_label_embed_l, structured_embed]))
fused_l = image_label_embed_l + gate_l * Linear(structured_embed)
logit_l = classifier_l(fused_l)
```

结构化 encoder 第一版用 MLP + categorical embedding。若第一版有效，再参考 TabTransformer/FT-Transformer 改成类别字段自注意力。

### 3.4 损失与配置

| 项目 | 建议 |
|---|---|
| 主损失 | 继承 `asymmetric` |
| gate 正则 | `lambda_gate_l1 = 0.001`，避免结构化信息压过图像 |
| structured dropout | `0.2`，训练时随机置空部分结构化字段 |
| 模态 dropout | `p=0.15`，部分 batch 只用图像，提升缺失鲁棒性 |

推荐配置名：

```yaml
experiment_dir_name: exp_8
run_dir_prefix: exp8_mm_struct_late_gate
train_max_instances: 64
eval_max_instances: 64
roi_enabled: false
multimodal:
  enabled: true
  mode: structured_late_gate
  structured_fields:
    - reportTitle
    - age
    - sex
    - operationValue
  modality_level: strict_deploy
```

### 3.5 预期与判读

- 若 `macro_f1` 和 `subset accuracy` 稳定提升，说明结构化字段能补充图像不确定性，优先纳入最终多模态基线。
- 若提升很小但训练稳定性变好，可以继续作为后续阶段的低风险基础字段。
- 若验证集提升但测试集下降，需要检查字段分布是否与 patient split、科室或时间强相关。

## 四、实验 2：标签文本原型引导 label graph

### 4.1 核心思路

当前 `long_mil` 已有 label-wise embedding 和 label graph。该实验不输入个体报告文本，而是为三个标签构造固定中文医学文本原型，用文本 encoder 得到 label prototype，再约束图像 label embedding 与对应 prototype 对齐。

这属于低泄漏的固定语义增强：测试阶段仍只依赖图像和固定标签定义，不读取患者报告内容。

### 4.2 标签原型

初始原型建议：

| 标签 | 文本原型 |
|---|---|
| `label_esophageal_smt` | `食管黏膜下隆起`、`食管黏膜下肿物`、`食管SMT`、`食管隆起性病变` |
| `label_esophageal_mucosal_or_tumor` | `食管黏膜病变`、`食管肿物`、`食管占位`、`食管新生物`、`食管早癌可疑` |
| `label_gastritis` | `慢性胃炎`、`活动性胃炎`、`萎缩性胃炎`、`糜烂性胃炎`、`胆汁反流性胃炎` |

文本 encoder 第一版可以使用冻结中文医学 BERT 或通用中文 BERT，离线生成 prototype embedding。后续可替换为 BiomedCLIP/UniMed-CLIP 类医学 VLM 的 text encoder。

### 4.3 模型结构

```text
图像 label embedding [B, 3, D]
固定标签文本 prototype [3, K, D]

每个标签：
proto_l = attention_pool(label_prototypes_l)
image_label_embed_l 与 proto_l 做投影对齐

label graph：
graph_prior = cosine(proto_i, proto_j)
dynamic_graph = alpha * learned_graph + (1 - alpha) * graph_prior
```

### 4.4 损失

```text
loss = ASL(logits, labels)
     + lambda_proto * supervised_contrast(image_label_embed, text_label_proto, labels)
     + lambda_graph * MSE(dynamic_graph, graph_prior)
```

推荐超参：

| 参数 | 值 |
|---|---|
| `lambda_proto` | `0.02`、`0.05` |
| `lambda_graph` | `0.001` |
| prototype dropout | `0.1` |
| graph prior mix `alpha` | `0.7` |

推荐配置名：

```yaml
multimodal:
  enabled: true
  mode: label_proto_graph
  prototype_source: fixed_label_terms
  text_encoder: frozen_chinese_medical_bert
  modality_level: fixed_proto
```

### 4.5 预期与判读

- 如果 `label_esophageal_smt` 与 `label_esophageal_mucosal_or_tumor` 混淆下降，说明标签语义原型帮助模型区分“黏膜下隆起”和“黏膜病变/肿物”。
- 如果整体提升不明显但 attention evidence 更合理，可作为后续 teacher 蒸馏和解释性分析的辅助模块。
- 若 graph prior 过强导致 `label_gastritis` 被食管标签拖累，需要降低 `lambda_graph` 或只用 prototype contrast，不约束 graph。

## 五、实验 3：图文对比预训练 + image-only 蒸馏微调

### 5.1 核心思路

参考 MedCLIP、BiomedCLIP、UniMed-CLIP 这类医学图文对比学习思路，先让 `long_mil` 的 bag embedding 与报告文本 embedding 对齐，再回到三标签分类。最终测试阶段只输入图像，因此归入训练期蒸馏。

这个实验的重点不是让模型在测试时读报告，而是用报告所见帮助图像分支形成更有医学语义的表示。

### 5.2 文本构造

建议做两版文本，主结果用低风险版：

| 版本 | 文本字段 | 用途 |
|---|---|---|
| `safe_text` | `reportTitle + hp + operationValue + specimen + score`，不含 `watchResult` | 主训练期语义增强 |
| `finding_text` | `watch`，可选择 mask 三个标签关键词 | 报告所见增强，单独汇报 |

关键词 mask 示例：

```text
食管SMT -> [DISEASE]
食管黏膜下隆起 -> [DISEASE]
食管黏膜病变 -> [DISEASE]
胃炎 -> [DISEASE]
```

`watchResult` 不进入本实验。

### 5.3 训练流程

1. 离线生成文本 embedding，保存到 `datasets/task_data/task2/text_embeddings/exp8_text_embeddings.parquet`。
2. 训练图文对比阶段：

```text
image_bag_embed = Long-MIL(images).bag_embed
text_embed = TextEncoder(text)
loss_itc = InfoNCE(image_bag_embed, text_embed)
```

3. 分类微调阶段：

```text
loss = ASL(logits, labels)
     + lambda_align * MSE(image_bag_embed, stopgrad(text_embed))
```

4. 测试阶段只输入图像。

### 5.4 推荐配置

```yaml
multimodal:
  enabled: true
  mode: text_contrast_distill
  text_fields:
    - reportTitle
    - hp
    - operationValue
    - specimen
    - score
  optional_text_fields:
    - watch
  mask_label_keywords: true
  inference_inputs: image
  modality_level: train_time_distill
  lambda_itc: 0.1
  lambda_align: 0.05
  temperature: 0.07
```

### 5.5 预期与判读

- 若 image-only 测试提升，说明文本预训练成功把报告语义迁移到了图像表示。
- 若训练期提升但测试期无提升，可能是文本 embedding 只记住报告模板，建议改用 `watch` 关键词 mask 或增强图像 crop/instance 对齐。
- 若 `macro_auc` 提升但 `macro_f1` 不升，优先做 per-label threshold 和温度校准。

## 六、实验 4：检查所见文本与图像 token cross-attention

### 6.1 核心思路

参考 ALBEF、BLIP 和 GLoRIA 的“先对齐、再融合”思想，把 `watch` 文本 token 与 64 张图像 instance token 做 cross-attention。相比简单拼接，cross-attention 能学习“哪段检查所见对应哪些图像证据”。

这个实验测试阶段输入 `watch`，因此它不是 image-only 主结果，而是“图像 + 检查所见”的报告辅助任务。

### 6.2 模型结构

```text
图像：
64 张原图 -> Long-MIL context encoder -> image tokens [B, N, D]

文本：
watch -> text encoder -> text tokens [B, T, D]

融合：
label queries [3, D]
label queries cross-attend image tokens
label queries cross-attend text tokens
gated fusion:
  fused_label_l = image_label_l
                + gate_text_l * text_label_l
                + gate_cross_l * cross_label_l
```

### 6.3 训练目标

```text
loss = ASL(fused_logits, labels)
     + lambda_image * ASL(image_only_logits, labels)
     + lambda_itm * image_text_matching_loss
     + lambda_align * token_alignment_loss
```

其中 `image_only_logits` 是辅助头，防止模型完全依赖文本。

### 6.4 推荐配置

```yaml
multimodal:
  enabled: true
  mode: watch_cross_attention
  text_fields:
    - watch
  image_tokens: long_mil_context_features
  fusion:
    type: label_query_cross_attention
    num_cross_layers: 2
    num_heads: 4
  aux_image_only_head: true
  modality_level: report_assist
  lambda_image: 0.5
  lambda_itm: 0.05
  lambda_align: 0.02
```

### 6.5 预期与判读

- 如果该实验明显超过实验 1 和实验 3，说明报告所见含有大量互补信息。
- 如果 `watch` 文本单独已经很强，则要在报告中明确这是报告辅助分类，不代表纯图像能力提升。
- 如果模型在测试集异常高，需要抽样检查 `watch` 是否直接包含 `watchResult` 或目标标签原词。

## 七、实验 5：医学 VLM/LLM teacher 多模态蒸馏

### 7.1 核心思路

参考 BLIP-2、LLaVA-Med 这类冻结视觉语言大模型加轻量适配的思路，用外部 teacher 生成软标签、难例权重或标签解释，然后把知识蒸馏回 `exp6_long_mil_64_no_roi` student。最终 student 测试只输入图像。

该实验分两版：

| 版本 | teacher 输入 | 汇报类型 |
|---|---|---|
| `teacher_safe` | attention top-k 图像 montage + `reportTitle/age/sex/hp/operationValue/specimen/score` | 训练期蒸馏 |
| `teacher_upper` | attention top-k 图像 montage + `watch/suggest` | upper bound，不作为正式结果 |

### 7.2 离线 teacher 产物

对每个检查保存：

```text
outputs/train_runs/task2/exp_8/teacher_cache/
├── teacher_predictions.parquet
├── teacher_prompts.jsonl
├── teacher_quality_audit.csv
└── sample_montages/
```

字段包括：

| 字段 | 说明 |
|---|---|
| `exam_dir` | 检查目录 |
| `teacher_prob_0..2` | 三标签 soft probability |
| `teacher_confidence` | teacher 置信度 |
| `teacher_rationale` | 简短中文解释，只用于审计，不输入 student |
| `teacher_input_level` | `safe` 或 `upper` |
| `prompt_version` | prompt 版本 |

批量生成 teacher 预测时必须使用进度条，并人工抽样检查至少 `50` 个样本。

### 7.3 Student 训练

```text
loss = ASL(student_logits, hard_labels)
     + lambda_kd * KL(student_prob / T, teacher_prob / T)
     + lambda_logit * MSE(student_logits, teacher_logits)
     + lambda_conf * confidence_weighted_bce(student_logits, teacher_prob)
```

推荐超参：

| 参数 | 值 |
|---|---|
| `lambda_kd` | `0.1`、`0.3` |
| `lambda_logit` | `0.05` |
| 蒸馏温度 `T` | `2`、`4` |
| teacher 置信度阈值 | `0.65` |
| low-confidence 样本 | 只用 hard label，不用 KD |

### 7.4 推荐配置

```yaml
multimodal:
  enabled: true
  mode: vlm_teacher_distill
  teacher_cache: outputs/train_runs/task2/exp_8/teacher_cache/teacher_predictions.parquet
  teacher_input_level: safe
  inference_inputs: image
  modality_level: train_time_distill
  kd_temperature: 2
  lambda_kd: 0.1
  lambda_logit: 0.05
```

### 7.5 预期与判读

- 如果 `teacher_safe` student 提升，说明外部多模态知识对图像分类有帮助，适合继续扩大 teacher ensemble。
- 如果只有 `teacher_upper` 有提升，说明收益主要来自诊断文本泄漏，不能写成正式多模态贡献。
- 如果 teacher 与 hard label 冲突较多，需要用置信度门控或只在验证集中表现稳定的标签上启用 KD。

## 八、实验 6：文本引导 Top-64 选图 + 图文空间映射 MIL

### 8.1 想法评价

这个实验方向值得做。它比简单 late fusion 更有解释性，也更贴合胃镜检查级 bag 的特点：一次检查可能有几十到上百张图，文本里通常已经描述了病灶部位、形态、活检位置或检查类型。如果能用多模态文本先帮助模型从全量图片里挑出最相关的 `64` 张，再学习“文本 token - 图像 instance - 图像局部区域”的对应关系，结果会比普通 `64` 张随机/均匀采样更容易解释。

但这个想法直接做成端到端“从所有图片中可微选择 64 张 + 文本局部空间映射”会比较重，主要难点有三个：

- 每个检查原始图片数量不固定，直接端到端 Top-K 会让显存和 batch 组织很不稳定。
- 文本字段里 `watch`、`suggest` 可能包含标签词，模型容易学成文本捷径，而不是图像-文本对应。
- 真正像素级空间映射需要 patch token、Grad-CAM 或 ROI/分割辅助，第一版不宜一上来追求精细到病灶边界。

因此建议第一版把目标改成更容易落地的两阶段方案：先做“文本引导的离线 Top-64 选图”，再在选出的 64 张图上做“图文 token 对齐 + 多标签分类”。空间映射第一版先做到文本片段对应图像 instance 和图像热力图，后续再升级到 ROI/patch 级。

### 8.2 推荐实验名

```text
exp8_mm_text_guided_top64_align
```

### 8.3 输入字段

该实验可以按 `multi-model.md` 的三阶段字段逐步做：

| 阶段 | 用于选图和融合的字段 | 泄漏级别 | 建议定位 |
|---|---|---|---|
| phase1 | `reportTitle`、`age`、`sex` | 低 | 低风险选图基线，预计文本信息较弱 |
| phase2 | phase1 + `hp`、`operationValue`、`specimen`、`score` | 低到中 | 推荐主实验，文本有临床上下文但泄漏相对可控 |
| phase3 | phase2 + `suggest`、`watch` | 中到高 | 报告辅助上限或训练期蒸馏，不能和 image-only 主结果混排 |

第一轮建议优先做 phase2。phase1 文本太弱，可能选图收益不明显；phase3 信息最强但泄漏风险也最高。

### 8.4 两阶段落地方案

第一阶段：离线文本引导选图。

```text
每个检查的所有图片
  -> 冻结图像 encoder
  -> image embeddings [N, D]

多模态文本字段
  -> 冻结文本 encoder
  -> text embedding [D] 或 text token embeddings [T, D]

计算：
score_i = cosine(image_embed_i, text_embed)
select top64 = MMR(score_i, diversity_i)
```

为了避免选出的 64 张图过于重复，建议不要只取相似度最高的 64 张，而是使用 MMR 或分段约束：

```text
final_score_i = alpha * text_image_similarity_i
              + beta * image_quality_i
              + gamma * temporal_coverage_i
              - delta * redundancy_i
```

推荐默认值：

| 参数 | 建议 |
|---|---|
| `alpha` | `1.0` |
| `beta` | `0.1`，可用清晰度/黑边比例/亮度异常等质量分 |
| `gamma` | `0.1`，保证检查序列前中后都有覆盖 |
| `delta` | `0.3`，降低重复图片 |
| `top_k` | `64` |

离线产物保存为：

```text
outputs/train_runs/task2/exp_8/text_guided_top64/
├── selected_images_phase1.csv
├── selected_images_phase2.csv
├── selected_images_phase3.csv
├── selection_scores.parquet
└── field_audit.json
```

批量扫描图片和生成 embedding 时必须加进度条。

第二阶段：在 Top-64 图像上训练图文对齐分类模型。

```text
Top-64 images
  -> Long-MIL context encoder
  -> image instance tokens [B, 64, D]

多模态文本
  -> text encoder
  -> text tokens [B, T, D]

label queries [3, D]
  -> cross-attend image tokens
  -> cross-attend text tokens
  -> 生成 label-wise fused embeddings
  -> 三标签 logits
```

### 8.5 空间映射解释方式

第一版建议做三层解释，不要一开始强求像素级精确定位：

| 层级 | 解释内容 | 实现方式 |
|---|---|---|
| 文本 token 到图片 | 哪些文本词/短语对应哪几张图 | cross-attention 权重 `text_token -> image_instance` |
| 标签到图片 | 每个标签主要看哪几张图 | label-wise attention top-k |
| 图片内热力图 | 选中图片内哪些区域支持预测 | Grad-CAM、attention rollout 或 ViT patch attention |

推荐每个测试样本导出：

```text
explain/
└── <exam_id>/
    ├── text_image_alignment.csv
    ├── label_top_images.csv
    ├── montage_top64.jpg
    └── heatmaps/
```

这样可以回答两个问题：

- 多模态文本中的哪些片段让模型选择了哪些图。
- 每个标签的预测主要由哪几张图和图内哪些区域支持。

### 8.6 训练损失

```text
loss = ASL(fused_logits, labels)
     + lambda_image * ASL(image_only_logits, labels)
     + lambda_text * ASL(text_only_logits, labels)
     + lambda_align * InfoNCE(image_instance_tokens, text_tokens)
     + lambda_sparse * attention_sparsity_loss
     + lambda_cons * KL(fused_prob, image_only_prob)
```

建议第一版关闭或弱化 `text_only_logits`，避免模型只读文本：

| 参数 | 建议值 |
|---|---|
| `lambda_image` | `0.5` |
| `lambda_text` | `0.1` 或第一版设为 `0` |
| `lambda_align` | `0.02` |
| `lambda_sparse` | `0.001` |
| `lambda_cons` | `0.05` |

### 8.7 推荐配置

```yaml
multimodal:
  enabled: true
  mode: text_guided_top64_align
  selection:
    enabled: true
    top_k: 64
    method: mmr_text_image_similarity
    cache_dir: outputs/train_runs/task2/exp_8/text_guided_top64
    image_encoder: frozen_long_mil_backbone
    text_encoder: frozen_chinese_text_encoder
    alpha_similarity: 1.0
    beta_quality: 0.1
    gamma_temporal: 0.1
    delta_redundancy: 0.3
  text_fields_phase1:
    - reportTitle
    - age
    - sex
  text_fields_phase2:
    - reportTitle
    - age
    - sex
    - hp
    - operationValue
    - specimen
    - score
  text_fields_phase3:
    - reportTitle
    - age
    - sex
    - hp
    - operationValue
    - specimen
    - score
    - suggest
    - watch
  fusion:
    type: label_query_text_image_cross_attention
    num_cross_layers: 2
    num_heads: 4
  explain:
    export_alignment: true
    export_top_images: true
    export_heatmaps: true
  modality_level: report_assist
```

### 8.8 最容易落地的修改版

建议把原始想法改成下面这个版本：

```text
不是一开始端到端学习“从所有图片中选 64 张”，
而是先离线用冻结 encoder + 文本相似度 + 多样性约束选 64 张；
训练时只输入这 64 张图和文本 token，
用 label-wise cross-attention 做多标签分类，
再导出文本 token 到图像 instance 的 attention 作为解释。
```

这样改有几个好处：

- 不改动现有 `Long-MIL` 的核心训练显存结构，仍然固定 `64` 张输入。
- 可以和 `exp6_long_mil_64_no_roi` 做公平对照：同样是 64 张图，只是采样策略从随机/均匀变为文本引导。
- 选图结果可以离线审计，便于判断是否真的选到了内镜关键图。
- 如果 phase3 使用 `watch/suggest` 后提升很大，也能通过关键词 mask 和 attention 导出检查是否存在文本捷径。

### 8.9 风险控制

- 必须同时跑 `random_64`、`uniform_64`、`text_guided_64` 三个采样对照，否则无法证明收益来自文本引导选图。
- phase3 的 `watch/suggest` 需要做关键词 mask 对照。
- 解释图只能作为模型证据可视化，不能直接当作真实病灶标注。
- 如果文本引导选择后 `label_gastritis` 下降，说明文本可能过度偏向局灶病变图，需要加入 temporal coverage 或保留部分均匀采样图。
- 建议 `64` 张中保留 `16` 张 uniform coverage 图、`48` 张 text-guided 图，避免全被文本相似度带偏。

## 九、推荐执行顺序

| 优先级 | 实验 | 原因 |
|---:|---|---|
| 1 | `exp8_mm_struct_late_gate` | 成本低、泄漏风险低，是正式多模态基线 |
| 2 | `exp8_mm_label_proto_graph` | 与现有 label graph 最贴合，不依赖个体报告文本 |
| 3 | `exp8_mm_text_contrast_distill` | 训练期利用报告语义，测试仍保持 image-only |
| 4 | `exp8_mm_watch_cross_attn` | 评估图像 + 检查所见的联合上限，但需严格标注任务类型 |
| 5 | `exp8_mm_text_guided_top64_align` | 与当前 64 图 Long-MIL 最贴合，能直接比较文本引导选图是否优于随机/均匀采样 |
| 6 | `exp8_mm_vlm_teacher_distill` | 潜力高但流程长，需要离线 teacher、审计和缓存 |

建议先完成实验 1 和实验 2，确认数据管线和多模态字段传递稳定后，再进入文本 encoder、cross-attention、文本引导选图和 teacher 蒸馏。

## 十、统一结果记录模板

每个实验的 `remark.txt` 至少包含：

```text
实验名：
基础模型：
图像输入：
额外模态：
推理输入：
泄漏级别：
是否使用 watchResult：
是否使用 watch：
是否使用 specimen：
是否训练期蒸馏：
主 checkpoint：
macro_f1：
micro_f1：
macro_auc：
subset_accuracy：
label_esophageal_smt_f1：
label_esophageal_mucosal_or_tumor_f1：
label_gastritis_f1：
相对 exp6_long_mil_64_no_roi 的变化：
失败/异常说明：
```

汇总表建议增加：

| 实验 | `macro_f1` | `macro_auc` | `subset_accuracy` | SMT F1 | 食管黏膜/肿物 F1 | 胃炎 F1 | 推理输入 | 泄漏级别 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `exp6_long_mil_64_no_roi` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image | none |
| `exp8_mm_struct_late_gate` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image+structured | strict_deploy |
| `exp8_mm_label_proto_graph` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image+fixed_proto | fixed_proto |
| `exp8_mm_text_contrast_distill` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image | train_time_distill |
| `exp8_mm_watch_cross_attn` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image+watch | report_assist |
| `exp8_mm_vlm_teacher_distill` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image | train_time_distill |
| `exp8_mm_text_guided_top64_align` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | image+text-guided-top64 | report_assist |

## 十一、落地代码建议

建议新增目录：

```text
exp_8/
├── __init__.py
├── models.py
├── text_encoder.py
├── structured.py
├── text_guided_selector.py
├── teacher_distill.py
└── preprocess_text_embeddings.py
```

建议模型注册：

| 模型名 | 类 |
|---|---|
| `exp8_mm_struct_late_gate` | `Exp8StructuredLateGateLongMILModel` |
| `exp8_mm_label_proto_graph` | `Exp8LabelProtoGraphLongMILModel` |
| `exp8_mm_text_contrast_distill` | `Exp8TextContrastDistillLongMILModel` |
| `exp8_mm_watch_cross_attn` | `Exp8WatchCrossAttentionLongMILModel` |
| `exp8_mm_vlm_teacher_distill` | `Exp8VLMTeacherDistillLongMILModel` |
| `exp8_mm_text_guided_top64_align` | `Exp8TextGuidedTop64AlignMILModel` |

最小实现顺序：

1. 改 datalist 和 dataset，保证结构化字段、文本字段能进入 batch。
2. 实现实验 1，验证多模态字段传递、缺失 mask、输出目录规范。
3. 实现固定标签原型离线 embedding 和实验 2。
4. 实现文本 embedding 缓存脚本和实验 3。
5. 实现 cross-attention 融合实验 4。
6. 实现文本引导 Top-64 选图缓存和实验 6。
7. 最后实现 teacher cache 与蒸馏实验 5。

## 十二、参考方法方向

- MedCLIP（https://arxiv.org/abs/2210.10163）：医学图文对比学习，强调医学图文弱配对和 false negative 问题，适合实验 3。
- BiomedCLIP（https://arxiv.org/abs/2303.00915）与 UniMed-CLIP（https://arxiv.org/abs/2412.10372）：医学领域视觉语言预训练，可作为文本 encoder 或图文对齐参考。
- GLoRIA（https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html）：医学图文 global-local 对齐，适合实验 4 的图像 instance 与报告 token 对齐。
- ALBEF（https://arxiv.org/abs/2107.07651）与 BLIP（https://arxiv.org/abs/2201.12086）：先图文对齐再融合，并用动量蒸馏或 caption/filter 缓解噪声图文监督。
- BLIP-2（https://arxiv.org/abs/2301.12597）与 LLaVA-Med（https://arxiv.org/abs/2306.00890）：冻结大模型并通过轻量连接模块适配，可作为实验 5 的 teacher 思路。
- TabTransformer（https://arxiv.org/abs/2012.06678）与 FT-Transformer（https://arxiv.org/abs/2106.11959）：结构化字段建模参考，适合实验 1 后续升级。
