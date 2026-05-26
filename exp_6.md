# exp_6 实验计划

## 一、实验背景与总目标

`exp_6` 面向 TASK2 胃镜检查级三标签多标签分类，目标是在当前最佳图像模型基础上，继续探索能够提升三标签整体性能、尾标签稳定性和多标签联合预测一致性的方法。

当前正式对比对象如下：

| 实验 | 训练目录 | 输入形式 | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Acc | Hamming Loss | Kappa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `exp_4 long_mil` | `outputs/train_runs/task2/exp_4/auto_exp_4/train_006_long_mil` | 最多 `128` 张原始胃镜图 | `0.8759` | `0.8723` | `0.9289` | `0.9139` | `0.7436` | `0.1265` | `0.7473` |
| `exp_5 roi_long_mil` | `outputs/train_runs/task2/exp_5/auto_exp_5/train_001_roi_long_mil` | 最多约 `64` 张原图 + `64` 张 SAM2 ROI crop | `0.8741` | `0.8712` | `0.9287` | `0.9112` | `0.7162` | `0.1288` | `0.7430` |

每标签 F1 变化：

| 标签 | `exp_4 long_mil` F1 | `exp_5 roi_long_mil` F1 | 初步判断 |
|---|---:|---:|---|
| `label_esophageal_smt` | `0.8465` | `0.8599` | ROI 对局灶性 SMT 有帮助迹象 |
| `label_esophageal_mucosal_or_tumor` | `0.8440` | `0.8429` | 基本持平 |
| `label_gastritis` | `0.9373` | `0.9194` | ROI 可能削弱全局黏膜上下文 |

核心判断：`exp_5` 没有明显超过 `exp_4`，并且 subset accuracy 从 `0.7436` 降到 `0.7162`。因此 `exp_6` 需要先做公平输入数量对照，再分别探索 ROI 弱监督改进和多模态增强，两条路线相互独立。

主比较指标建议仍以 `macro_f1` 为主，同时报告 `micro_f1`、`macro_roc_auc`、`macro_pr_auc`、`subset_accuracy`、`hamming_loss`、`kappa` 和三标签各自的 F1/recall/precision。

### 当前一键运行矩阵

当前代码已实现 `auto_exp_6`，默认执行下面 12 个可落地实验；运行方式为：

```bash
python train.py
```

| 自动实验名 | 基础模型 | 输入设置 | 目的 |
|---|---|---|---|
| `exp6_long_mil_64_no_roi` | `long_mil` | `64` 原图，无 ROI | 第一部分公平对照 |
| `exp6_roi_mix_64_32` | `long_mil` | `64` 原图 + `32` ROI | 对照 ROI 少量补充 |
| `exp6_roi_mix_64_64` | `long_mil` | `64` 原图 + `64` ROI | 复核 `exp_5` 风格输入 |
| `exp6_roi_context_128_16` | `long_mil` | `128` 原图 + `16` ROI | 保留完整上下文后追加 ROI |
| `exp6_roi_context_128_32` | `long_mil` | `128` 原图 + `32` ROI | 推荐 ROI 追加对照 |
| `exp6_roi_context_128_64` | `long_mil` | `128` 原图 + `64` ROI | 检查 ROI 数量增大是否引入噪声 |
| `exp6_roi_dual_128_16` | `exp6_dual_stream_long_mil` | `128` 原图 + `16` ROI | 双路 ROI 融合 |
| `exp6_roi_dual_128_32` | `exp6_dual_stream_long_mil` | `128` 原图 + `32` ROI | 推荐双路主实验 |
| `exp6_roi_dual_128_64` | `exp6_dual_stream_long_mil` | `128` 原图 + `64` ROI | 双路大 ROI 数量实验 |
| `exp6_roi_filter_96_32` | `exp6_dual_stream_long_mil` | `96` 原图 + `32` 高分 ROI | ROI 质量过滤 |
| `exp6_roi_filter_128_32` | `exp6_dual_stream_long_mil` | `128` 原图 + `32` 高分 ROI | 保留上下文的 ROI 过滤 |
| `exp6_roi_cons_128_32` | `exp6_dual_stream_long_mil` | `128` 原图 + `32` ROI | 双路融合 + 预测一致性正则 |

第三部分多模态方法仍是后续研究设计：需要先明确可用文本字段、结构化字段和泄漏控制策略，再新增对应数据读取与模型分支；当前一键运行矩阵不把诊断结论文本作为测试输入。

## 二、Long-MIL 输入数量 64 对照实验

### 2.1 为什么做

`exp_4 long_mil` 使用最多 `128` 张原图，是当前最佳图像模型。`exp_5 roi_long_mil` 虽然总 instance 数仍为 `128`，但实际变成最多约 `64` 张原图加 `64` 张 ROI crop。若不做 `64` 张原图对照，就无法判断 `exp_5` 的结果到底来自 ROI 增益，还是来自原图上下文减少后的抵消结果。

该实验是 `exp_6` 的第一优先级，也是后续 ROI 改进方案的公平基线。

### 2.2 实验设计

| 项目 | 设置 |
|---|---|
| 推荐实验名 | `exp6_long_mil_64_no_roi` |
| 基础模型 | `long_mil` |
| 对照来源 | `exp_4/auto_exp_4/train_006_long_mil` |
| 输入 | 最多 `64` 张原始胃镜图 |
| ROI | 不启用 |
| 采样策略 | `train_sampling_strategy: uniform`，`eval_sampling_strategy: uniform` |
| 类别重平衡 | 保持启用，沿用 `multilabel_minority_oversample` |
| 训练轮数 | `max_epochs: 30` |
| batch 设置 | `batch_size: 1`，`eval_batch_size: 1`，`grad_accum_steps: 4` |
| 优化器 | `adamw` |
| 学习率 | `lr: 2.0e-05` |
| 权重衰减 | `weight_decay: 0.02` |
| warmup | `warmup_ratio: 0.2` |
| 损失函数 | `loss_name: asymmetric` |
| 混合精度 | `amp: true` |

建议关键配置：

```yaml
experiment_dir_name: exp_6
run_dir_prefix: exp6_long_mil_64_no_roi
enabled_models:
  - long_mil
train_max_instances: 64
eval_max_instances: 64
train_max_batch_instances: 128
eval_max_batch_instances: 128
train_sampling_strategy: uniform
eval_sampling_strategy: uniform
auto_exp_5_roi:
  enabled: false
```

### 2.3 对比与判断逻辑

| 对比 | 解释 |
|---|---|
| `exp6_long_mil_64_no_roi` 明显低于 `exp_4 long_mil` | `exp_4` 依赖 `128` 张输入，长序列上下文确实重要 |
| `exp6_long_mil_64_no_roi` 接近 `exp_4 long_mil` | `long_mil` 对输入数量不敏感，后续应优先优化训练稳定性和证据选择 |
| `exp6_long_mil_64_no_roi` 低于 `exp_5 roi_long_mil` | ROI crop 对减少原图后的信息缺口有补偿 |
| `exp6_long_mil_64_no_roi` 接近或优于 `exp_5 roi_long_mil` | 当前 ROI 方案收益有限，SAM2 crop 或混合方式可能引入噪声 |

## 三、基于 exp_5 的弱监督 ROI 改进方案

### 3.1 当前 ROI 方案可能不明显的原因

`exp_5 roi_long_mil` 的核心变化是把 SAM2 ROI crop 作为额外 instance 加入 bag，但它不是在 `128` 张原图基础上额外增加 ROI，而是让 ROI 占用了原图名额。当前效果不明显，可能来自以下问题：

- ROI crop 替代了部分原图上下文，尤其会影响 `label_gastritis` 这类依赖全局黏膜状态的标签。
- SAM2 生成的是显著区域或可分割区域，不等于真实病灶区域；高质量 mask 也可能只是皱襞、反光、泡沫、器械边缘或正常解剖结构。
- ROI 与原图简单混合，模型没有明确区分全局证据和局部证据，也没有按标签控制 ROI 权重。
- 训练标签只有检查级标签，缺少 ROI 级或 instance 级弱监督，ROI 分支可能学到非因果相关。
- `exp_4` 和 `exp_5` 都存在 train/val loss 差距偏大、验证损失后期反弹的问题，ROI 噪声可能进一步放大过拟合。

下面 5 个方案可以单独执行，也可以组合。优先目标是提升 `macro_f1` 和 subset accuracy，同时保住 `label_gastritis`，并进一步提升 `label_esophageal_smt` 与 `label_esophageal_mucosal_or_tumor`。

### 3.2 方案 1：原图-ROI 双路分支融合

| 项目 | 内容 |
|---|---|
| 核心思路 | ROI 只作为局部补充证据，不再替代原图上下文。原图分支负责全局检查级信息，ROI 分支负责候选病灶局部信息，最后按标签做 gated late fusion。 |
| 实验设计 | 原图分支沿用 `exp_4 long_mil`，输入最多 `128` 张原图；ROI 分支单独输入 `16/32/64` 张 ROI crop；两个分支各自做 MIL 聚合，输出 label-wise logits；融合方式可设为 `logits = logits_global + gate_label * logits_roi`。加入 instance type embedding 区分 `full_image` 与 `roi_crop`。 |
| 推荐实验名 | `exp6_roi_dual_128_16`、`exp6_roi_dual_128_32`、`exp6_roi_dual_128_64` |
| 预期收益 | 修复 `exp_5` 中 ROI 占用原图名额的问题；SMT 和黏膜病变/肿瘤可能从 ROI 获益，胃炎仍由全图上下文兜底。 |

推荐优先跑 `exp6_roi_dual_128_32`，在成本和噪声之间较平衡。

### 3.3 方案 2：ROI 质量过滤与 teacher attention 引导采样

| 项目 | 内容 |
|---|---|
| 核心思路 | 先减少 ROI 候选噪声，再把有限 ROI 名额分配给更可能有诊断价值的原图。 |
| 实验设计 | 对 SAM2 ROI 做面积、长宽比、贴边比例、反光/黑边/文字遮挡过滤；同一原图内做 NMS 或去重；用 `exp_4 long_mil` 的 label-wise attention 作为 teacher，优先从高 attention 原图提取 ROI，低 attention 原图减少 ROI 配额。 |
| 推荐实验名 | `exp6_roi_filter_96_32`、`exp6_roi_teacher_96_32`、`exp6_roi_teacher_128_32` |
| 预期收益 | 降低 SAM2 ROI 噪声，提高局灶病变 precision；避免 ROI 数量过多造成上下文稀释。 |

建议优先设置 `roi_max_crops_per_bag: 32`、`roi_max_crops_per_source: 1`，并记录过滤前后 ROI 数量分布。若批量扫描 ROI index 或图片，应加入进度条。

### 3.4 方案 3：多视图一致性与注意力正则

| 项目 | 内容 |
|---|---|
| 核心思路 | 同一检查的原图视图、ROI 混合视图应给出一致预测；同时避免 attention 只集中到少数噪声 ROI。 |
| 实验设计 | 构造两个视图：A 为 `64/128` 张原图，B 为原图 + ROI crop；两视图共享模型参数，主损失仍为 ASL，额外加入预测一致性损失 `KL(prob_A, prob_B)`。对 label-wise attention 加 entropy regularization、attention dropout 和 top-attention instance dropping。 |
| 推荐实验名 | `exp6_roi_cons_128_32`、`exp6_roi_cons_96_32_attndrop` |
| 预期收益 | 缓解 ROI 噪声和后期过拟合，提高 subset accuracy；促使模型同时利用全局证据和局部证据。 |

推荐超参：

| 参数 | 建议值 |
|---|---|
| `lambda_cons` | `0.05`、`0.1`、`0.2` |
| `attention_dropout` | `0.1`、`0.2` |
| `top_attention_drop_ratio` | `0.1` |
| `attention_entropy_weight` | `0.001`、`0.005` |

### 3.5 方案 4：Teacher-Student 软伪标签与 ROI instance 辅助损失

| 项目 | 内容 |
|---|---|
| 核心思路 | 用当前强模型生成检查级软伪标签，再用高 attention 图像/ROI 生成 instance 级弱监督，让 ROI 分支不只依赖检查级标签反传。 |
| 实验设计 | Teacher 使用 `exp_4 long_mil`、`exp_5 roi_long_mil` 或二者 ensemble；Student 输入原图 + ROI。训练时同时使用真实检查级标签、teacher soft label、top-k ROI/图像 instance auxiliary head。只对 teacher 高置信样本启用伪监督，降低噪声。 |
| 推荐实验名 | `exp6_roi_kd_exp4_teacher`、`exp6_roi_kd_ensemble_teacher`、`exp6_roi_inst_aux_top5` |
| 预期收益 | 提高 ROI 对真实判别区域的聚焦能力；减少标签噪声影响；有望提升 macro F1 与局灶标签 precision。 |

建议损失：

```text
loss = ASL(bag_logits, hard_labels)
     + lambda_kd * KL(student_prob, teacher_prob)
     + lambda_inst * BCE(instance_logits_topk, pseudo_instance_labels)
     + lambda_cons * consistency_loss
```

推荐超参：

| 参数 | 建议值 |
|---|---|
| `lambda_kd` | `0.1`、`0.3` |
| `lambda_inst` | `0.05`、`0.1` |
| `topk_instance` | `3`、`5` |
| teacher 置信度阈值 | `0.75`、`0.8` |

### 3.6 方案 5：ROI pseudo-bag 与实例级聚类约束

| 项目 | 内容 |
|---|---|
| 核心思路 | 借鉴 CLAM 的 instance-level clustering 和 DTFD-MIL 的 pseudo-bag / double-tier 思路，把一个检查拆成多个原图/ROI 子包，让模型学习更稳定的局部证据，而不是把所有 ROI 简单混成一个长 bag。 |
| 实验设计 | 对每个检查构造若干 pseudo-bag：全图子包、ROI 子包、高 attention 子包、随机上下文子包。第一层对子包内 instance 聚合，第二层对子包表示做检查级聚合。对每个标签选 top-k ROI 作为伪阳性候选、bottom-k ROI 作为伪阴性候选，加入聚类或对比约束。 |
| 推荐实验名 | `exp6_roi_pseudobag_dtfd`、`exp6_roi_clam_inst_cluster`、`exp6_roi_labelwise_topk_cluster` |
| 预期收益 | 提高小数据下 ROI 证据利用效率；降低单个噪声 ROI 对 bag 预测的影响；增强模型可解释性，便于后续人工复核 attention evidence。 |

### 3.7 ROI 路线推荐执行顺序

| 优先级 | 实验 | 原因 |
|---:|---|---|
| 1 | `exp6_roi_dual_128_32` | 最直接修复 ROI 替代原图上下文的问题 |
| 2 | `exp6_roi_teacher_96_32` | 成本较低，直接降低 ROI 噪声 |
| 3 | `exp6_roi_cons_128_32` | 针对过拟合和视图不一致 |
| 4 | `exp6_roi_kd_ensemble_teacher` | 潜力更高，但需要先生成 teacher 预测 |
| 5 | `exp6_roi_pseudobag_dtfd` | 结构改动较大，适合作为强方案扩展 |

## 四、基于 exp_4 的多模态分类增强方案

### 4.1 基本原则

本部分与第三部分 ROI 弱监督改进相互独立，以 `exp_4 long_mil` 为图像主干，不依赖 ROI crop。多模态实验的目标是利用报告文本、诊断前结构化信息或标签语义，提升三标签多标签分类性能。

必须特别注意：当前标签来自报告文本。如果测试阶段直接输入最终诊断结论文本，极容易产生标签泄漏。因此多模态实验必须区分：

- 严格实验：测试阶段只输入图像和诊断前可获得信息，例如年龄、性别、检查类型、报告标题、检查所见、检查描述等。
- 上限实验：允许使用最终诊断结论文本，用来估计多模态理论上限，但不能作为临床部署结果，也不能与 image-only 模型做同等部署比较。
- 蒸馏实验：训练期可以使用诊断文本或多模态 teacher，但最终 student 推理只输入图像；这类实验需要明确标注为 `training-time text distillation`。

下面 5 个方法中，前 3 个参考近年多模态/SOTA 思路，后 2 个结合当前项目结构做改进创新。

### 4.2 方法 1：MedCLIP / BiomedCLIP / UniMed-CLIP 式图文对比预训练

| 项目 | 内容 |
|---|---|
| 类型 | 参考 SOTA 方法 |
| 核心思路 | 用检查级图像 bag embedding 与报告文本或标签文本做对比学习，先增强图像表示的医学语义，再微调三标签分类头。MedCLIP 适合医学图文弱配对，BiomedCLIP/UniMed-CLIP 提供医学领域 VLM 预训练思路。 |
| 实验设计 | 图像侧使用 `long_mil` 的 bag embedding 或 label-wise embedding；文本侧使用冻结中文医学 BERT、通用中文 BERT，或可离线抽取的医学 VLM text encoder。先做 image-text contrastive pretraining，再用 ASL + 类别重平衡微调三标签分类。文本分严格版和上限版。 |
| 推荐实验名 | `exp6_mm_contrast_title`、`exp6_mm_contrast_finding`、`exp6_mm_contrast_diagnosis_upper` |
| 预期收益 | 提高图像编码器对疾病语义、部位词、病灶描述词的敏感度；可能改善 SMT 与黏膜病变/肿瘤的区分。 |
| 风险控制 | 诊断结论文本只能用于 `upper` 或训练期蒸馏；正式比较优先使用报告标题、检查所见、检查描述等低泄漏字段。 |

### 4.3 方法 2：ALBEF / BLIP 式先对齐再融合

| 项目 | 内容 |
|---|---|
| 类型 | 参考 SOTA 方法 |
| 核心思路 | 先用图文对比损失对齐图像和文本表示，再通过 cross-attention 融合。相比简单 concat，模型能学习检查图像 instance 与文本 token 之间的对应关系。BLIP 的 caption/filter 思路也可用于清洗噪声报告片段。 |
| 实验设计 | 图像 token 使用 `long_mil` instance features、bag embedding 或 label-wise embeddings；文本 token 使用检查所见/描述 token。训练目标包括图文对比损失、图文匹配损失和三标签分类损失；可选 momentum teacher 稳定小数据训练。 |
| 推荐实验名 | `exp6_mm_albef_bag`、`exp6_mm_albef_label`、`exp6_mm_albef_instance` |
| 预期收益 | 比 late fusion 更强地利用文本细节；对边界模糊的食管 SMT 与食管黏膜病变/肿瘤可能有帮助。 |
| 风险控制 | `exp6_mm_albef_instance` 显存压力最大，先跑 `bag` 或 `label` 版；最终诊断文本只做 `upper`，不能作为严格推理输入。 |

### 4.4 方法 3：LLaVA-Med / BLIP-2 式冻结大模型 teacher

| 项目 | 内容 |
|---|---|
| 类型 | 参考 SOTA 方法 |
| 核心思路 | 不直接把大 VLM 放进部署模型，而是用冻结医学 VLM 或通用 VLM 作为 teacher，生成检查级软标签、标签解释、难例权重或文本原型，再回到 `long_mil` 分类训练。 |
| 实验设计 | 先把每个检查的代表性图像、attention top-k 图像或 montage 输入 VLM teacher，结合安全 prompt 生成三标签相关的 soft score 或解释；再训练 `long_mil` student，使用 hard label + teacher soft label + confidence weight。若 VLM 不能稳定处理胃镜图，可只使用文本 encoder/LLM 生成标签原型和同义词。 |
| 推荐实验名 | `exp6_mm_vlm_teacher_softlabel`、`exp6_mm_vlm_teacher_proto`、`exp6_mm_vlm_teacher_upper` |
| 预期收益 | 利用外部医学视觉语言知识补充小数据标签语义；帮助发现难例和标签混淆模式。 |
| 风险控制 | teacher 输出必须离线保存并人工抽样检查；不得把真实诊断结论拼进测试 prompt；若使用诊断结论生成 teacher，实验必须标注为上限或训练期蒸馏。 |

### 4.5 方法 4：诊断前结构化信息 label-wise late fusion

| 项目 | 内容 |
|---|---|
| 类型 | 项目化创新方法 |
| 核心思路 | 在 `exp_4 long_mil` 图像主干外，加入诊断前可获得的结构化信息，做低泄漏、可部署的多模态 late fusion。 |
| 实验设计 | 图像分支沿用 `long_mil`；结构化分支输入年龄、性别、检查类型、检查时间、门诊/住院、科室、适应证等字段。连续变量标准化，离散变量 embedding。融合时用 label-wise gate 控制每个标签对结构化信息的依赖，避免非因果字段压过图像证据。 |
| 推荐实验名 | `exp6_mm_clinical_basic`、`exp6_mm_clinical_full`、`exp6_mm_clinical_label_gate` |
| 预期收益 | 提高校准、subset accuracy 和稳定性；部署风险低，适合作为正式多模态基线。 |
| 风险控制 | 只使用诊断前字段；排除由诊断结果后填或强相关编码得到的字段；记录缺失率，并对缺失值做独立 embedding 或 mask。 |

### 4.6 方法 5：标签文本原型引导 label graph + 多模态 teacher 到 image-only student 蒸馏

| 项目 | 内容 |
|---|---|
| 类型 | 项目化创新方法 |
| 核心思路 | 当前 `long_mil` 已有 label graph 和 label-wise 表示，可把三个标签的医学文本描述转成 label prototype，约束 label token 和图像 label embedding；同时训练强多模态 teacher，再蒸馏回 image-only student。 |
| 实验设计 | 为每个标签构造多条中文医学原型，例如 `食管黏膜下肿物`、`食管隆起性病变`、`食管黏膜病变`、`食管肿瘤`、`胃炎`、`糜烂性胃炎` 等。文本原型用于初始化或正则化 `LabelGraphReasoner`；teacher 输入图像 + 非诊断文本或结构化字段，student 只输入图像。 |
| 推荐实验名 | `exp6_mm_proto_label_graph`、`exp6_mm_teacher_finding_student`、`exp6_mm_teacher_diagnosis_student_upper` |
| 预期收益 | 加强标签语义和标签关系建模；若 teacher 足够强，student 可在 image-only 推理下获得多模态训练收益。 |
| 风险控制 | 原型词表需要由训练集和医学常识构造，不能从测试标签答案泄漏；诊断结论 teacher 只能作为上限；最终汇报要区分 teacher 多模态结果和 student image-only 结果。 |

建议损失：

```text
loss_student = ASL(student_logits, hard_labels)
             + lambda_kd * KL(student_prob, teacher_prob)
             + lambda_logit * MSE(student_logits, teacher_logits)
             + lambda_proto * contrast(image_label_embed, text_label_proto)
             + lambda_graph * graph_prior_regularization
```

推荐超参：

| 参数 | 建议值 |
|---|---|
| `lambda_kd` | `0.1`、`0.3` |
| `lambda_logit` | `0.05` |
| `lambda_proto` | `0.02`、`0.05`、`0.1` |
| `lambda_graph` | `0.001`、`0.005` |
| 蒸馏温度 | `T=2`、`T=4` |

### 4.7 多模态路线推荐执行顺序

| 优先级 | 实验 | 原因 |
|---:|---|---|
| 1 | `exp6_mm_clinical_basic` | 泄漏风险低，实现成本低，适合作为严格多模态基线 |
| 2 | `exp6_mm_proto_label_graph` | 与当前 `long_mil` 的 label graph 最贴合，推理可保持 image-only |
| 3 | `exp6_mm_contrast_finding` | 可提升图像语义表示，适合作为后续模型初始化 |
| 4 | `exp6_mm_albef_label` | 融合能力更强，但实现和显存成本较高 |
| 5 | `exp6_mm_teacher_finding_student` | 冲性能潜力大，流程更长，需要先训练 teacher |

## 五、结果记录与执行要求

- 所有 `exp_6` 训练输出建议放在 `outputs/train_runs/task2/exp_6/` 下；自动批量实验使用 `experiment_dir -> model_dir -> train_dir` 三层结构，单次训练使用 `model_dir -> train_dir` 两层结构。
- 每个训练目录必须保存 `config.yaml`、`log.csv`、`test_result.csv`、`checkpoints/`，并在实验汇总中记录对应路径。
- 新实验默认继承 `exp_4 long_mil` 的数据划分、类别重平衡、损失函数、优化器和主干设置，除非该实验明确要改变对应变量。
- 若涉及批量生成 ROI、teacher 预测、文本原型或离线 embedding，应加入进度条，并保存可复查的中间文件。
- 所有多模态实验必须在结果表中标注文本字段来源：`title`、`finding`、`description`、`structured`、`diagnosis_upper` 或 `training_time_distill`。
- 严格实验、上限实验和 image-only student 蒸馏实验必须分开汇报，不能混在同一排名中。

## 六、参考方法方向

- CLAM（https://arxiv.org/abs/2004.09666）：弱监督 MIL 中用 attention 找高诊断价值区域，并加入 instance-level clustering 约束。
- DTFD-MIL（https://arxiv.org/abs/2203.12081）：通过 pseudo-bag 和 double-tier feature distillation 增强小样本 MIL 的实例证据利用。
- Noisy Student（https://arxiv.org/abs/1911.04252）：用 teacher 伪标签和带噪 student 训练增强泛化。
- MedCLIP（https://arxiv.org/abs/2210.10163）：医学图文对比学习，关注医学图文弱配对和 false negative 问题。
- BiomedCLIP（https://arxiv.org/abs/2303.00915）/ UniMed-CLIP（https://arxiv.org/abs/2412.10372）：医学领域视觉语言预训练，用于增强跨模态医学语义表示。
- ALBEF（https://arxiv.org/abs/2107.07651）/ BLIP（https://arxiv.org/abs/2201.12086）：先图文对齐再融合，并通过 momentum distillation 或 caption/filter 处理噪声图文监督。
- LLaVA-Med（https://arxiv.org/abs/2306.00890）/ BLIP-2（https://arxiv.org/abs/2301.12597）：冻结视觉或语言大模型并做轻量适配，可作为训练期 teacher 或原型生成工具。
