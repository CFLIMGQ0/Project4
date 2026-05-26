# TASK2 三标签多任务多模态方案

## 1. 核心判断

TASK2 不再采用扩展长尾标签集合作为主任务，主任务回到 3 个稳定标签：

| 标签 | 含义 | 当前可复现实验分布 |
|---|---|---:|
| `label_esophageal_smt` | 食管 SMT / 食管黏膜下隆起或肿物 | 1392 / 2875，48.4% |
| `label_esophageal_mucosal_or_tumor` | 食管黏膜病变 / 食管肿物 / SESCC 相关病变 | 约 50% |
| `label_gastritis` | 胃炎，任何类型 | 1199 / 2875，41.7% |

这个选择更适合二区论文。三个标签不是“任务太简单”，而是可以把论文重点从“堆更多长尾类别”转到“如何利用常规报告中天然存在的弱监督信号，提升检查级多标签内镜诊断”。旧扩展标签集合中若干低频类别测试集正例很少，指标方差会很大，不适合作为论文核心结论。

推荐的论文主线是：

> 利用常规内镜图像 bag、报告所见文本、活检部位和结构化临床信息，构建一个报告与活检引导的弱监督多实例学习框架，用于稳定的三标签上消化道检查级多标签分类。

## 2. 数据支撑

基于 `/home/Lim/Project4/datasets/valid_dicts_report_for task2.csv` 复核，三标签任务有以下优势：

- 三个主标签阳性率均在 40% 到 55% 左右，不再是极端长尾问题。
- `watch` 字段 100% 存在，且已经是清洗合并后的单一内镜所见文本。
- `specimen` 在三标签候选样本中约 74% 非空，能提供医生实际活检区域。
- `hp` 已检约 40%，其中阳性与胃炎高度相关，但存在大量未检，因此应带 missing mask 使用。
- 年龄、性别、`reportTitle` 基本完整，`operationValue` 覆盖约 97%，适合做检查类型和临床上下文建模。
- 标签共现有临床合理性：食管 SMT + 胃炎、食管黏膜病变 + 胃炎频繁共现；食管 SMT 与食管黏膜病变本身共现少，说明模型需要 label-wise attention，而不是一个统一 bag 表征。

更重要的是，CSV 中的 `watch` 和 `specimen` 不是普通元数据，而是弱标注资源：

- `watch` 描述了“在哪个部位看到什么”，例如食管距门齿、胃体、胃窦、胃角、十二指肠等。
- `specimen` 描述了医生真正取材的部位，能作为“诊断相关区域”的弱金标准。
- `hp`、年龄、性别、检查类型能作为结构化辅助变量，但不能替代图像诊断。

## 3. 论文故事

### 3.1 临床问题

真实内镜检查不是单张图像分类，而是一次检查包含几十到上百张图像。医生最终报告是检查级结论，常规数据通常没有框级或图像级标注。因此，直接训练图像分类模型会遇到两个问题：

1. 检查级标签很粗，模型不知道哪几张图像真正支持某个诊断。
2. 多个诊断可能共存，不同标签需要关注不同解剖区域和不同图像证据。

### 3.2 方法切入点

常规报告虽然没有人工框标注，但报告本身包含三类可利用信号：

1. `watch`：弱区域描述。它告诉模型病变大致出现在哪个解剖区域，以及是否存在隆起、病变、萎缩、糜烂、息肉、溃疡等视觉线索。
2. `specimen`：弱诊断锚点。医生选择活检的位置通常是最有诊断价值或最可疑的区域。
3. 结构化字段：年龄、性别、HP、检查类型，可以作为上下文变量，帮助模型校准不同检查场景下的风险。

因此，论文故事不是“输入报告直接预测报告诊断”，而是：

> 在训练阶段利用报告和活检部位生成弱监督，教会 MIL 模型更合理地分配注意力；在推理阶段主要依赖图像 bag 和可获得的结构化变量，避免诊断文本泄漏。

这一点必须在论文中明确，否则审稿人会质疑 `watch` 或 `watchResult` 泄漏标签。

## 4. 多模态实现设计

### 4.1 输入模态

主输入：

- 图像 bag：一次检查的所有内镜图像，训练/验证时按现有策略采样。

训练阶段辅助监督：

- `watch`：解析为区域、病变极性、病变关键词和区域顺序。
- `specimen`：解析为活检区域、多区域取材和取材数量。

可选推理输入：

- 年龄、性别、`reportTitle`、`operationValue`。
- `hp` 只在“临床变量增强”实验中使用，并必须带 missing mask；主结果最好同时报告不使用 HP 的版本。

禁止作为推理输入：

- `watchResult`：这是标签来源，不能输入模型。
- `watch` 原文：这是医生看图后的结构化所见，直接输入容易造成临床文本泄漏。它适合训练阶段生成弱监督，不适合主模型推理阶段直接使用。
- `suggest`：常包含诊断和处理建议，泄漏风险高，不作为输入。

### 4.2 报告解析

现有 `tasks/task2/text_parser.py` 已有基础版本，可继续扩展：

1. 将 `watch` 按句号、分号、逗号切分为 clause。
2. 识别解剖区域：
   - `esophagus`：食管、食道、门齿、齿状线
   - `cardia_fundus`：贲门、胃底
   - `gastric_body`：胃体、大弯、小弯
   - `antrum_angle`：胃窦、胃角、幽门
   - `duodenum`：十二指肠、球部、降部
   - `other`
3. 识别病变极性：
   - 阳性线索：隆起、黏膜下、肿物、病变、萎缩、糜烂、充血、水肿、溃疡、发红、褪色、结节等。
   - 阴性线索：无异常、未见、光滑、正常、无殊等。
4. 解析 `specimen` 中的区域和数量，例如 `胃窦 *1`、`胃体部 *2`。
5. 生成以下离线字段：
   - `pseudo_region_labels`：每张采样图像对应的弱区域标签。
   - `pseudo_relevance`：每张图像的诊断相关性软标签。
   - `specimen_region_multihot`：检查级活检区域多热向量。
   - `specimen_region_count`：检查级活检区域计数。
   - `label_region_prior`：三标签与区域的先验关系。

`label_region_prior` 建议设为：

| 标签 | 主要区域 |
|---|---|
| 食管 SMT | `esophagus` |
| 食管黏膜病变 / 肿物 | `esophagus` |
| 胃炎 | `cardia_fundus`、`gastric_body`、`antrum_angle`，其中胃窦/胃角权重最高 |

### 4.3 模型结构

建议模型命名为 `ReportBiopsyGuidedMIL`，或在工程中继续沿用现有模型命名；论文表述应聚焦“三标签报告-活检引导 MIL”。

模型包含 5 个部分：

1. 图像实例编码器  
   使用 ConvNeXt-Tiny 或 ResNet50 编码每张图像，得到实例特征。

2. 标签级注意力 MIL  
   为三个标签分别学习 attention。食管 SMT、食管黏膜病变和胃炎不共享同一个 bag 表征，因为它们关注的图像证据不同。

3. 区域与相关性预测分支  
   对每张图像预测：
   - 解剖区域 logits。
   - 诊断相关性 logits。

   用 `pseudo_region_labels` 和 `pseudo_relevance` 监督这两个分支。

4. 标签-区域注意力偏置  
   将报告解析得到的区域先验融入 attention：

   ```text
   attention_logit(label, image)
   = 原始 label-wise attention logit
   + alpha_label * relevance_score(image)
   + beta_label * label_region_prior(label, predicted_region(image))
   ```

   这样食管标签天然更关注食管图像，胃炎标签更关注胃窦、胃角、胃体等区域，但仍允许模型根据图像证据修正。

5. 结构化临床变量融合  
   对年龄、性别、检查类型、HP 状态和 HP missing mask 建立一个小型 MLP/embedding encoder，输出 `clinical_context`。融合方式建议用 FiLM 或 classifier bias：

   ```text
   label_embedding = label_embedding * gamma(clinical_context) + beta(clinical_context)
   ```

   主论文可以把 clinical fusion 放在最后一个增强版本中，防止审稿人认为模型过度依赖临床变量。

### 4.4 损失函数

总损失建议为：

```text
L = L_diag
  + lambda_region * L_region
  + lambda_relevance * L_relevance
  + lambda_attention * L_attention_align
  + lambda_aux * L_aux
```

各项含义：

- `L_diag`：三标签 BCE / ASL 主分类损失。
- `L_region`：图像区域预测交叉熵，忽略未知区域。
- `L_relevance`：图像相关性 BCE，使用软标签。
- `L_attention_align`：将标签 attention 在区域上的聚合分布，与 `label_region_prior` 或 `specimen_region_multihot` 做 KL / BCE 对齐。
- `L_aux`：可选辅助任务，例如活动性胃炎、中重度萎缩、HP 阳性预测，只用于提升表征，不作为主论文主要终点。

初始权重建议：

```yaml
region_cls_weight: 0.2
relevance_weight: 0.2
attention_align_weight: 0.1
clinical_fusion_weight: 1.0
aux_task_weight: 0.1
```

如果发现辅助监督影响主分类收敛，应先降低 `region_cls_weight` 和 `relevance_weight`，而不是增加主模型复杂度。

## 5. 实现路线

### 阶段一：恢复 TASK2 三标签主任务

1. 修改 `tasks/task2/selection.py`：
   - `TASK2_LABEL_NAMES` 改为三个标签。
   - `display_name` 改为 `TASK2 胃镜三标签多任务`。
   - 旧扩展标签规则如需保留，应迁移为历史脚本或历史实验说明，避免影响当前 TASK2 主任务。
2. 重新运行：

   ```bash
   python scripts/task2_build_datalist.py
   ```

3. 检查新 datalist：
   - 总样本数约 2875。
   - 三标签阳性率应与当前统计接近。
   - 患者级 6:2:2 划分后，各标签在 train/val/test 都应稳定。

### 阶段二：扩展数据字段

1. 在 datalist 中新增离线解析字段：
   - `pseudo_region_labels_json`
   - `pseudo_relevance_json`
   - `specimen_region_multihot`
   - `specimen_region_count`
   - `clinical_features_json`
   - `clinical_missing_mask_json`
2. 遍历 CSV 和图像样本时必须保留进度条，因为该步骤会处理数千个检查目录和十几万图像路径。
3. `training/data.py` 当前已经能把 `pseudo_region_labels` 和 `pseudo_relevance` 放入 batch，后续需要继续加入：
   - `clinical_features`
   - `clinical_missing_mask`
   - `specimen_region_targets`

### 阶段三：改造模型

优先在现有 `model/gastro_label_graph_mil/` 上做增量优化，不把可选扩展模型作为默认主线。若需要复用 `model/rg_hmil/` 中的区域分组、相关性预测等模块，应按三标签任务逐步迁移。

1. `gastro_label_graph_mil` 的 `num_labels` 必须解析为 3。
2. 保留并优化：
   - `InstanceEncoder`
   - `MultiLabelAttentionMIL`
   - `LabelGraphReasoner`
   - per-label classifier
3. 可选迁移：
   - `InstanceRelevancePredictor`
   - `AnatomicalInstanceGrouper`
   - relevance-aware attention
4. 可选新增：
   - `LabelRegionPriorBias`
   - `ClinicalContextEncoder`
   - `AttentionAlignmentLoss`
5. `forward` 支持：
   - `clinical_features`
   - `clinical_missing_mask`
   - `specimen_region_targets`

### 阶段四：训练配置

更新 `configs/task2/model.yaml`：

- 所有标签维度配置改为 3 维，例如 `class_prior`、`label_difficulty`、`head_label_indices`。
- `gastro_label_graph_mil` 默认开启，先建立三标签强基线。
- 报告弱监督和区域分组相关配置先作为增量实验开启，不作为默认训练配置。

更新 `configs/task2/train.yaml`：

- `experiment_dir_name` 使用新的实验目录，例如 `label_graph_mil_3label`。
- 主监控指标建议用 `macro_f1` 或 `macro_auc`，但最终论文主表同时报告 AUROC、PR-AUC、F1、sensitivity、specificity。
- 保持患者级划分：`group_by_patient: true`。

训练输出目录按项目规范：

```text
outputs/train_runs/task2/
└── label_graph_mil_3label/
    └── gastro_label_graph_mil/
        └── train_001/
            ├── config.yaml
            ├── log.csv
            ├── test_result.csv
            ├── checkpoints/
            └── attention_examples/
```

## 6. 实验设计

### 6.1 主结果表

比较以下模型：

| 模型 | 目的 |
|---|---|
| Attention MIL | 基础检查级 MIL |
| Label Graph MIL | 验证标签相关性建模是否有用 |
| CLAM-MB / DSMIL / TransMIL / DTFD-MIL | 与 MIL SOTA 对比 |
| Label Graph MIL image-only | 当前默认基础模型 |
| Label Graph MIL + watch relevance | 验证报告所见弱监督 |
| Label Graph MIL + watch + specimen | 验证活检锚点价值 |
| Label Graph MIL + watch + specimen + clinical | 验证结构化变量增益 |

### 6.2 消融实验

必须做以下消融，否则故事不完整：

1. 去掉 `pseudo_region_labels`。
2. 去掉 `pseudo_relevance`。
3. 去掉 `specimen` attention alignment。
4. 去掉 label-wise attention，改成共享 attention。
5. 去掉 clinical context。
6. 不同 bag size：8、12、16、24。
7. 不同推理输入设定：
   - image only
   - image + age/sex/check type
   - image + age/sex/check type + HP mask

### 6.3 可解释性实验

重点展示模型是否真的学到了合理证据：

1. 每个标签输出 top-k attention 图像。
2. 食管标签 attention 是否集中在食管区域。
3. 胃炎标签 attention 是否集中在胃窦、胃角、胃体等区域。
4. 有 `specimen` 的样本中，attention top-k 与活检区域的一致率。
5. 错误案例分析：
   - 食管 SMT 与食管黏膜病变混淆。
   - 胃炎阴性但 HP 阳性。
   - 手术胃镜与常规胃镜分布差异。

### 6.4 统计分析

论文建议报告：

- patient-level split 的测试集结果。
- bootstrap 95% CI，重采样单位用患者而不是检查。
- 每标签 AUROC、PR-AUC、F1、sensitivity、specificity。
- macro / micro 指标。
- McNemar 或 bootstrap paired test 比较 proposed 与最强 baseline。

## 7. 风险与规避

### 7.1 文本泄漏

最大风险是把 `watch` 或 `suggest` 当作推理输入。主论文必须明确：

- `watchResult` 只用于派生标签。
- `watch` 和 `specimen` 只在训练阶段生成弱监督或辅助目标。
- 主推理模型不读取医生诊断结论。

如果要展示“多模态推理模型”，只能使用诊疗前或检查时可合理获得的结构化信息，例如年龄、性别、检查类型。HP 是否能作为推理输入要谨慎，建议放补充实验。

### 7.2 伪区域标签噪声

内镜图像顺序和报告描述顺序并非完全一致，不能把 `watch` 解析结果当强标签。实现上应使用软标签、弱权重和 attention-level 约束，避免强制每张图像精确对应某个 clause。

### 7.3 检查类型偏倚

数据中包含常规胃镜、染色胃镜、超声胃镜、手术胃镜。模型可能学习到检查类型和疾病的捷径。需要：

- 做 `reportTitle` / `operationValue` 分层结果。
- 做去除检查类型变量后的对照。
- 必要时在训练中加入检查类型重加权或分层采样。

### 7.4 HP 缺失不是随机缺失

HP 未检占比高，不能简单把未检当阴性。实现时必须编码为：

```text
hp_positive, hp_negative, hp_missing
```

或使用数值 + missing mask。

## 8. 预期论文贡献

最终论文不强调“做了很多类别”，而强调以下贡献：

1. 构建了一个真实世界检查级多标签胃镜数据集，包含约 2875 次上消化道检查和 17 万余张图像。
2. 提出报告与活检引导的弱监督 MIL 框架，将常规报告中的区域描述和活检部位转化为训练监督。
3. 在无图像级人工框标注的前提下，提升三标签检查级分类性能，并提供标签级 attention 可解释证据。
4. 系统验证 `watch`、`specimen` 和结构化临床变量的独立贡献，证明常规临床报告可作为低成本监督来源。

这个故事比扩展长尾标签主任务更稳，因为主标签样本量充足，方法创新点清晰，临床流程也更自然。

## 9. 近期执行清单

1. 先改 TASK2 为三标签 datalist，并复核分布。
2. 保留旧扩展标签实验结果作为“为什么不采用长尾主任务”的内部依据，不作为主论文核心表格。
3. 扩展 `text_parser.py`，输出区域、相关性、活检区域和临床变量。
4. 扩展 `training/data.py` 的 record 与 collate。
5. 在 `gastro_label_graph_mil` 基础上加入 label-region prior bias 和 clinical context encoder。
6. 先跑 image-only、+watch、+watch+specimen 三个版本，确认是否稳定提升。
7. 再决定是否把 HP 和其他临床变量放入主模型，或仅作为补充实验。
