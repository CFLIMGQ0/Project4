# exp_8 多模态结构化键消融实验设计

本文基于 `reportTitle`、`age`、`sex`、`hp`、`operationValue` 五个结构化键的键-标签关联分析和键-键关联分析，设计 TASK2 胃镜三标签多模态模型的字段选择、泄漏风险判断和消融实验。

本实验只讨论结构化/半结构化短字段，不纳入 `watch`、`suggest`、`specimen`、`score`。其中 `operationValue` 对应用户分析中写到的 `openationValue`。

## 一、结论先行

### 1.1 已完成实验后的字段建议

| 字段 | 推荐级别 | 主要理由 | 建议用法 |
|---|---|---|---|
| `age` | 保留 | 单独加入提升很小，但与 `sex` 组合后 `subset_accuracy` 明显高于图像基线 | 连续数值标准化 + 缺失 mask |
| `sex` | 保留 | 低风险、低成本，与 `age` 组合构成保守结构化基线 | 类别 embedding，缺失单独一类 |
| `hp` | 暂不放入默认 exp_8 | `image + age + sex + hp` 未超过图像基线，也低于 `image + age + sex` | 作为胃炎专项或补充实验分析 |

保守低风险结构化组合建议为：

```text
image + age + sex
```

当前 `exp_8` 默认性能候选不是这个保守组合，而是 `image + reportTitle + age + sex + operationValue`。该组合来自非置乱正式候选中的最佳结果，需要和流程字段风险说明一起汇报。

### 1.2 有效但需要谨慎加入的字段

| 字段 | 风险级别 | 主要理由 | 建议用法 |
|---|---|---|---|
| `reportTitle` | 中等风险 | 与三个标签均有关联，尤其与胃炎极强；也与 `hp`、`operationValue` 明显相关 | 单独消融、置乱验证，不能只看性能提升 |
| `operationValue` | 中到高风险 | 与三个标签均有关联，可能编码检查流程、治疗路径或术式选择 | 单独消融、置乱验证、分层验证；正式结论需谨慎 |

`reportTitle` 和 `operationValue` 很可能提供真实有用的检查场景信息，但也可能让模型学习“检查流程/术式路径”而不是图像证据。后续不能直接把 `image + 全字段` 作为唯一主结果，必须配套消融和置乱实验。

## 二、泄漏风险判断

### 2.1 低风险字段

`age`、`sex` 通常属于诊断前可获得信息，泄漏风险低。

它们的主要风险不是标签泄漏，而是学习人群分布偏差。例如某些标签在特定年龄段更常见，模型可能改善校准，但不一定真正改善图像判别能力。

### 2.2 中风险字段

`hp` 是临床检测结果，不是最终诊断文本，但它对 `label_gastritis` 的关联非常强。它可能是有医学意义的辅助信息，也可能在当前数据中与检查流程绑定。

因此 `hp` 适合作为胃炎辅助字段，但建议：

- 使用 label-wise gate，避免它影响两个食管标签。
- 单独记录 `label_gastritis` 的 F1、AUC、PR-AUC、recall 和 specificity。
- 做 `image + age + sex` 与 `image + age + sex + hp` 对照，确认收益是否主要落在胃炎标签。

### 2.3 中到高风险字段

`reportTitle`、`operationValue` 的风险来自两点：

1. 与标签关联强，可能真实反映检查场景。
2. 与彼此和 `hp` 也明显相关，可能共同编码检查流程、术式选择、治疗路径或数据来源。

尤其是 `operationValue`，如果包含 `ESD`、`STER`、`内镜下切除`、`超声内镜` 等信息，模型可能直接利用“已经进入某种处理路径”来预测标签。这个信号未必适合作为正式可部署模型输入。

## 三、统一实验设置

| 项目 | 设置 |
|---|---|
| 任务 | TASK2 胃镜三标签多标签分类 |
| 图像基线 | `exp6_long_mil_64_no_roi` |
| 图像输入 | 每个检查最多 `64` 张原图 |
| 数据划分 | patient-level split，所有消融共用同一划分 |
| 主指标 | `macro_f1` |
| 辅助指标 | `micro_f1`、`macro_auc`、`macro_pr_auc`、`subset_accuracy`、`hamming_loss` |
| 每标签指标 | F1、recall、precision、specificity、ROC-AUC、PR-AUC |
| 融合方式 | 第一版推荐 label-wise gated late fusion |
| 结构化正则 | structured dropout、modality dropout、gate L1 正则 |
| 标签关系模块 | TASK1 同款 `label_hypergraph` |
| 输出目录 | `outputs/train_runs/task2/exp_mm_ablation_hypergraph/` |

当前代码已接入 `auto_exp_8_mm_ablation`，默认配置位于 `configs/task2/train.yaml`。在项目 `src` 目录下直接运行：

```bash
python train.py
```

即可按 `exp_6`、`exp_7` 相同的自动批量结构运行本文件中的结构化字段消融实验。当前版本已将原 label-graph 路径切换为 TASK1 中的 `LabelHypergraphReasoner`，新结果写入 `exp_mm_ablation_hypergraph`，避免和 `exp_8` 主实验目录混淆，也避免复用旧 label-graph checkpoint 或测试结果。每个实验会生成独立训练目录，并保存 `config.yaml`、`log.csv`、`test_result.csv`、`test_report.csv`、`field_audit.csv`、`missing_rate.json`、`structured_metadata.json`，结构化融合模型还会额外保存 `modality_gate_stats.csv`。

字段编码建议：

| 字段 | 编码方式 |
|---|---|
| `age` | 标准化连续值 + 缺失 mask；可额外做年龄段 embedding 对照 |
| `sex` | 类别 embedding，缺失单独一类 |
| `hp` | 类别 embedding，缺失单独一类 |
| `reportTitle` | 类别 embedding，低频标题合并为 `other` |
| `operationValue` | 类别 embedding，低频操作合并为 `other`，缺失单独一类 |

推荐融合结构：

```text
image bag
  -> Long-MIL
  -> label-wise image embedding [B, 3, D]

structured fields
  -> categorical embedding / numeric projection
  -> structured embedding [B, D]

label-wise fusion:
gate_l = sigmoid(MLP_l([image_label_embed_l, structured_embed]))
fused_l = image_label_embed_l + gate_l * Linear_l(structured_embed)
logit_l = classifier_l(fused_l)
```

## 四、主消融实验矩阵

### 4.1 基础与低风险字段

| 编号 | 实验名 | 输入字段 | 目的 | 预期现象 |
|---:|---|---|---|---|
| A0 | `exp8_mm_ablation_image_baseline` | 图像 | 建立所有多模态实验的基线 | 作为性能下限和稳定性参考 |
| A1 | `exp8_mm_ablation_age` | 图像 + `age` | 验证年龄对两个食管标签的独立贡献 | SMT 和黏膜/肿瘤标签可能提升，胃炎提升有限 |
| A2 | `exp8_mm_ablation_age_sex` | 图像 + `age` + `sex` | 建立低风险人口学基线 | 相比 A1 可能小幅变化，`sex` 不应带来大幅提升 |
| A3 | `exp8_mm_ablation_age_sex_hp` | 图像 + `age` + `sex` + `hp` | 验证最推荐的基础结构化组合 | `label_gastritis` 应明显改善，两个食管标签不应明显受损 |

判读重点：

- A1 若提升两个食管标签，说明 `age` 是值得保留的基础键。
- A2 若几乎不变，说明 `sex` 可保留但不是核心贡献字段。
- A3 若只提升胃炎，符合预期；若两个食管标签明显下降，需要加强 label-wise gate 或对 `hp` 做标签专属使用。

### 4.2 检查场景与操作路径字段

| 编号 | 实验名 | 输入字段 | 目的 | 预期现象 |
|---:|---|---|---|---|
| B1 | `exp8_mm_ablation_reportTitle` | 图像 + `reportTitle` | 单独评估报告标题的贡献 | 三个标签都可能提升，胃炎提升可能最大 |
| B2 | `exp8_mm_ablation_operationValue` | 图像 + `operationValue` | 单独评估操作/检查类型的贡献 | 三个标签都可能提升，但要警惕异常大幅提升 |
| B3 | `exp8_mm_ablation_title_operation` | 图像 + `reportTitle` + `operationValue` | 检查两个强字段是否叠加或冗余 | 若提升小于 B1/B2 之和，说明二者信息重叠明显 |
| B4 | `exp8_mm_ablation_all_structured` | 图像 + 五个字段 | 评估全部结构化字段上限 | 性能可能最高，但不能直接作为安全结论 |

判读重点：

- 如果 B1/B2 比 A3 提升很多，需要查看是否集中来自 `label_gastritis` 或某个检查流程。
- 如果 B3 与 B1 或 B2 接近，说明 `reportTitle` 和 `operationValue` 冗余较高。
- B4 如果最强，只能说明“全结构化字段有预测价值”，不能直接说明它们适合作为正式输入。

### 4.3 去字段消融

| 编号 | 实验名 | 输入字段 | 目的 | 预期现象 |
|---:|---|---|---|---|
| C1 | `exp8_mm_ablation_all_without_title` | 全部字段去掉 `reportTitle` | 判断全字段收益是否依赖标题 | 若明显下降，说明标题贡献大 |
| C2 | `exp8_mm_ablation_all_without_operation` | 全部字段去掉 `operationValue` | 判断全字段收益是否依赖操作路径 | 若明显下降，说明操作字段贡献大且需查泄漏 |
| C3 | `exp8_mm_ablation_all_without_hp` | 全部字段去掉 `hp` | 判断胃炎收益是否主要来自 HP | 胃炎指标应下降，食管标签变化应较小 |
| C4 | `exp8_mm_ablation_all_without_age` | 全部字段去掉 `age` | 判断食管标签是否依赖年龄先验 | 两个食管标签可能下降 |

去字段消融用于解释 B4。如果 B4 很强，但 C2 去掉 `operationValue` 后大幅下降，那么最终模型不能简单采用 B4，需要进一步做置乱和分层验证。

## 五、置乱实验设计

置乱实验只针对 `reportTitle` 和 `operationValue`，因为这两个字段最可能编码检查流程或治疗路径。

### 5.1 测试期置乱

| 编号 | 实验名 | 训练输入 | 测试输入 | 目的 |
|---:|---|---|---|---|
| D1 | `exp8_mm_ablation_all_shuffle_title_test` | 全部真实字段 | 测试集置乱 `reportTitle` | 检查模型对标题的依赖程度 |
| D2 | `exp8_mm_ablation_all_shuffle_operation_test` | 全部真实字段 | 测试集置乱 `operationValue` | 检查模型对操作字段的依赖程度 |
| D3 | `exp8_mm_ablation_all_shuffle_title_operation_test` | 全部真实字段 | 测试集同时置乱二者 | 检查强流程字段整体依赖 |

预期现象：

- 如果置乱后性能明显下降，说明模型确实在使用该字段。
- 如果置乱 `operationValue` 后性能大幅下降，尤其是食管标签下降明显，需要警惕模型依赖术式/治疗路径。
- 如果置乱后几乎不下降，说明该字段虽然统计相关，但模型没有强依赖，或者图像分支已经提供相同信息。

### 5.2 训练期置乱

| 编号 | 实验名 | 训练输入 | 测试输入 | 目的 |
|---:|---|---|---|---|
| D4 | `exp8_mm_ablation_shuffle_title_train` | 训练/验证/测试均置乱 `reportTitle` | 置乱 `reportTitle` | 检查标题真实语义是否必要 |
| D5 | `exp8_mm_ablation_shuffle_operation_train` | 训练/验证/测试均置乱 `operationValue` | 置乱 `operationValue` | 检查操作字段真实语义是否必要 |

预期现象：

- D4/D5 应接近没有该字段的对应消融结果。
- 如果训练期置乱后仍明显提升，说明可能存在实现问题、分布泄漏，或模型从缺失模式/类别频率中学到了异常信号。

置乱必须在每个 split 内分别打乱，不能跨 train/val/test 混洗，避免引入额外分布变化。

## 六、分层验证建议

为了判断 `reportTitle` 和 `operationValue` 是否只是流程代理变量，建议增加分层评估。

| 分层方式 | 目的 |
|---|---|
| 按 `reportTitle` 分层评估 | 看模型是否只在某些标题下提升 |
| 按 `operationValue` 分层评估 | 看模型是否依赖某些术式或超声内镜相关类别 |
| 按 `hp` 分层评估胃炎 | 看胃炎提升是否来自 HP 状态，而不是图像证据 |
| 按年龄段分层评估食管标签 | 看年龄先验是否改善泛化，还是只改善某个年龄段 |

每个实验建议额外保存：

```text
field_audit.csv
missing_rate.json
per_field_group_metrics.csv
modality_gate_stats.csv
test_result.csv
test_report.csv
```

`modality_gate_stats.csv` 至少记录每个标签的结构化 gate 均值。理想情况下：

- `hp` 对 `label_gastritis` 的 gate 更高；
- `hp` 对两个食管标签的 gate 较低；
- `age` 对两个食管标签有一定贡献；
- `reportTitle`、`operationValue` 的 gate 不应完全压过图像分支。

## 七、推荐执行顺序

第一轮建议只跑最关键的 8 组：

| 顺序 | 实验名 | 原因 |
|---:|---|---|
| 1 | `exp8_mm_ablation_image_baseline` | 固定图像基线 |
| 2 | `exp8_mm_ablation_age` | 验证最独立的强基础键 |
| 3 | `exp8_mm_ablation_age_sex` | 建立低风险人口学组合 |
| 4 | `exp8_mm_ablation_age_sex_hp` | 建立推荐基础结构化模型 |
| 5 | `exp8_mm_ablation_reportTitle` | 单独评估标题 |
| 6 | `exp8_mm_ablation_operationValue` | 单独评估操作字段 |
| 7 | `exp8_mm_ablation_title_operation` | 评估强字段冗余 |
| 8 | `exp8_mm_ablation_all_structured` | 得到全结构化上限 |

第二轮根据第一轮结果补充：

```text
exp8_mm_ablation_all_without_title
exp8_mm_ablation_all_without_operation
exp8_mm_ablation_all_without_hp
exp8_mm_ablation_all_shuffle_title_test
exp8_mm_ablation_all_shuffle_operation_test
exp8_mm_ablation_all_shuffle_title_operation_test
```

如果第二轮显示 `reportTitle` 或 `operationValue` 置乱后性能大幅下降，则必须把对应字段标注为“流程强依赖字段”，并避免将其作为严格可部署主模型的唯一结论来源。

## 八、最终推荐方案

### 8.1 保守低风险多模态基线

从已完成结果看，低风险字段中 `image + age + sex` 比图像基线略有提升，而 `image + age + sex + hp` 没有带来稳定增益。因此低风险可部署基线建议使用：

```text
image + age + sex
```

实验名：

```text
exp8_mm_ablation_age_sex
```

对应结果：

| 实验名 | 输入 | macro_f1 | micro_f1 | subset_accuracy |
|---|---|---:|---:|---:|
| `exp8_mm_ablation_image_baseline` | image | 0.8719 | 0.8701 | 0.7478 |
| `exp8_mm_ablation_age_sex` | image + `age` + `sex` | 0.8738 | 0.8715 | 0.7654 |
| `exp8_mm_ablation_age_sex_hp` | image + `age` + `sex` + `hp` | 0.8713 | 0.8689 | 0.7443 |

`hp` 仍有明确临床意义，但在本轮训练里没有改善整体 `macro_f1`，建议作为补充实验或胃炎标签专项分析，不放入当前默认 `exp_8` 主模型。

### 8.2 当前 exp_8 默认性能候选

非置乱正式候选中，表现最好的是 `all_without_hp`：

```text
image + reportTitle + age + sex + operationValue
```

实验名：

```text
exp8_mm_ablation_all_without_hp
```

对应结果：

| 实验名 | 输入 | macro_f1 | micro_f1 | subset_accuracy |
|---|---|---:|---:|---:|
| `exp8_mm_ablation_all_structured` | image + 全部五个字段 | 0.8766 | 0.8746 | 0.7601 |
| `exp8_mm_ablation_all_without_hp` | image + `reportTitle` + `age` + `sex` + `operationValue` | 0.8823 | 0.8800 | 0.7690 |
| `exp8_mm_ablation_all_without_age` | image + `reportTitle` + `sex` + `hp` + `operationValue` | 0.8807 | 0.8789 | 0.7725 |

因此当前 `exp_8` 默认入口已经切换为：

```text
exp8_structured_late_gate_mil
structured_fields = reportTitle, age, sex, operationValue
```

需要强调：`reportTitle` 和 `operationValue` 是流程相关字段，可能携带检查路径或术式选择代理信息。它们可以作为当前性能最好的 `exp_8` 主模型配置，但不应被解释为纯图像能力提升。

### 8.3 置乱审计结论

本轮最高 `macro_f1` 来自：

```text
exp8_mm_ablation_shuffle_operation_train
```

该实验在训练/验证/测试中置乱 `operationValue`，属于字段依赖审计，不作为正式默认输入。它的高分说明当前收益并不完全依赖 `operationValue` 的真实取值，也提示 `reportTitle`、`age`、`sex` 或数据划分中的其他结构化分布仍可能提供较强先验。

置乱审计结果应与以下实验一起汇报：

```text
exp8_mm_ablation_all_shuffle_title_test
exp8_mm_ablation_all_shuffle_operation_test
exp8_mm_ablation_all_shuffle_title_operation_test
exp8_mm_ablation_shuffle_title_train
exp8_mm_ablation_shuffle_operation_train
```

### 8.4 论文或实验汇报建议

建议最终分三类汇报：

| 类型 | 模型 | 汇报定位 |
|---|---|---|
| image-only | `exp8_mm_ablation_image_baseline` | 基础图像模型 |
| low-risk structured | `exp8_mm_ablation_age_sex` | 保守低风险多模态基线 |
| exp_8 default | `exp8_mm_ablation_all_without_hp` | 当前性能最优的非置乱正式候选 |
| structured audit | 去字段与置乱实验 | 结构化字段依赖和流程代理风险审计 |

建议结论写成：

```text
低风险人口学字段只能带来小幅增益；加入 reportTitle 和 operationValue 后性能更高，但二者存在检查路径代理风险。当前 exp_8 默认采用 image + reportTitle + age + sex + operationValue 作为性能候选，同时保留 image + age + sex 作为更保守的低风险基线。
```

## 九、结果记录模板

| 实验名 | 字段 | macro_f1 | subset_accuracy | SMT F1 | 黏膜/肿瘤 F1 | 胃炎 F1 | 主要结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `exp8_mm_ablation_image_baseline` | image | 0.8719 | 0.7478 | 0.8421 | 0.8686 | 0.9049 | 图像基线 |
| `exp8_mm_ablation_age` | image+age | 0.8720 | 0.7513 | 0.8402 | 0.8682 | 0.9075 | 年龄单字段提升很小 |
| `exp8_mm_ablation_age_sex` | image+age+sex | 0.8738 | 0.7654 | 0.8427 | 0.8664 | 0.9122 | 保守低风险结构化基线 |
| `exp8_mm_ablation_age_sex_hp` | image+age+sex+hp | 0.8713 | 0.7443 | 0.8360 | 0.8661 | 0.9118 | HP 未带来整体增益 |
| `exp8_mm_ablation_reportTitle` | image+reportTitle | 0.8773 | 0.7549 | 0.8484 | 0.8768 | 0.9068 | 标题有稳定贡献但需审计流程依赖 |
| `exp8_mm_ablation_operationValue` | image+operationValue | 0.8757 | 0.7496 | 0.8421 | 0.8744 | 0.9107 | 操作字段有贡献但风险较高 |
| `exp8_mm_ablation_title_operation` | image+reportTitle+operationValue | 0.8761 | 0.7619 | 0.8462 | 0.8722 | 0.9099 | 两个强字段组合未明显叠加 |
| `exp8_mm_ablation_all_structured` | image+all | 0.8766 | 0.7601 | 0.8467 | 0.8694 | 0.9137 | 全字段不如去 HP |
| `exp8_mm_ablation_all_without_title` | image+age+sex+hp+operationValue | 0.8788 | 0.7619 | 0.8448 | 0.8726 | 0.9191 | 去标题后仍优于全字段 |
| `exp8_mm_ablation_all_without_operation` | image+reportTitle+age+sex+hp | 0.8761 | 0.7531 | 0.8457 | 0.8760 | 0.9067 | 去操作字段后接近全字段 |
| `exp8_mm_ablation_all_without_hp` | image+reportTitle+age+sex+operationValue | 0.8823 | 0.7690 | 0.8489 | 0.8764 | 0.9214 | 非置乱正式候选最优，当前 exp_8 默认 |
| `exp8_mm_ablation_all_without_age` | image+reportTitle+sex+hp+operationValue | 0.8807 | 0.7725 | 0.8566 | 0.8732 | 0.9122 | 去年龄后仍强，流程字段贡献明显 |
| `exp8_mm_ablation_all_shuffle_title_test` | 全字段，测试置乱 reportTitle | 0.8763 | 0.7619 | 0.8467 | 0.8708 | 0.9114 | 标题测试置乱下降不明显 |
| `exp8_mm_ablation_all_shuffle_operation_test` | 全字段，测试置乱 operationValue | 0.8747 | 0.7566 | 0.8467 | 0.8676 | 0.9099 | 操作测试置乱略降 |
| `exp8_mm_ablation_all_shuffle_title_operation_test` | 全字段，测试同时置乱二者 | 0.8758 | 0.7601 | 0.8467 | 0.8694 | 0.9114 | 同时置乱未出现崩塌 |
| `exp8_mm_ablation_shuffle_title_train` | 全字段，训练/验证/测试置乱 reportTitle | 0.8781 | 0.7601 | 0.8406 | 0.8758 | 0.9179 | 审计实验，不作正式输入 |
| `exp8_mm_ablation_shuffle_operation_train` | 全字段，训练/验证/测试置乱 operationValue | 0.8829 | 0.7637 | 0.8477 | 0.8806 | 0.9204 | 最高分但属于审计实验 |
