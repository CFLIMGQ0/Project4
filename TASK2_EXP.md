# TASK2 实验结果与问题分析

## 1. 文档边界

本文档用于记录当前 TASK2 三标签任务的训练结果、诊断结论和后续改进方向。TASK2 当前主任务为胃镜检查级三标签多标签分类，标签为：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

旧实验结果如果与当前三标签定义不一致，不再作为当前 TASK2 的正式结论。后续需要重新生成 TASK2 三标签 datalist，并在统一划分下重新记录实验结果。

## 2. 当前实验口径

### 2.1 数据划分

TASK2 后续实验默认使用：

- 患者级划分：`group_by_patient: true`
- 随机种子：`seed=42`
- 划分比例：`split_ratio: [0.6, 0.2, 0.2]`
- 训练样本表：`/home/Lim/Project4/datasets/task_data/task2/gastro_multilabel_task_datalist.csv`

如重新生成 datalist，需要在本节同步记录：

| 项目 | 数值 |
|---|---:|
| 总检查数 | 待更新 |
| 总患者数 | 待更新 |
| 训练集检查数 | 待更新 |
| 验证集检查数 | 待更新 |
| 测试集检查数 | 待更新 |

### 2.2 主指标

每次实验至少记录：

- 每标签 F1、recall、precision、specificity、ROC-AUC、PR-AUC。
- `macro_f1`
- `micro_f1`
- `macro_auc`
- `macro_ap`
- `subset_accuracy`
- `hamming_loss`
- `kappa`
- `test_loss`

模型选择必须写清楚 checkpoint 来源，例如：

- `best_macro_f1`
- `best_micro_f1`
- `best_val_loss`

## 3. 当前基线计划

后续第一轮重新实验建议按以下顺序运行：

| 模型 | 目的 | 状态 |
|---|---|---|
| `gastro_label_graph_mil` | 当前默认基础模型 | 待重新训练 |
| `gastro_attention_mil_baseline` | 标准 attention MIL 对照 | 待重新训练 |
| `gastro_mean_pool_baseline` | 全局平均池化下界 | 待重新训练 |
| `gastro_max_pool_baseline` | 单关键帧假设对照 | 待重新训练 |
| `gastro_topk_mil_baseline` | 局部 top-k 证据对照 | 待重新训练 |
| `gastro_transformer_mil_baseline` | 实例上下文建模对照 | 待重新训练 |

SOTA 对照和 `rg_hmil` 扩展模型应在基础模型稳定后再补充。

## 4. 结果记录模板

### 4.1 总表

| 模型 | Checkpoint | Test Loss | Macro F1 | Micro F1 | Macro AUC | Macro AP | Hamming Loss | Kappa | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 |

### 4.2 每标签指标

| 模型 | 标签 | Recall | Precision | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 待更新 | `label_esophageal_smt` | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 |
| 待更新 | `label_esophageal_mucosal_or_tumor` | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 |
| 待更新 | `label_gastritis` | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 |

### 4.3 训练过程诊断

| 模型 | Best Epoch | Best Val Loss | Final Val Loss | 是否过拟合 | 诊断 |
|---|---:|---:|---:|---|---|
| 待更新 | 待更新 | 待更新 | 待更新 | 待更新 | 待更新 |

## 5. 当前问题清单

后续实验优先检查以下问题：

1. TASK2 datalist 是否已经按三标签重新生成。
2. `tasks/task2/selection.py` 的标签字段是否与 TASK1 三标签一致。
3. `configs/task2/train.yaml` 中 `class_balance.label_names` 是否只包含三个标签。
4. 自动实验配置里的标签维度参数是否同步调整。
5. `gastro_label_graph_mil` 在 TASK2 上的 `num_labels` 是否解析为 `3`。
6. `log.csv` 与 `test_result.csv` 是否包含三标签对应的指标列。

## 6. 后续改进方向

当前优先方向：

1. 先建立 `gastro_label_graph_mil` 三标签强基线。
2. 比较固定阈值与验证集 per-label threshold。
3. 检查三个标签的混淆模式，尤其是食管 SMT 与食管黏膜病变/肿物的区分。
4. 评估 `gastro_transformer_mil_baseline` 是否能通过实例上下文提升稳定性。
5. 在基础模型稳定后，再加入 `watch/specimen` 弱监督分支。

所有后续结论都必须基于当前三标签 TASK2 定义重新记录。
