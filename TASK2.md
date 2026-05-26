# TASK2：胃镜三标签多标签分类任务

## 1. 当前任务定位

TASK2 当前重新定义为胃镜检查级三标签多标签分类任务，任务目标与 TASK1 保持一致，但作为后续优化实验的独立任务入口保留。

当前目标不是扩大标签集合，而是在三个稳定主标签上进一步提升分类准确度、稳定性和可解释性。后续实验默认以 `model/gastro_label_graph_mil` 作为基础模型，在此基础上逐步优化注意力、标签图、弱监督和训练策略。

## 2. 标签定义

TASK2 的标签字段与 TASK1 对齐：

| 标签字段 | 中文含义 | 说明 |
|---|---|---|
| `label_esophageal_smt` | 食管 SMT | 包括食管黏膜下隆起、食管黏膜下肿物等表述 |
| `label_esophageal_mucosal_or_tumor` | 食管黏膜病变/肿物 | 包括食管黏膜病变、食管肿物、占位、新生物等表述 |
| `label_gastritis` | 胃炎 | 包括慢性胃炎、活动性/非活动性胃炎、萎缩性胃炎及京都分级相关胃炎表述 |

这是多标签任务，同一次检查可以同时命中多个标签。例如，一条记录可以同时为 `label_esophageal_smt=1` 和 `label_gastritis=1`。

## 3. 数据来源

TASK2 默认使用清洗后的检查目录级报告总表：

- `/home/Lim/Project4/datasets/valid_dicts_report_for task2.csv`

任务样本表默认生成到：

- `/home/Lim/Project4/datasets/task_data/task2/gastro_multilabel_task_datalist.csv`

生成命令：

```bash
python scripts/task2_build_datalist.py
```

样本粒度是一条检查记录：

```text
exam_dir = 一个 bag = 同一次胃镜检查中的多张图像
labels = 三个检查级多标签
```

`watchResult` 是标签派生来源；`watch`、`specimen`、`hp` 等字段可以用于数据分析或训练阶段弱监督，但不能把 `watchResult` 输入模型。

## 4. 与 TASK1 的关系

TASK1 和 TASK2 的核心标签口径保持一致：

- TASK1：作为三标签任务的基础版本和已有实验口径。
- TASK2：继续使用三标签任务，但面向后续模型优化、患者级划分、报告弱监督和更严格实验记录。

这样做的目的，是把论文和实验重点放在“如何提升稳定三标签胃镜 MIL 分类性能”，而不是把主要结论建立在低频标签上。

## 5. 默认基础模型

TASK2 后续默认基础架构为：

- `model/gastro_label_graph_mil`

该模型包含四个核心部分：

1. `InstanceEncoder`：对 bag 内每张图像提取实例特征。
2. `MultiLabelAttentionMIL`：为三个标签分别学习 attention 聚合。
3. `LabelGraphReasoner`：学习标签间关系并做标签图传播。
4. 标签级分类头：每个标签输出一个 logit。

整体流程：

```text
图像 bag -> 实例特征 -> 标签级 attention 聚合 -> 标签图传播 -> 三标签预测
```

## 6. 后续优化方向

当前后续优化应围绕三标签准确度和泛化稳定性展开，优先级如下：

1. 提升 `gastro_label_graph_mil` 的三标签分类性能。
2. 保持 patient-level split，避免同一患者多次检查造成数据泄漏。
3. 优化 attention，使三个标签能关注不同图像证据。
4. 利用 `watch` 与 `specimen` 生成训练阶段弱监督，例如区域先验、相关性先验和活检区域锚点。
5. 做 per-label threshold、温度校准、早停策略和数据增强对照。
6. 保留 baseline、SOTA 和消融实验，但所有主结果都按三标签任务汇报。

## 7. 数据泄漏约束

后续实验必须明确区分标签来源、训练辅助信息和推理输入：

- `watchResult`：只用于派生标签，不允许输入模型。
- `watch`：可用于训练阶段弱监督或离线解析，不作为主推理输入。
- `specimen`：可用于训练阶段活检区域弱监督，不作为最终诊断文本输入。
- `hp`、年龄、性别、检查类型：如用于模型，应带缺失标记，并单独做消融。

如果使用报告字段作为弱监督，需要在实验记录中说明该字段只参与训练阶段辅助目标，避免被解释为直接读取诊断文本。

## 8. 配置与输出目录要求

TASK2 配置文件位于：

- `configs/task2/path.yaml`
- `configs/task2/data.yaml`
- `configs/task2/model.yaml`
- `configs/task2/train.yaml`

后续三标签配置需要确保：

- `tasks/task2/selection.py` 的 `TASK2_SPEC.label_names` 只包含三个标签。
- `configs/task2/train.yaml` 中所有 `class_balance.label_names` 只包含三个标签。
- 自动实验配置中的 `class_prior`、`label_difficulty`、`head_label_indices` 等标签维度参数与三标签一致。

训练输出目录遵守项目统一规范：

```text
outputs/train_runs/task2/
└── <experiment_dir>/
    └── <model_dir>/
        └── <train_dir>/
            ├── config.yaml
            ├── log.csv
            ├── test_result.csv
            ├── checkpoints/
            └── 其他训练产物
```

单次训练可以直接运行：

```bash
python train.py --task task2 --models gastro_label_graph_mil
```

## 9. 评估指标

TASK2 三标签实验至少记录以下指标：

- 每标签：accuracy、recall、precision、specificity、F1、ROC-AUC、PR-AUC。
- 汇总：macro F1、micro F1、macro ROC-AUC、macro PR-AUC、subset accuracy、hamming loss、kappa。
- 训练过程：train loss、val loss、best epoch、checkpoint 选择依据。

默认主比较指标建议使用 `macro_f1` 或 `macro_auc`，但最终报告需要同时给出每个标签的细分表现。
