# 项目说明

## 项目概况

本项目用于整理消化内镜检查数据，并在此基础上开展胃镜多标签 MIL 训练实验。

当前仓库的工作分为两部分：

- 数据侧：清洗检查目录、汇总报告字段、生成任务样本表；
- 模型侧：围绕胃镜任务训练 baseline、标签图模型以及可选扩展模型。

当前正式保留 2 个胃镜任务：

1. `task1`：胃镜三标签多标签分类。
2. `task2`：胃镜三标签多标签分类，标签口径与 TASK1 保持一致，用于后续优化实验。

其中，`train.py` 现在默认以 `task2` 为主任务，并按 TASK2 配置运行当前启用模型。

## 目录结构

核心目录如下：

```text
/home/Lim/Project4/
├── src/
│   ├── README.md
│   ├── DATASETS.md
│   ├── TASK2.md
│   ├── TASK2_EXP.md
│   ├── Paper/
│   ├── configs/
│   │   ├── task1/
│   │   └── task2/
│   ├── tasks/
│   │   ├── task1/
│   │   └── task2/
│   ├── baselines/task1/gastro_baseline/
│   ├── sotas/task1/gastro_sota/
│   ├── model/
│   │   ├── gastro_label_graph_mil/
│   │   └── rg_hmil/
│   ├── scripts/
│   └── train.py
├── datasets/
├── outputs/
└── pre_weights/
```

## 任务说明

### TASK1

`task1` 是胃镜三标签多标签分类任务，标签如下：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

任务规则定义在 [tasks/task1/selection.py](/home/Lim/Project4/src/tasks/task1/selection.py)。

### TASK2

`task2` 是胃镜三标签多标签分类任务，标签口径与 TASK1 保持一致，标签如下：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

任务规则定义在 [tasks/task2/selection.py](/home/Lim/Project4/src/tasks/task2/selection.py)，任务设计文档见 [TASK2.md](/home/Lim/Project4/src/TASK2.md)。

## 数据与任务样本表

数据根路径相关配置按任务拆分在：

- [configs/task1/path.yaml](/home/Lim/Project4/src/configs/task1/path.yaml)
- [configs/task2/path.yaml](/home/Lim/Project4/src/configs/task2/path.yaml)

任务样本表默认生成到：

- `/home/Lim/Project4/datasets/task_data/task1/gastro_multilabel_task_datalist.csv`
- `/home/Lim/Project4/datasets/task_data/task2/gastro_multilabel_task_datalist.csv`

生成命令：

```bash
python scripts/task1_build_datalist.py
python scripts/task2_build_datalist.py
```

## 训练使用方式

### 默认运行方式

当前直接执行：

```bash
python train.py
```

默认行为是：

- 默认任务：`task2`
- 默认配置文件：`configs/task2/path.yaml`、`configs/task2/train.yaml`、`configs/task2/model.yaml`
- 默认模型：以 `configs/task2/train.yaml` 中的 `enabled_models` 为准，当前主模型建议使用 `gastro_label_graph_mil`

也就是说，`python train.py` 会按 TASK2 的当前启用模型启动训练。

### 运行 TASK2 主模型

若要运行 `gastro_label_graph_mil`，可直接执行：

```bash
python train.py --task task2 --models gastro_label_graph_mil
```

或将 [configs/task2/train.yaml](/home/Lim/Project4/src/configs/task2/train.yaml) 中的 `enabled_models` 改为：

```yaml
enabled_models:
  - gastro_label_graph_mil
```

### 显式切换任务

运行 TASK1：

```bash
python train.py --task task1
```

运行 TASK2：

```bash
python train.py --task task2
```

也可以同时覆盖模型：

```bash
python train.py --task task2 --models gastro_attention_mil_baseline
python train.py --task task2 --models gastro_label_graph_mil
python train.py --task task2 --models rg_hmil
```

### 自动跑完整 baseline 对照

如果要一次性运行 TASK2 的 5 个 baseline，可以把 [configs/task2/train.yaml](/home/Lim/Project4/src/configs/task2/train.yaml) 中的：

```yaml
auto_baselines: true
enabled_models: []
```

然后执行：

```bash
python train.py --task task2
```

对应配置文件是 [configs/task2/auto_baselines.yaml](/home/Lim/Project4/src/configs/task2/auto_baselines.yaml)。

## 当前模型

### Baseline

当前 baseline 位于 [baselines/task1/gastro_baseline](/home/Lim/Project4/src/baselines/task1/gastro_baseline)，包括：

- `gastro_attention_mil_baseline`
- `gastro_mean_pool_baseline`
- `gastro_max_pool_baseline`
- `gastro_topk_mil_baseline`
- `gastro_transformer_mil_baseline`

这些 baseline 本身是按 `num_labels` 参数化的，因此可直接用于 `task1` 和 `task2`。

### 现有主模型

当前已有标签图模型位于 [model/gastro_label_graph_mil](/home/Lim/Project4/src/model/gastro_label_graph_mil)。

### 可选扩展模型

可选扩展模型 `rg_hmil` 位于 [model/rg_hmil](/home/Lim/Project4/src/model/rg_hmil)，后续可按 TASK2 三标签目标改造为报告弱监督和区域先验增强模型。

## 常用脚本

- [scripts/show_reportTitle.py](/home/Lim/Project4/src/scripts/show_reportTitle.py)：统计 `reportTitle` 分布。
- [scripts/task1_temp_show_watchResult.py](/home/Lim/Project4/src/scripts/task1_temp_show_watchResult.py)：统计 `watchResult` 分布。
- [scripts/clean_values.py](/home/Lim/Project4/src/scripts/clean_values.py)：清洗 `hp`、`operationValue`。
- [scripts/check_report_modification.py](/home/Lim/Project4/src/scripts/check_report_modification.py)：比较报告 CSV 修改前后差异。
- [scripts/task1_delete_broken_data.py](/home/Lim/Project4/src/scripts/task1_delete_broken_data.py)：清洗损坏文件和不完整目录。
- [scripts/task1_combine_reports.py](/home/Lim/Project4/src/scripts/task1_combine_reports.py)：做报告唯一性确认与汇总。

## 文档索引

- [DATASETS.md](/home/Lim/Project4/src/DATASETS.md)：数据集结构、清洗流程、TASK1/TASK2 数据说明。
- [TASK2.md](/home/Lim/Project4/src/TASK2.md)：TASK2 任务设计。
- [MODELS.md](/home/Lim/Project4/src/MODELS.md)：TASK2 模型结构说明。
- [TASK2_EXP.md](/home/Lim/Project4/src/TASK2_EXP.md)：TASK2 实验记录。
- [temp.md](/home/Lim/Project4/src/temp.md)：TASK2 临时指标表。
- [baselines/task1/gastro_baseline/TASK1_BASELINE.MD](/home/Lim/Project4/src/baselines/task1/gastro_baseline/TASK1_BASELINE.MD)：baseline 说明。
- [sotas/task1/gastro_sota/TASK1_SOTA.MD](/home/Lim/Project4/src/sotas/task1/gastro_sota/TASK1_SOTA.MD)：SOTA 说明。
- [configs/task1/TASK1_EXPLORE.MD](/home/Lim/Project4/src/configs/task1/TASK1_EXPLORE.MD)：自动探索说明。
- [model/gastro_label_graph_mil/MODEL.MD](/home/Lim/Project4/src/model/gastro_label_graph_mil/MODEL.MD)：标签图模型结构说明。
- [Paper/TASK1_PAPER_OVERVIEW.MD](/home/Lim/Project4/src/Paper/TASK1_PAPER_OVERVIEW.MD)：论文相关记录。
- [Paper/Sage_LaTeX_Guidelines.tex](/home/Lim/Project4/src/Paper/Sage_LaTeX_Guidelines.tex)：期刊 LaTeX 模板说明。

## 说明

- 文档中的命令默认在 `/home/Lim/Project4/src` 下执行。
- 图像缓存根目录当前配置为 `/home/Lim/Project4/datasets/image_cache`，训练时会按任务自动拆到 `task1/` 与 `task2/`。
- 若训练前尚未生成对应任务样本表，`train.py` 会提示先运行对应的 `scripts/task1_build_datalist.py` 或 `scripts/task2_build_datalist.py`。
