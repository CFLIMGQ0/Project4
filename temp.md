# 临时上下文记录

## 当前时间点

本文件用于在新终端中续接当前任务上下文。

## 已完成事项

### 1. TASK2 数据清洗已完成

已处理文件：

- `/home/Lim/Project4/datasets/valid_dicts_report_for task2.csv`

新增说明文档：

- `/home/Lim/Project4/src/TASK2_DATA.MD`

处理内容：

- 只修改了 `valid_dicts_report_for task2.csv` 的 `watch` 列
- 没有修改 `watch_num`
- 没有修改 `suggest`
- 没有修改 `watchResult`
- 没有修改训练代码

清洗结果：

- 总行数：3447
- 原始 `watch` 含 `|` 的行数：3440
- 去重后真正存在非同文本冲突的行数：419
- 其中人工处理的低相似度难例：27
- 后续模板残句/占位符清洗行数：62
- 最终 `watch` 中残留 `|`：0
- 最终明显占位符残留：0

合并原则已经写在：

- `/home/Lim/Project4/src/TASK2_DATA.MD`

### 2. 当前没有继续修改项目结构

你后面问了“如何重构项目，把 TASK1 和 TASK2 分清楚”，目前**只给了方案，还没有改代码或目录结构**。

## 当前项目现状判断

当前项目已经出现明显的“TASK1 偏置”，主要体现在：

- `training/data.py` 里仍然写死了 TASK1 的胃镜三标签：
  - `label_esophageal_smt`
  - `label_esophageal_mucosal_or_tumor`
  - `label_gastritis`
- `task_data_selection.py` 现在只会生成：
  - `gastro_multilabel_task_datalist.csv`
  - `colonoscopy_binary_task_datalist.csv`
- `train.py` 里是“模型名 -> 任务名”的硬编码映射，不利于同一模型切换 TASK1 / TASK2
- `configs/path.yaml` 里的 `valid_dicts_report_csv` 仍默认指向：
  - `/home/Lim/Project4/datasets/valid_dicts_report.csv`
- `TASK.md` 和 `TASK2.MD` 已分开，但代码运行层并没有真正任务化

## 已查看过的关键文件

- `/home/Lim/Project4/src/train.py`
- `/home/Lim/Project4/src/training/data.py`
- `/home/Lim/Project4/src/training/trainer.py`
- `/home/Lim/Project4/src/training/metrics.py`
- `/home/Lim/Project4/src/training/losses.py`
- `/home/Lim/Project4/src/task_data_selection.py`
- `/home/Lim/Project4/src/configs/train.yaml`
- `/home/Lim/Project4/src/configs/model.yaml`
- `/home/Lim/Project4/src/configs/path.yaml`
- `/home/Lim/Project4/src/TASK.md`
- `/home/Lim/Project4/src/TASK2.MD`
- `/home/Lim/Project4/src/TASK_DATA.md`

## 我给出的重构建议核心结论

不要复制成两套项目。

建议改成：

- 一套共享训练框架
- 多个独立任务定义

建议最少拆成 3 个任务：

- `task1_gastro3`
- `task2_gastro8`
- `colon_binary`

核心思想：

- 现在项目需要从“模型驱动任务”改成“任务驱动模型”
- 以后训练入口应该是：
  - 先选任务
  - 再选模型

而不是让模型名反向决定训练任务。

## 建议的未来目录方向

建议增加：

```text
src/tasks/
  registry.py
  task1_gastro3/
    spec.yaml
    labels.py
    selection.py
    dataset_builder.py
  task2_gastro8/
    spec.yaml
    labels.py
    selection.py
    dataset_builder.py
  colon_binary/
    spec.yaml
    labels.py
    selection.py
    dataset_builder.py
```

建议配置拆分：

```text
configs/
  base/
    train.yaml
    model.yaml
    path.yaml
  tasks/
    task1_gastro3.yaml
    task2_gastro8.yaml
    colon_binary.yaml
```

建议数据与输出目录拆分：

```text
datasets/task_data/
  task1_gastro3/
  task2_gastro8/
  colon_binary/

outputs/train_runs/
  task1_gastro3/
  task2_gastro8/
  colon_binary/
```

## 哪些东西应该共享

这些建议继续共享，不按 TASK1/TASK2 复制：

- `training/`
- `model/`
- `baselines/`
- `sotas/`
- `losses`
- `metrics`

## 哪些东西必须按任务拆开

- 标签定义
- 数据筛选规则
- datalist 生成逻辑
- 任务默认配置
- 任务默认数据源
- 输出目录
- 缓存目录
- 任务文档

## 当前最重要的重构方向

### 第一优先级

先引入 `TaskSpec` / `tasks registry` 概念，把任务从公共代码中抽出来。

### 第二优先级

把 `TASK1` 先迁移为一个独立任务定义，确保旧流程还能跑。

### 第三优先级

把 `TASK2` 定义成独立任务，绑定：

- `/home/Lim/Project4/datasets/valid_dicts_report_for task2.csv`

### 第四优先级

再改 `train.py`，改成：

- `--task`
- `--model`

而不是当前按模型隐式决定任务。

## 下一步建议

如果你在新终端继续，我建议直接做下面这件事：

1. 先不碰模型
2. 先设计任务层抽象
3. 先出一份“重构蓝图”
4. 再按蓝图逐步落地

最合适的下一步任务是：

- 让我先输出一版**详细重构蓝图**

这份蓝图应该明确：

- 新目录结构
- 保留文件
- 拆分文件
- 迁移顺序
- `TaskSpec` 字段设计
- 训练入口参数设计
- TASK1/TASK2 的配置归属

## 补充：建议项目结构树（带文件作用注释）

下面这版是结合当前讨论后整理出的**推荐结构树**。

注意：

- 这是**建议的目标结构**，不是说当前仓库已经改成这样
- 这版结构里**保留 `scripts/`**
- 这版结构里 `baselines/` 与 `sotas/` 后续按任务拆开
- 这版结构里每个目录/文件后面的中文说明，就是该项的职责

```text
/home/Lim/Project4/
├── datasets/                                           # 数据总目录，放原始数据、任务数据、缓存
│   ├── main_data/                                      # 原始患者检查目录与图像/PDF 主数据
│   ├── reports/                                        # 报告汇总表目录，统一放各类 CSV
│   │   ├── valid_dicts_report.csv                      # TASK1 默认使用的报告总表
│   │   └── valid_dicts_report_for_task2.csv            # TASK2 使用的清洗后报告总表，建议去掉文件名空格
│   ├── task_data/                                      # 按任务生成的中间数据与样本表
│   │   ├── task1_gastro3/                              # TASK1 的任务数据目录
│   │   │   ├── datalist.csv                            # TASK1 最终样本列表，每行对应一个检查样本
│   │   │   ├── split_seed42.json                       # TASK1 固定训练/验证/测试划分结果
│   │   │   └── stats.json                              # TASK1 标签统计、样本统计结果
│   │   ├── task2_gastro8/                              # TASK2 的任务数据目录
│   │   │   ├── datalist.csv                            # TASK2 最终样本列表
│   │   │   ├── split_seed42.json                       # TASK2 固定划分结果，建议按患者分层
│   │   │   ├── stats.json                              # TASK2 标签分布、共现矩阵、覆盖率统计
│   │   │   ├── pseudo_labels.cache                     # TASK2 文本伪标签缓存结果
│   │   │   └── parser_debug_samples.csv                # TASK2 文本解析抽查结果，便于人工核对
│   │   └── colon_binary/                               # 肠镜任务的任务数据目录
│   │       ├── datalist.csv                            # 肠镜二分类样本列表
│   │       ├── split_seed42.json                       # 肠镜固定划分结果
│   │       └── stats.json                              # 肠镜样本统计与类别分布
│   └── image_cache/                                    # 图像缓存根目录，避免重复读图/缩放
│       ├── task1_gastro3/                              # TASK1 图像缓存
│       ├── task2_gastro8/                              # TASK2 图像缓存
│       └── colon_binary/                               # 肠镜任务图像缓存
│
├── outputs/                                            # 输出目录，放训练结果、图表、分析报告
│   ├── train_runs/                                     # 所有训练运行目录
│   │   ├── task1_gastro3/                              # TASK1 的训练输出根目录
│   │   ├── task2_gastro8/                              # TASK2 的训练输出根目录
│   │   └── colon_binary/                               # 肠镜任务的训练输出根目录
│   ├── reports/                                        # 自动实验汇总报告目录
│   ├── figures/                                        # 可视化图表目录
│   └── cache_combine_reports/                          # 报告合并阶段的过程缓存输出
│
├── pre_weights/                                        # 预训练模型权重目录
│
└── src/                                                # 源代码主目录
    ├── train.py                                        # 统一训练入口，未来建议支持 --task --model --experiment
    ├── infer.py                                        # 统一推理入口，后续需要时新增
    ├── task_data_selection.py                          # 过渡期任务样本生成入口，后续应逐步拆薄
    │
    ├── tasks/                                          # 任务定义层，核心职责是把 TASK1/TASK2/肠镜彻底解耦
    │   ├── __init__.py                                 # 导出任务注册接口，方便外部统一导入
    │   ├── base.py                                     # 定义 TaskSpec、TaskBuilder 等任务抽象基类
    │   ├── registry.py                                 # 注册所有任务，例如 task1_gastro3 / task2_gastro8 / colon_binary
    │   │
    │   ├── task1_gastro3/                              # TASK1：胃镜三标签任务目录
    │   │   ├── __init__.py                             # 导出 TASK1 任务对象
    │   │   ├── task.py                                 # TASK1 的主任务定义，组织 labels/selection/split 等模块
    │   │   ├── labels.py                               # TASK1 标签名、标签顺序、显示名、标签解释
    │   │   ├── selection.py                            # 从原始报告表筛选 TASK1 可用样本
    │   │   ├── records.py                              # 把 TASK1 CSV 行构造成训练 records
    │   │   ├── split.py                                # 定义 TASK1 的数据切分逻辑
    │   │   └── stats.py                                # 统计 TASK1 标签分布与任务规模
    │   │
    │   ├── task2_gastro8/                              # TASK2：胃镜八标签任务目录
    │   │   ├── __init__.py                             # 导出 TASK2 任务对象
    │   │   ├── task.py                                 # TASK2 的主任务定义，组织 labels/parser/pseudo_labels 等模块
    │   │   ├── labels.py                               # TASK2 八标签名称、顺序、显示名、标签说明
    │   │   ├── hierarchy.py                            # TASK2 父子标签层次关系与层次约束定义
    │   │   ├── selection.py                            # 从 TASK2 报告表筛选样本并生成 datalist
    │   │   ├── records.py                              # 构造 TASK2 训练 records，连接图像、标签和文本字段
    │   │   ├── split.py                                # 定义 TASK2 按患者分层切分逻辑
    │   │   ├── stats.py                                # 统计 TASK2 标签分布、共现矩阵、覆盖率
    │   │   ├── text_parser.py                          # 解析 watch/specimen 文本，抽取结构化信息
    │   │   ├── pseudo_labels.py                        # 生成实例级区域标签与 relevance 伪标签
    │   │   └── validation.py                           # 检查文本解析质量与伪标签质量
    │   │
    │   └── colon_binary/                               # 肠镜二分类任务目录
    │       ├── __init__.py                             # 导出肠镜任务对象
    │       ├── task.py                                 # 肠镜任务主定义
    │       ├── labels.py                               # 定义 normal / polyp 等类别信息
    │       ├── selection.py                            # 从报告表中筛选肠镜任务样本
    │       ├── records.py                              # 构造肠镜训练 records
    │       ├── split.py                                # 定义肠镜任务切分逻辑
    │       └── stats.py                                # 输出肠镜任务统计结果
    │
    ├── training/                                       # 通用训练引擎层，只负责训练过程，不再写死任务标签
    │   ├── __init__.py                                 # 导出训练模块公共接口
    │   ├── dataset.py                                  # MILBagDataset，负责读图、采样、缓存、返回 bag 数据
    │   ├── collate.py                                  # mil_collate_fn，负责 batch padding 与组装
    │   ├── sampler.py                                  # InstanceAwareBatchSampler，控制 batch 中实例总量
    │   ├── trainer.py                                  # Trainer 主训练循环，负责训练/验证/测试流程
    │   ├── losses.py                                   # 各类损失函数与损失组合逻辑
    │   ├── metrics.py                                  # 统一计算 macro/micro/per-label 指标
    │   ├── visualization.py                            # 生成 ROC、PR、混淆矩阵、训练曲线等图
    │   └── utils.py                                    # 通用训练辅助函数
    │
    ├── model/                                          # 自定义主模型目录
    │   ├── __init__.py                                 # 导出主模型注册接口
    │   ├── common/                                     # 主模型共用组件目录
    │   │   ├── __init__.py                             # 导出公共组件
    │   │   ├── backbones.py                            # 封装 ConvNeXt、ResNet 等主干网络
    │   │   ├── pooling.py                              # 封装 GatedAttention、MultiLabelAttention 等聚合模块
    │   │   └── graph.py                                # 标签图推理相关公共图模块
    │   │
    │   ├── gastro_label_graph_mil/                     # 当前已有主模型目录，先保留作为旧主线
    │   │   ├── __init__.py                             # 导出旧主模型
    │   │   ├── modules.py                              # 旧主模型的内部组件
    │   │   └── network.py                              # 旧主模型的网络结构
    │   │
    │   └── rg_hmil/                                    # TASK2 未来的新主模型目录
    │       ├── __init__.py                             # 导出 RG-HMIL 模型
    │       ├── modules.py                              # IRP、区域分组、条件图推理等子模块
    │       └── network.py                              # RG-HMIL 顶层网络定义
    │
    ├── baselines/                                      # baseline 模型目录，已从 baseline 改名为 baselines
    │   ├── __init__.py                                 # baseline 模型统一导出与注册入口
    │   │
    │   ├── task1_gastro3/                              # TASK1 的 baseline 模型集合
    │   │   ├── __init__.py                             # 导出 TASK1 baseline 注册表
    │   │   ├── common.py                               # TASK1 baseline 共用基础模块
    │   │   ├── attention_mil.py                        # TASK1 的 ABMIL baseline
    │   │   ├── mean_pool.py                            # TASK1 的均值池化 baseline
    │   │   ├── max_pool.py                             # TASK1 的最大池化 baseline
    │   │   ├── topk_mil.py                             # TASK1 的 Top-k baseline
    │   │   └── transformer_mil.py                      # TASK1 的 Transformer baseline
    │   │
    │   ├── task2_gastro8/                              # TASK2 的 baseline 模型集合
    │   │   ├── __init__.py                             # 导出 TASK2 baseline 注册表
    │   │   ├── common.py                               # TASK2 baseline 共用基础模块
    │   │   ├── attention_mil.py                        # TASK2 的 ABMIL-8L baseline
    │   │   ├── mean_pool.py                            # TASK2 的均值池化 baseline
    │   │   ├── max_pool.py                             # TASK2 的最大池化 baseline
    │   │   ├── topk_mil.py                             # TASK2 的 Top-k baseline
    │   │   └── transformer_mil.py                      # TASK2 的 Transformer baseline
    │   │
    │   ├── colon_binary/                               # 肠镜 baseline 目录
    │   │   ├── __init__.py                             # 导出肠镜 baseline 接口
    │   │   └── colonoscopy_mil_baseline.py             # 肠镜二分类基础 MIL 模型
    │   │
    │   └── legacy_alias/                               # 兼容旧导入路径的过渡目录，稳定后可删除
    │       └── colonocopy_baseline.py                  # 兼容旧拼写 colonocopy 的别名文件
    │
    ├── sotas/                                          # SOTA 对照模型目录
    │   ├── __init__.py                                 # SOTA 模型统一导出与注册入口
    │   │
    │   ├── task1_gastro3/                              # TASK1 的 SOTA 模型集合
    │   │   ├── __init__.py                             # 导出 TASK1 SOTA 注册表
    │   │   ├── common.py                               # TASK1 SOTA 共用模块
    │   │   ├── clam_sb.py                              # TASK1 的 CLAM-SB 实现
    │   │   ├── clam_mb.py                              # TASK1 的 CLAM-MB 实现
    │   │   ├── dsmil.py                                # TASK1 的 DS-MIL 实现
    │   │   ├── transmil.py                             # TASK1 的 TransMIL 实现
    │   │   └── dtfd_mil.py                             # TASK1 的 DTFD-MIL 实现
    │   │
    │   ├── task2_gastro8/                              # TASK2 的 SOTA 模型集合
    │   │   ├── __init__.py                             # 导出 TASK2 SOTA 注册表
    │   │   ├── common.py                               # TASK2 SOTA 共用模块
    │   │   ├── clam_sb.py                              # TASK2 的 CLAM-SB 八标签版本
    │   │   ├── clam_mb.py                              # TASK2 的 CLAM-MB 八标签版本
    │   │   ├── dsmil.py                                # TASK2 的 DS-MIL 八标签版本
    │   │   ├── transmil.py                             # TASK2 的 TransMIL 八标签版本
    │   │   ├── dtfd_mil.py                             # TASK2 的 DTFD-MIL 八标签版本
    │   │   └── wikg_mil.py                             # TASK2 可新增的 WIKG-MIL 实现
    │   │
    │   └── colon_binary/                               # 肠镜 SOTA 目录，后续需要时可扩展
    │       └── __init__.py                             # 预留肠镜 SOTA 注册入口
    │
    ├── scripts/                                        # 脚本入口层，负责命令行调用与串联流程，不应消失
    │   ├── common/                                     # 通用脚本目录
    │   │   ├── combine_reports.py                      # 合并报告与生成总表的脚本
    │   │   ├── delete_broken_data.py                   # 统一清理破损数据的脚本
    │   │   ├── show_reportTitle.py                     # 统计 reportTitle 分布的脚本
    │   │   ├── show_watch_result.py                    # 统计 watchResult 分布的脚本
    │   │   ├── clean_values.py                         # 清洗和标准化字段值的脚本
    │   │   └── check_report_modification.py            # 检查报告修改结果一致性的脚本
    │   │
    │   ├── task1_gastro3/                              # TASK1 专属脚本目录
    │   │   ├── build_datalist.py                       # 生成 TASK1 datalist 的命令行脚本
    │   │   ├── validate_labels.py                      # 验证 TASK1 标签命中和样本质量的脚本
    │   │   └── summarize_stats.py                      # 汇总 TASK1 统计结果的脚本
    │   │
    │   ├── task2_gastro8/                              # TASK2 专属脚本目录
    │   │   ├── build_datalist.py                       # 生成 TASK2 datalist 的命令行脚本
    │   │   ├── validate_labels.py                      # 验证 TASK2 八标签规则覆盖率的脚本
    │   │   ├── debug_text_parser.py                    # 调试 watch/specimen 文本解析的脚本
    │   │   ├── inspect_pseudo_labels.py                # 抽查 TASK2 伪标签质量的脚本
    │   │   └── summarize_stats.py                      # 汇总 TASK2 统计结果的脚本
    │   │
    │   └── colon_binary/                               # 肠镜任务脚本目录
    │       ├── build_datalist.py                       # 生成肠镜样本列表的脚本
    │       └── summarize_stats.py                      # 汇总肠镜统计结果的脚本
    │
    ├── configs/                                        # 配置目录
    │   ├── base/                                       # 全局共享配置目录
    │   │   ├── path.yaml                               # 只保存根路径和公共路径，不再写死具体任务 CSV
    │   │   ├── train.yaml                              # 全局默认训练超参数
    │   │   └── runtime.yaml                            # 运行时配置，例如日志、cache、seed 等
    │   │
    │   ├── tasks/                                      # 各任务自己的配置目录
    │   │   ├── task1_gastro3.yaml                      # TASK1 的标签数、数据源、输出目录等配置
    │   │   ├── task2_gastro8.yaml                      # TASK2 的数据源、文本伪标签参数、标签配置
    │   │   └── colon_binary.yaml                       # 肠镜任务配置
    │   │
    │   ├── models/                                     # 各模型的默认结构参数配置目录
    │   │   ├── gastro_label_graph_mil.yaml             # 当前旧主模型的默认参数
    │   │   ├── rg_hmil.yaml                            # RG-HMIL 的默认参数
    │   │   ├── baselines.yaml                          # baseline 模型的默认参数
    │   │   └── sotas.yaml                              # SOTA 模型的默认参数
    │   │
    │   └── experiments/                                # 实验编排配置目录
    │       ├── phase0_label_expand.yaml                # Phase 0：标签扩展验证配置
    │       ├── phase1_text_guidance.yaml               # Phase 1：文本引导实验配置
    │       ├── phase2_conditional_graph.yaml           # Phase 2：条件图推理实验配置
    │       ├── phase3_full_compare.yaml                # Phase 3：完整对比实验配置
    │       ├── auto_baselines.yaml                     # 自动运行 baseline 的配置
    │       ├── auto_sotas.yaml                         # 自动运行 SOTA 的配置
    │       └── auto_ablations.yaml                     # 自动运行消融实验的配置
    │
    ├── docs/                                           # 文档目录，逐步替代散落在根目录的说明文档
    │   ├── project/                                    # 项目级文档目录
    │   │   ├── README.md                               # 项目总说明文档
    │   │   ├── AGENTS.md                               # 协作规则与开发约束说明
    │   │   ├── DATASETS.md                             # 数据集结构与数据说明文档
    │   │   ├── MODEL.md                                # 模型结构与模型目录说明文档
    │   │   ├── BASELINES.md                            # baseline 说明文档
    │   │   └── SOTAS.md                                # SOTA 说明文档
    │   │
    │   ├── task1/                                      # TASK1 文档目录
    │   │   ├── TASK.md                                 # TASK1 任务定义文档
    │   │   ├── TASK_DATA.md                            # TASK1 数据说明文档
    │   │   └── paper_overview.md                       # TASK1 论文概览或故事线文档
    │   │
    │   └── task2/                                      # TASK2 文档目录
    │       ├── TASK2.md                                # TASK2 主任务定义文档
    │       ├── TASK2_DATA.md                           # TASK2 数据清洗与字段变更说明
    │       ├── TODO_LIST.md                            # TASK2 实现 TODO 与模型设计清单
    │       ├── refactor_plan.md                        # TASK2 对应的项目重构蓝图
    │       └── parser_notes.md                         # TASK2 文本解析规则说明
    │
    └── legacy/                                         # 过渡期临时文件存放区，后续逐步清空
        ├── temp.md                                     # 临时上下文记录文件，当前用于续接对话上下文
        ├── temp.py                                     # 临时代码实验文件
        └── temp2.py                                    # 临时代码实验文件
```

## 关于这版结构树的补充说明

### 1. 为什么 `scripts/` 还要保留

我之前那版结构树没有把 `scripts/` 明确写出来，这是不完整的。

这里重新明确：

- `scripts/` 不应该删除
- `scripts/` 负责命令行入口和流程编排
- 真正的任务逻辑应该下沉到 `tasks/`
- 真正的训练逻辑应该放在 `training/`

也就是说：

- `scripts/` = “怎么调用”
- `tasks/` = “任务规则是什么”
- `training/` = “训练流程怎么跑”
- `model/` / `baselines/` / `sotas/` = “模型结构是什么”

### 2. 为什么 `baselines/` 和 `sotas/` 后续要按任务拆

因为后续不同任务的输出维度和标签定义已经不同：

- TASK1 是胃镜三标签
- TASK2 是胃镜八标签
- 肠镜任务是二分类

如果仍然把 baseline 和 SOTA 混在同一层目录里，后续代码会越来越依赖大量条件分支，不利于维护。

所以建议：

- `baselines/task1_gastro3/`
- `baselines/task2_gastro8/`
- `baselines/colon_binary/`

SOTA 同理拆分。

### 3. 当前最适合的落地方向

当前最适合的重构顺序仍然是：

1. 先引入 `tasks/registry.py`
2. 先把 TASK1/TASK2 的标签规则和样本构造逻辑从公共代码抽出来
3. 再改 `train.py` 为 `--task + --model`
4. 再去做 TASK2 的文本解析和伪标签
5. 最后再实现 `RG-HMIL`

## 当前文件修改事实

截至本文件当前版本，实际改动如下：

- 已修改：`/home/Lim/Project4/datasets/valid_dicts_report_for task2.csv`
- 已新增：`/home/Lim/Project4/src/TASK2_DATA.MD`
- 已修改：`/home/Lim/Project4/src/train.py`
- 已修改：`/home/Lim/Project4/src/MODEL.md`
- 已修改：`/home/Lim/Project4/src/BASELINE.MD`
- 已修改：`/home/Lim/Project4/src/TASK2.MD`
- 已修改：`/home/Lim/Project4/src/temp.md`
- 已将目录：`/home/Lim/Project4/src/baseline` 重命名为 `/home/Lim/Project4/src/baselines`

说明：

- 上述代码与文档改动，主要来自“`baseline` 改名为 `baselines`”这一步
- 当前这个 `temp.md` 已额外补充了“建议结构树 + 文件作用说明”
