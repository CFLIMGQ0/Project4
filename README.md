# 检查数据集项目说明

## 项目定位

本仓库当前主要用于整理患者检查数据集，并持续推进多轮数据清洗工作。

当前重点包括：

- 维护患者级目录结构；
- 梳理每名患者的多次检查数据；
- 对图片与报告等文件组织进行基础核查；
- 为后续数据整理、标注与任务设计提供稳定的数据基础。

## 当前阶段说明

目前工作重点仍然是**先处理数据集本身**，后续是否开展分类任务尚未确定，相关方向统一视为**暂定**。

因此，本仓库文档暂不展开任务定义、类别划分或建模目标，而是以数据组织与清洗准备为主。

`scripts/show_reportTitle.py` 用于统计 `valid_dicts_report.csv` 中的 `reportTitle` 类型分布。

`scripts/temp_show_watchResult.py` 用于统计 `valid_dicts_report.csv` 中的 `watchResult` 类型分布，说明文档见 `show_watchResult.md`。

`clean_values.py` 用于清洗 `valid_dicts_report.csv` 中的 `hp` 与 `operationValue` 字段，适合专项数据修正使用。其中 `operationValue` 会统一将 `，` / `,` 改为 `|`，并去除末尾操作编码括号。

## 项目目录约定

当前项目根目录固定为 `/home/Lim/Project4`，目录约定如下：

- `src/`：工作环境目录，代码、脚本、说明文档与 `configs/path.yaml` 均位于此目录；
- `datasets/`：数据集根目录；
- `outputs/`：脚本输出根目录；
- `pre_weights/`：预训练模型权重目录。

目录关系示例如下：

```text
/home/Lim/Project4/
├── src/                      # 工作环境目录
│   ├── configs/path.yaml     # 路径配置文件
│   └── scripts/              # 数据处理与统计脚本
├── datasets/                 # 数据集根目录
├── outputs/                  # 输出根目录
└── pre_weights/              # 预训练模型权重目录
```

文档中的 `python scripts/...` 命令默认在 `/home/Lim/Project4/src` 下执行；若在项目根目录 `/home/Lim/Project4` 下执行，请将脚本路径写成 `python src/scripts/...`。

## 当前数据目录约定

工作目录 `src/` 中的路径参数统一记录在 `configs/path.yaml` 中，包含项目根目录、数据根目录、输出目录以及相关结果文件路径。

需要区分两个概念（并在 `configs/path.yaml` 中分别配置）：

- **数据集根路径（存放说明文件）**：`paths.dataset_base_root`；
- **实际数据目录（存放患者目录）**：`paths.dataset_root`。

也就是说，脚本默认读取的是 `paths.dataset_root`，而不是 `paths.dataset_base_root`。

`configs/path.yaml` 关键字段约定（文档统一使用变量名，不直接写死具体值）：

- `paths.project_root`：项目根目录（`src`、`datasets`、`outputs`、`pre_weights` 所在位置）；
- `paths.dataset_base_root`：数据集根路径（说明文件、统计文件、辅助文件所在位置）；
- `paths.dataset_root`：实际数据目录（患者目录所在位置，脚本默认读取此路径）；
- `paths.valid_dicts_pdf_csv`：`valid_dicts_pdf.csv` 的显式路径（当前环境建议使用绝对路径）；
- `paths.valid_dicts_report_csv`：`valid_dicts_report.csv` 的显式路径（当前环境建议使用绝对路径）；
- `paths.output_dir`：脚本输出根目录；
- `paths.process_cache_dir_name`：`combine_reports.py` 过程文件子目录名（默认 `cache_combine_reports`）。
- `paths.check_similarity_dir_name`：`temp_check_similarity.py` 输出子目录名（位于 `paths.output_dir` 下，默认 `check_similarity`）。

目录组织示例如下：

```text
paths.dataset_root/              # 实际数据目录（脚本默认读取）
└── ZSxxxxxxxx/                  # 患者目录：一级目录，一名患者对应一个目录
    ├── ZSxxxxxxxx/              # 检查目录：该患者的一次检查
    │   ├── img/                 # 图片目录：保存该次检查的图像文件
    │   └── pdf/                 # 报告目录：保存该次检查的 PDF 报告
    ├── ZSxxxxxxxx/              # 另一检查目录：同一患者的另一轮检查
    │   ├── img/                 # 该轮检查对应的图像文件
    │   └── pdf/                 # 该轮检查对应的 PDF 报告
    └── ...                      # 省略其余检查目录或其他患者目录
```

说明：

- `paths.dataset_base_root` 表示数据集根路径；
- `paths.dataset_root` 表示实际数据目录；
- 患者目录位于 `paths.dataset_root` 下，每个一级目录代表一名患者；
- 每名患者下可有 `1~n` 个检查目录；
- 每个检查目录下通常包含 `img` 与 `pdf` 两类子目录；
- 这里的 `xxxxxxxx` 表示数字部分；
- 患者目录名与其下各检查目录名中的 `xxxxxxxx` 可能不同，不要求一致；
- 实际检查目录名应保持为数据里的具体目录名，而不是中文描述性命名。
- `useless_key.json` 建议放在 `paths.dataset_base_root/useless_key.json`，用于记录全量 PDF 中始终为空的英文字段键（不放在 `paths.dataset_root` 内）。


## 当前工作目录（src）内容

- 数据处理脚本；
- 路径配置文件；
- 数据结构与清洗阶段说明文档。

## 训练输出目录约定（新增）

`train.py` 当前的训练输出统一落在：

`paths.output_dir / train_run_dir_name`

下面再按任务拆分：

- `gastro_multilabel_task/`
- `colonoscopy_binary_task/`

### 常规训练

常规训练时，运行目录直接就是训练目录，不再额外套一层模型子目录。

目录命名规则：

- 胃镜：`gastro_<运行次数>`
- 肠镜：`colonoscopy_<运行次数>`

例如：

- `train_runs/gastro_multilabel_task/gastro_1/`
- `train_runs/colonoscopy_binary_task/colonoscopy_2/`

每个训练目录下会生成以下核心文件：

- `config.yaml`：记录该训练目录对应的模型参数、训练参数、切分统计等；
- `log.csv`：按 `epoch + split(train/val)` 记录损失、学习率以及各类训练/验证指标；
- `loss_curve.png`：每个 epoch 验证结束后刷新；
- `last_confusion_matrix.png`：每个 epoch 验证结束后刷新；
- `checkpoints/last.ckpt`：当前最后一个 epoch 模型；
- `checkpoints/best_macro_f1.ckpt`：验证集 `Macro F1` 最佳模型；
- `checkpoints/best_micro_f1.ckpt`：验证集 `Micro F1` 最佳模型；
- `checkpoints/best_val_loss.ckpt`：验证集 `loss` 最低模型；
- `best_macro_f1_val_confusion_matrix.png`：`best_macro_f1` 对应的验证混淆矩阵；
- `best_micro_f1_val_confusion_matrix.png`：`best_micro_f1` 对应的验证混淆矩阵；
- `best_val_loss_val_confusion_matrix.png`：`best_val_loss` 对应的验证混淆矩阵；
- `test_macro_f1/`：保存 `best_macro_f1.ckpt` 的测试 ROC、PR、`best_macro_f1_test_confusion_matrix.png` 与测试指标；
- `test_micro_f1/`：保存 `best_micro_f1.ckpt` 的测试 ROC、PR、`best_micro_f1_test_confusion_matrix.png` 与测试指标；
- `test_val_loss/`：保存 `best_val_loss.ckpt` 的测试 ROC、PR、`best_val_loss_test_confusion_matrix.png` 与测试指标；
- `test_result.csv`：对三个最佳模型分别测试后的结果汇总；
- `test_report.csv`：与 `test_result.csv` 同内容的兼容副本。

测试完成后，终端会输出三个最佳模型各自的测试结果摘要；`test_result.csv` 与 `test_report.csv` 会记录对应测试指标。当前测试指标包含：

- `test_loss`
- `Label-wise ACC`
- `Per-class Recall`
- `Per-class Precision`
- `Per-class Specificity`
- `Per-class ROC-AUC`
- `Per-class PR-AUC`
- `Macro Recall`
- `Macro Precision`
- `Macro Specificity`
- `Macro F1`
- `Macro ROC-AUC`
- `Macro PR-AUC`
- `Micro Recall`
- `Micro Precision`
- `Micro F1`
- `Hamming Loss`
- `Kappa`

其中按类别展开的指标会在 CSV 中按列展开保存，命名形式类似：

- `label_wise_acc_<类别名>`
- `recall_<类别名>`
- `precision_<类别名>`
- `specificity_<类别名>`
- `roc_auc_<类别名>`
- `pr_auc_<类别名>`

### 自动探索

自动探索时，运行目录下会包含多个训练目录。

目录命名规则：

- 胃镜：`gastro_<运行次数>_para_auto`
- 肠镜：`colonoscopy_<运行次数>_para_auto`

自动探索运行目录根部会额外生成：

- `notes.json`：机器可读的结构化摘要；
- `remark.txt`：给人看的简短总结。

每个自动探索运行目录下包含多个：

- `train_001/`
- `train_002/`
- `train_003/`

每个 `train_xxx/` 目录直接就是训练产物，不再生成 `01_gastro_label_graph_mil` 这类模型子目录，也不再在训练目录内生成 `remark.txt`。

`log.csv` 当前会尽量展开常用标量指标列，便于后续直接做统计和绘图，重点包括：

- Label-wise ACC
- Per-class Recall
- Per-class Precision
- Per-class Specificity
- Per-class ROC-AUC
- Per-class PR-AUC
- Macro Recall
- Macro Precision
- Macro Specificity
- Macro F1
- Macro ROC-AUC
- Macro PR-AUC
- Micro F1
- Hamming Loss
- Kappa

说明：

- 胃镜三标签任务的混淆矩阵会以“一个文件内多个标签子图”的方式保存；
- 肠镜二分类任务的混淆矩阵会保存为标准 2x2 图；
- 终端输出已改为按 epoch 紧凑展示，不再输出长横线分隔。

## reportTitle / watchResult 相关脚本（新增）

以下命令默认在 `/home/Lim/Project4/src` 下执行：

1. 统计 `reportTitle` 类型分布：

   ```bash
   python scripts/show_reportTitle.py
   ```

2. 统计 `watchResult` 类型分布：

   ```bash
   python scripts/temp_show_watchResult.py
   ```

   详细说明见：`show_watchResult.md`。

3. 专项清洗脚本 `clean_values.py` 用于修正 `hp` 字段值，并标准化 `operationValue` 的保存格式。

## 使用提醒

- 当前以数据整理与清洗为主；
- 原始数据不应提交到 Git 仓库；
- 实际路径请以 `configs/path.yaml` 中的参数配置为准；
- 文档中的 `python scripts/...` 命令默认在 `/home/Lim/Project4/src` 执行；
- 预训练模型权重文件统一在项目根目录 `/home/Lim/Project4/pre_weights` 下读取与下载；首次运行若本地缺失，相关脚本会自动下载到该目录；
- 若默认路径不存在，请修改配置后再执行相应脚本。

## 清洗脚本执行顺序与效果

当前目录级清洗统一改为执行总脚本：

```bash
python scripts/delete_broken_data.py
```

该脚本会在运行前先统计以下四项：

- 患者数量；
- 检查目录数量；
- 图片数量；
- 报告数量。

随后按以下五步顺序执行，并且每一步都会先计算“待删除目录或文件数量”；若数量大于 0，则先询问是否执行；若数量为 0，则直接输出说明并进入下一步：

1. 删除损坏文件：
   会遍历数据集内的 PDF 与图片文件，检查文件头、文件尾、关键结构（如 PDF 的 `startxref`、PNG 的 `IEND`、JPEG 的 `EOI` 等）是否完整，并输出 `broken_files.csv`。
2. 删除空 `img/pdf` 子目录：
   会检查每个检查目录下的 `img/` 与 `pdf/` 子目录是否存在且包含对应类型文件，并删除实际存在且为空或异常的子目录。
3. 删除不完整检查目录：
   会识别仅含 `img` 或仅含 `pdf` 的检查目录，并按确认结果删除对应检查目录。
4. 手动删除指定目录：
   当前会纳入经过人工检查确认需要手动处理的目录，删除原因是“检查为图片数量不全且损坏”。该步会逐个询问是否删除具体目录。
5. 删除空患者目录：
   会扫描并删除已经不包含任何检查目录的空患者目录。

全部步骤结束后，脚本会再次统计患者数量、检查目录数量、图片数量与报告数量，用于对比清洗前后的变化。

此外，`delete_broken_data.py` 会在数据集根路径 `paths.dataset_base_root` 下额外生成 `delete_broken_data.json`，其中仅记录 `patient_count`、`exam_count`、`image_count`、`report_count` 这 4 个统计值，用作后续唯一性确认缓存的版本判断依据。

## 检查目录唯一性确认脚本说明（新增）

`scripts/combine_reports.py` 现用于执行两项任务：

1. 对每个检查目录做第一轮唯一性确认：
   - 读取该检查目录下所有 PDF 的非空键值；
   - 在逐份补充键值过程中，若同名键出现不同非空值，则记为冲突键；
   - 全部补充完成后无冲突则判定为有效目录（`is_valid=1`），有冲突则判定为无效目录（`is_valid=0`）。
2. 对有效目录输出补充后的有效键结果。

脚本输出文件默认写入 `paths.output_dir`（可通过 `--output-dir` 覆盖）。其中，过程文件会写入
`paths.output_dir/paths.process_cache_dir_name`（默认目录名为 `cache_combine_reports`），并区分四轮（第一轮 + 第二类 + 第三类 + 第四轮）结果：

- 第一轮：`valid_dicts_pdf_round1.csv`、`valid_dicts_report_round1.csv`、`combine_reports_round1.jsonl`；
- 第二类唯一性确认（非重要有效键）：`valid_dicts_pdf_round2.csv`、`valid_dicts_report_round2.csv`、`combine_reports_round2.jsonl`；
- 第三类唯一性确认（重要有效键）：`valid_dicts_pdf_round3.csv`、`valid_dicts_report_round3.csv`、`combine_reports_round3.jsonl`；
- 第四轮唯一性确认（统计 suggest/watch 冲突并保留）：`valid_dicts_pdf_round4.csv`、`valid_dicts_report_round4.csv`、`combine_reports_round4.jsonl`；
- 最终（第四轮）结果会将 `valid_dicts_pdf.csv` 写入过程目录 `paths.output_dir/paths.process_cache_dir_name`，并仅将 `valid_dicts_report.csv` 写入数据集根目录（`paths.dataset_base_root`）。

第二类唯一性确认（非重要有效键）冲突处理规则：

- `archiveTime`/`checkTime` 冲突：取最晚时间；
- `roomName` 冲突：置空；
- `anesthesiologistName` 冲突：置空；
- `narcosisType` 冲突：置空；
- `doctorName` 冲突：先剔除含数字值，再取剩余值里长度最长者。
- `endoscopeName` 冲突：按逗号拆分并合并去重，仅剔除“无数字且被更长值完整包含”的泛化项（如 `肠镜` + `肠镜136` 合并为 `肠镜136`，但 `肠镜13` + `肠镜136` 会同时保留）。

第三类唯一性确认（重要有效键）冲突处理规则：

- `badness` 冲突：统一置为 `有`；
- `hp` 冲突：按 `阳性 > 阴性 > 待确认 > 未检` 取值。
- `score` 冲突：取分数更大的值；
- `operationValue` 冲突：按逗号拆分多值后合并去重；后续如运行 `clean_values.py`，会进一步统一写成 `操作1|操作2|...` 的形式，并去除每个操作末尾的编码括号（如 `(43.4108)`、`(45.4300x009)`）。
- `specimen` 冲突：按部位拆分后合并去重；同一部位有多个数量时取较大数量，并保留全部部位。
- `watchResult` 冲突：按逗号拆分为多个类型后合并去重。

第四轮唯一性确认补充规则：

- 第四轮统一统计 `suggest` 与 `watch` 冲突：不做唯一值确认，不移除冲突键，仅统计冲突目录数与冲突项数量；
- 第四轮定位为“兼容记录轮次”而非“继续处理轮次”：完成后会输出“冲突已记录”，并将 `suggest/watch` 视为“已记录冲突”；
- 第四轮完成后，若检查目录仅剩 `suggest/watch` 冲突，则该检查目录记为有效目录（可进入后续处理）；仅当仍存在其他键冲突时，才视为“冲突未完全解决”；
- 每轮完成后都会生成该轮的 `valid_dicts_pdf_roundX.csv` / `valid_dicts_report_roundX.csv` / `combine_reports_roundX.jsonl`；
- 第四轮完成后，脚本会更新兼容输出：`valid_dicts_pdf.csv`（过程目录）与 `valid_dicts_report.csv`（数据集根目录）。

缓存版本校验补充说明：

- `combine_reports.py` 运行时会先读取 `paths.dataset_base_root/delete_broken_data.json`；
- 同时会检查 `paths.output_dir/paths.process_cache_dir_name/delete_broken_data.json` 是否存在；
- 若过程目录中的统计文件缺失、损坏，或其中 4 个统计值与当前数据集统计值不一致，则视为数据集已更新，当前过程缓存全部失效；
- 一旦失效，`paths.output_dir/paths.process_cache_dir_name` 下原有过程文件会整体清空，并从第一轮开始重新运行；
- 全部轮次完成后，会把当前的 `delete_broken_data.json` 同步写入过程目录，作为下次判断缓存是否可复用的基准。

新增输出指标说明（`valid_dicts_pdf_roundX.csv`）：

- `suggest_num`：若 `suggest` 无冲突则为 `1`；若存在冲突则记录冲突总数量 `n`（按该键在目录内出现的非空值总次数统计）；
- `watch_num`：若 `watch` 无冲突则为 `1`；若存在冲突则记录冲突总数量 `n`（按该键在目录内出现的非空值总次数统计）；
- `conflict_key_types` 中会按数量展开：例如 `suggest_num=3` 时会出现 `suggest|suggest|suggest`，`watch_num=2` 时会出现 `watch|watch`。
- `conflict_instance_count`：冲突实例总数（`suggest`/`watch` 按其冲突数量展开计数，其余键按 1 计数）。

脚本会在每轮开始前检查过程目录（`paths.output_dir/paths.process_cache_dir_name`）中是否已有该轮确认结果（`roundX` 的 csv + jsonl）。若存在则跳过该轮计算并直接进入下一轮；若不存在则执行该轮并落盘。

## `valid_dicts_pdf.csv` 与 `valid_dicts_report.csv` 补充说明（已调整）

- `valid_dicts_pdf.csv`：PDF 粒度结果文件，仅保留在过程目录 `paths.output_dir/paths.process_cache_dir_name` 下，不再同步到数据集根目录。
- `valid_dicts_report.csv`：检查目录粒度结果文件，写入数据集根目录。第四轮会额外保存 `suggest_num`、`watch_num`，并将多值 `suggest/watch` 以 `watch1 | watch2 | ...` 的形式拼接保存。
- 如执行 `clean_values.py`，`valid_dicts_report.csv` 中的 `operationValue` 会再统一规范为 `操作1|操作2|...`，且去除末尾操作编码括号，仅保留操作名称本身。

建议在 `configs/path.yaml` 中通过 `paths.valid_dicts_pdf_csv` 与 `paths.valid_dicts_report_csv` 显式配置两者路径，并优先使用绝对路径；同时保留 `paths.project_root`，便于统一识别项目根目录。

## 训练配置说明

当前训练入口为：

```bash
python train.py --train-config configs/train.yaml --model-config configs/model.yaml
```

配置文件职责拆分如下：

- `configs/train.yaml`：记录训练参数、数据切分参数、采样参数、batch 参数和启用模型列表。
- `configs/model.yaml`：只记录自定义模型 `gastro_label_graph_mil` 的结构参数。

说明：

- baseline 的结构参数不写入配置文件，直接使用代码默认值。
- baseline 与自定义模型的训练流程默认都先读取 `configs/train.yaml` 中的全局训练参数。

`configs/train.yaml` 当前关键字段说明：

- `enabled_models`：当前要运行的模型列表。
- `batch_size`：默认训练 batch size。
- `eval_batch_size`：默认验证/测试 batch size。
- `train_max_instances`：训练阶段每个 bag 最多取多少张图。
- `eval_max_instances`：验证/测试阶段每个 bag 最多取多少张图。
- `train_max_batch_instances`：训练阶段单个 batch 的实例总数上限。
- `eval_max_batch_instances`：验证/测试阶段单个 batch 的实例总数上限。
- `image_cache_mode`：图像缓存模式，当前支持 `none`、`memory`、`disk`、`memory_and_disk`。
- `image_cache_dir`：图像缓存根目录；当前配置为 `/home/Lim/Project4/datasets/task_data`，训练时会按任务自动拆成 `cache_gastro_multilabel_image/` 与 `colonoscopy_binary_image_cache/`。
- `image_cache_warmup`：是否在训练前先预构建当前任务实际会用到的图像缓存。
- `memory_cache_size`：每个 DataLoader worker 的内存图像缓存条数，仅在 `memory` 或 `memory_and_disk` 模式下生效。
- `random_instance_dropout`：训练阶段实例随机丢弃比例。
- `optimizer_name`：优化器类型，当前支持 `adamw`、`adam`、`sgd`。
- `lr`：默认学习率。
- `weight_decay`：默认权重衰减。
- `warmup_ratio`：学习率预热比例。
- `grad_accum_steps`：默认梯度累积步数。
- `amp`：是否启用混合精度训练。
- `topk_evidence`：预留字段，当前这版训练输出不再额外保存证据文件。
- `loss_name`：默认损失函数名称。
- `monitor_metric` / `monitor_mode`：训练过程主监控指标及方向，可用于把早停目标切到 `val_loss`。
- `auto_explore`：是否在执行 `python train.py` 时直接进入自动探索模式。

当前训练图像缓存约定：

- 图像缓存默认不做全库预处理，只会预构建当前训练任务 `train/val/test` 实际涉及的图像。
- 若当前运行的是胃镜模型（如 `gastro_label_graph_mil`、`gastro_baseline`），则只会写入和读取 `cache_gastro_multilabel_image/`。
- 若当前运行的是肠镜模型（如 `colonoscopy_baseline`），则只会写入和读取 `colonoscopy_binary_image_cache/`。
- 若后续切换任务再次训练，会继续复用对应任务目录下已存在的缓存文件，不会误加载另一类任务的缓存目录。

## 自动探索说明（新增）

若 `configs/train.yaml` 中：

```yaml
auto_explore: true
```

则执行：

```bash
python train.py --train-config configs/train.yaml --model-config configs/model.yaml --auto-explore-config configs/auto_explore.yaml
```

时，不再只跑单次训练，而是会额外读取 `configs/auto_explore.yaml`，自动进行多组 trial 的随机搜索。

自动探索的当前实现规则如下：

- 搜索空间写在 `configs/auto_explore.yaml`；
- 只有 `enabled: true` 的参数会被采样；
- 每个 trial 会复用同一套数据切分，便于公平比较；
- 每个 trial 默认只训练到 `trial_max_epochs`，并使用 `trial_patience` 提前停止；
- `configs/auto_explore.yaml` 中的 `stability_filter` 会基于 `final_val_loss`、`final_train_loss` 与 `best_val_loss` 自动标记“稳定收敛候选”；
- 自动探索阶段仍然根据验证集最优 checkpoint 选参，但每个训练目录训练结束后会立即跑三次测试；
- 每个训练目录会分别对 `best_macro_f1`、`best_micro_f1`、`best_val_loss` 做测试；
- 每次测试的指标与图像结果都写入对应的 `test_*` 目录；
- 每个 trial 结束后会持续刷新 `notes.json` 与 `remark.txt`，便于夜间长时间运行。

自动探索输出目录示例：

```text
paths.output_dir / train_run_dir_name / gastro_multilabel_task / gastro_1_para_auto
```

目录下会额外生成：

- `notes.json`：所有训练目录的结构化摘要；
- `remark.txt`：当前自动探索运行目录的人工摘要；
- `train_001/`、`train_002/` ...：每个训练目录的独立输出；
- 每个 `train_xxx/` 下直接保留 `config.yaml`、`log.csv`、checkpoint、`best_*` 测试目录等训练产物。

收敛优先阶段建议：

- 把 `monitor_metric` 设为 `val_loss`，`monitor_mode` 设为 `min`；
- 自动探索先优先搜索 `lr`、`weight_decay`、`warmup_ratio`、`random_instance_dropout`、`loss_name`；
- 先固定 `batch_size`、`grad_accum_steps`、`train_max_instances`，避免把“资源问题”混进“收敛问题”里；
- 先看 `notes.json` 里的 `stable_trials` 和 `best_stable_trial`，再从这些稳定候选里挑后续正式训练参数。

建议：

- 具体当前探索内容与 `remark` 规则见 `EXPLORE.md`；
- 第一轮先关注稳定收敛候选，再从稳定候选里挑分数更好的配置；
- 显存稳定后，再考虑打开 `batch_size`、`grad_accum_steps`、`train_max_instances`；
- 自动探索结束后，再用最优参数关闭 `auto_explore`，做一次完整正式训练与测试。
