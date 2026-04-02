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

针对目录级清洗，当前统一按以下顺序执行脚本：

1. `python scripts/delete_broken_files.py`
2. `python scripts/delete_empty_dicts.py`
3. `python scripts/delete_broken_dicts.py`
4. `python scripts/delete_empty_patients.py`

各脚本执行效果说明如下：

- `delete_broken_files.py`：遍历数据集内的 PDF 与图片文件，检查文件头、文件尾、关键结构（如 PDF 的 `startxref`、PNG 的 `IEND`、JPEG 的 `EOI` 等）是否完整，识别潜在损坏文件。
- `delete_empty_dicts.py`：检查每个检查目录下的 `img/` 与 `pdf/` 子目录是否存在并包含目标类型文件；对不合规子目录可执行删除。
- `delete_broken_dicts.py`：识别仅含 `img` 或仅含 `pdf` 的不完整检查目录，并按确认结果删除对应检查目录。
- `delete_empty_patients.py`：扫描并删除已经不包含任何检查目录的空患者目录，避免无效患者目录残留。

这样可以先处理文件级异常，再处理子目录级异常，再处理检查目录级异常，最后处理患者目录级异常，减少清洗过程中的重复判断。

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
