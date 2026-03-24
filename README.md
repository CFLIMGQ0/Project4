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

`temp.py` 已调整为一次性执行文件，仅用于临时任务或短期排查，不作为长期维护脚本。建议优先使用 `scripts/` 目录下的正式清洗脚本完成日常处理。

## 当前数据目录约定

项目的路径参数统一记录在 `configs/path.yaml` 中，包含数据根目录、输出目录以及相关结果文件路径。

需要区分两个概念（并在 `configs/path.yaml` 中分别配置）：

- **数据集根路径（存放说明文件）**：`paths.dataset_base_root`；
- **实际数据目录（存放患者目录）**：`paths.dataset_root`。

也就是说，脚本默认读取的是 `paths.dataset_root`，而不是 `paths.dataset_base_root`。

`configs/path.yaml` 关键字段约定（文档统一使用变量名，不直接写死具体值）：

- `paths.dataset_base_root`：数据集根路径（说明文件、统计文件、辅助文件所在位置）；
- `paths.dataset_root`：实际数据目录（患者目录所在位置，脚本默认读取此路径）。

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


## 当前仓库内容

- 数据处理脚本；
- 路径配置文件；
- 数据结构与清洗阶段说明文档。

## 使用提醒

- 当前以数据整理与清洗为主；
- 原始数据不应提交到 Git 仓库；
- 实际路径请以 `configs/path.yaml` 中的参数配置为准；
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

## 有效检查目录脚本说明（新增）

`scripts/solve_conflicted_pdfs.py` 现用于执行两项任务：

1. 对每个检查目录做第一轮有效性确认：
   - 读取该检查目录下所有 PDF 的非空键值；
   - 在逐份补充键值过程中，若同名键出现不同非空值，则记为冲突键；
   - 全部补充完成后无冲突则判定为有效目录（`is_valid=1`），有冲突则判定为无效目录（`is_valid=0`）。
2. 对有效目录输出补充后的有效键结果。

脚本输出文件默认写入 `paths.output_dir`（可通过 `--output-dir` 覆盖），并区分两轮结果：

- 第一轮：`valid_dicts_pdf_round1.csv`、`valid_dicts_report_round1.csv`、`solve_conflicted_pdfs_round1.jsonl`；
- 第二轮：`valid_dicts_pdf_round2.csv`、`valid_dicts_report_round2.csv`、`solve_conflicted_pdfs_round2.jsonl`；
- 为兼容历史流程，第二轮结果会同步写入 `valid_dicts_pdf.csv` 与 `valid_dicts_report.csv`。

第二轮冲突处理规则：

- `archiveTime`/`checkTime` 冲突：取最晚时间；
- `badness` 冲突：统一置为 `有`；
- `roomName` 冲突：置空；
- `hp` 冲突：按 `阳性 > 阴性 > 待确认 > 未检` 取值；
- `anesthesiologistName` 冲突：置空；
- `doctorName` 冲突：先剔除含数字值，再取剩余值里长度最长者。

脚本会在每轮开始前检查输出目录是否已有对应 `.jsonl` 缓存：如果存在，则直接读取该轮结果并进入下一轮，避免重复全量扫描。
