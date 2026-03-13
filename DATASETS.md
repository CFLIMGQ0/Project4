# 数据集说明文档（SCI-4 阶段）

## 1. 数据集说明
本项目数据采用**患者级目录结构**组织。总体上：

- 数据集根目录下包含多个患者文件夹；
- 每个患者文件夹通常对应一个患者级样本包；
- 单个患者目录通常包含以下三类图像：
  1. 报告截图
  2. 白光胃镜图（WLE）
  3. 若干 EUS 静态图

本文件用于说明当前阶段（SCI-4）可用数据形态与使用边界，不代表数据已完成标准化清洗。

---

## 2. 数据位置说明
- 原始数据集目录（仓库外部）：`/home/Lim/.cache/kagglehub/datasets/eus_dataset`
- 数据集根目录参数（`--dataset-root`）建议填写：`/home/Lim/.cache/kagglehub/datasets/eus_dataset/`
- 患者信息 Excel 参数（`--excel-path`）固定为：`/home/Lim/.cache/kagglehub/datasets/eus_dataset/all_patients_raw.xlsx`
- 脚本默认会优先尝试读取：`{dataset-root}/all_patients_raw.xlsx`；若使用上述根目录则可不传 `--excel-path`。
- 代码项目目录：`/home/Lim/projects/eus-gist-leiomyoma`
- 训练输出根目录：`/home/Lim/outputs/eus-gist-leiomyoma`

执行 `check_data.py` 时推荐至少传入 `--dataset-root`，脚本会自动寻找 Excel：

```bash
python scripts/data_cleaning/check_data.py \
  --dataset-root /home/Lim/.cache/kagglehub/datasets/eus_dataset/
```

如需覆盖默认行为，也可显式传入：

```bash
python scripts/data_cleaning/check_data.py \
  --dataset-root /home/Lim/.cache/kagglehub/datasets/eus_dataset/ \
  --excel-path /home/Lim/.cache/kagglehub/datasets/eus_dataset/all_patients_raw.xlsx
```

原则：
1. 原始数据不应直接提交到 GitHub 仓库；
2. 项目仓库内部仅管理代码、文档与派生文件；
3. 后续生成的 `manifest`、`metadata`、`split` 文件应存放在仓库内规范目录（如 `data/manifests`、`data/metadata`、`data/splits`）。
4. 每次模型训练结果应输出到 `/home/Lim/outputs/eus-gist-leiomyoma` 下的独立训练子目录。

---

## 3. 数据集根目录下的患者信息组织形式
数据集根目录由多个患者文件夹组成。患者文件夹名称通常包含：

- 检查日期
- 患者姓名
- 病案号或编号（若存在）

示例（仅作结构示意）：
- `2021-11-26; 黄文焕; 661923`
- `2022-01-07; 何可诚; 1191386`
- `2022-02-22; 周祥明; 1196583`

补充说明：
- 可能存在仅包含“日期+姓名”、缺少编号的目录；
- 也可能存在命名分隔符或格式不完全一致的情况；
- **实际情况以原始数据为准**。

---

## 4. 单个患者目录下的文件组织形式
一个患者文件夹通常对应一个患者级样本，目录内常见文件形态：

- `1.png`：1 张报告截图
- `2.png`：1 张白光胃镜图（WLE）
- `3.png` ~ `n.png`：多张 EUS 静态图

注意：
- 个别病例可能存在文件数量差异、命名偏差或缺失；
- 对这类差异应在 metadata/manifest 中显式记录，不应在文档中假设全部样本完全规范；
- **实际情况以原始数据为准**。

---

## 5. 当前已知的数据组织形式
基于当前已知信息，通常可归纳为：

1. 报告截图：通常 1 张，常见命名 `1.png`；
2. 白光胃镜图：通常 1 张，常见命名 `2.png`；
3. EUS 图像：通常 3 张及以上，常见命名 `3~n.png`，数量不固定。

说明：以上为“常见形式”，并非强约束规则，实际情况以后续清点结果为准。

---

## 6. 各类文件在当前阶段的作用
在 SCI-4 阶段，各类文件用途定义如下：

- **报告截图**：用于人工核对、后续元信息整理与标签辅助确认；当前不作为图像模型输入。
- **白光胃镜图（WLE）**：用于“白光单模态分类”任务。
- **EUS 图像**：用于“EUS 单模态分类”任务。

---

## 7. 当前阶段（SCI-4）的数据使用范围
SCI-4 仅围绕以下两项任务：

1. 白光胃镜图单模态分类；
2. EUS 图像单模态分类。

边界约束：
- 不使用文本模态；
- 不做多模态融合训练；
- 不把报告截图直接输入当前基线模型。

---

## 8. 数据整理原则
1. **患者为最小管理单位**：manifest 与 split 均以患者为主键组织；
2. **同患者同划分**：同一患者全部图像必须归属同一数据子集；
3. **严禁患者级泄漏**：禁止同患者图像出现在训练集与验证/测试集的不同子集；
4. **统一元数据管理**：后续通过 metadata/manifest 维护路径、标签、模态完整性等信息；
5. **原始文件只作数据源**：不假设原始数据已完成标准化命名或清洗。

---

## 9. 后续建议整理的元数据字段（建议项）
以下字段为建议方案，当前不代表已全部完成：

- `patient_id`
- `raw_folder_name`（原始文件夹名）
- `exam_date`（检查日期）
- `patient_name`（患者姓名，后续可按规范去标识化）
- `record_no`（病案号/编号，如存在）
- `pathology_label`（病理标签）
- `report_image_path`（报告截图路径）
- `wle_image_path`（白光图路径）
- `eus_image_paths`（EUS 图路径列表）
- `eus_image_count`（EUS 图数量）
- `is_multimodal_complete`（是否具备双模态）
- `split`（`train/val/test`）
- `notes`（异常情况或人工备注）

---

## 10. 当前数据文档中尚未确定 / 待补充信息
以下内容当前统一标记为“待补充”：

- 样本总数
- 标签分布
- 缺失模态情况
- 病理标签获取方式
- 纳入/排除标准
- 图像筛选标准
- 匿名化与去标识化规则

---

## 11. 与后续任务的关系
- **SCI-4（当前）**：数据说明 + 两个单模态任务（白光、EUS）。
- **SCI-3 / SCI-2（后续）**：可在 SCI-4 数据组织基础上扩展更复杂建模方案。

本文件当前以 SCI-4 为中心，不展开后续阶段实现细节。


---

## 12. 训练输出目录约定（新增）
为支持后续持续建模，统一约定：

- 输出根目录固定为：`/home/Lim/outputs/eus-gist-leiomyoma`；
- 每次训练创建一个新子目录，命名为“编号+训练目录名”；
- 建议格式：`{阿拉伯数字序号}_{xxx}`，其中 `xxx` 可根据后续实验内容再指定。例如：
  - `1_eus_baseline_resnet18`
  - `2_wle_baseline_efficientnet_b0`
  - `3_eus_ablation_aug_v2`

每次训练目录内建议至少包含：
- `config.yaml`（训练配置快照）
- `train.log`（训练日志）
- `checkpoints/`（模型权重）
- `metrics/`（验证/测试指标）

说明：
1. 编号必须递增，确保实验顺序可追溯；
2. 即使是同一实验重跑，也建议使用新编号，并在名称中标注 `rerun` 或 `seed`；
3. 该输出目录不纳入 Git 版本管理（建议通过 `.gitignore` 管理）。

---

## 13. 预处理结果文件约定（新增）
当前预处理脚本 `scripts/preprocessing.py` 生成两类结果文件：

1. **患者目录内报告文件**：`report.csv`
   - 每位患者各 1 份；
   - 采用纵向键值结构（`字段`,`值`）；
   - 包含报告抽取字段（姓名、性别、年龄、检查号、病区-床号、门诊号、病案号、检查日期、申请科室、机型、诊断描述、镜下诊断、检查图象）与总结字段（类别四分类、WLS/EUS 是否存在及数量）。

2. **全局汇总文件**：`patient_summary.csv`
   - 输出于 `--output-dir` 指定目录；
   - 每位患者一行；
   - 用于后续患者级统计、筛选与数据划分。

说明：脚本会先进行 report/wle/eus 模态识别，再对报告图执行 OCR 并抽取字段；若 OCR 失败，相关字段可能为空，需人工复核。
