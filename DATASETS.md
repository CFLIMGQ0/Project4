# 数据集说明文档（SCI-4 阶段）

## 1. 数据集概览
本项目采用患者级目录管理影像数据，并使用患者级 Excel 台账维护结构化字段。

当前你已将历史修改后的 `all_patients_raw.xlsx` 正式重命名为 `all_patients.xlsx` 并作为主数据文件使用。

---

## 2. Excel 文件命名约定（重点更新）
为避免混淆，项目中明确区分两个名称：

- `all_patients.xlsx`：**当前主文件（推荐/默认）**，后续脚本与流程均应以该文件为准；
- `all_patients_raw.xlsx`：**历史文件名（兼容用途）**，仅用于兼容旧脚本、旧文档或历史记录。

在当前仓库中，`check_data.py` 已更新为默认优先读取 `all_patients.xlsx`，并兼容回退到 `all_patients_raw.xlsx`。

---

## 3. 数据位置说明
- 原始数据集目录（仓库外部）：`/home/Lim/.cache/kagglehub/datasets/eus_dataset`
- 数据集根目录参数（`--dataset-root`）建议填写：`/home/Lim/.cache/kagglehub/datasets/eus_dataset/`
- 患者信息 Excel 参数（`--excel-path`）建议填写：`/home/Lim/.cache/kagglehub/datasets/eus_dataset/all_patients.xlsx`
- 代码项目目录：`/home/Lim/projects/eus-gist-leiomyoma`
- 训练输出根目录：`/home/Lim/outputs/eus-gist-leiomyoma`

`check_data.py` 默认尝试顺序：
1. `{dataset-root}/all_patients.xlsx`
2. `{dataset-root}/all_patients_raw.xlsx`
3. 若未提供 `dataset-root`，再尝试仓库内与缓存目录常见路径（先新名后旧名）。

---

## 4. all_patients.xlsx 当前字段说明（按你提供的表头更新）
当前主表包含以下核心字段（节选按逻辑分组）：

### 4.1 患者基础信息
- `name`：患者姓名
- `sex`：性别
- `age`：年龄
- `exam_no`：检查号
- `ward_bed_no`：病区床号
- `outpatient_no`：门诊号
- `medical_record_no`：病案号
- `exam_date`：检查日期
- `department`：科室
- `machine_type`：设备型号

### 4.2 报告文本与病灶描述
- `report_text_raw`：原始报告文本
- `lesion_location_raw` / `lesion_location_std`：病灶部位原始值/标准化值
- `bulge`、`surface`、`color`：镜下形态描述
- `origin_layer_raw` / `origin_layer_std`：来源层次原始值/标准化值
- `echo_type`、`echo_homogeneous`：回声类型及均匀性
- `lesion_size_raw`、`boundary`：病灶大小原始描述与边界
- `lesion_long_mm`、`lesion_short_mm`：长径/短径（毫米）

### 4.3 诊断与标签
- `endoscopic_diagnosis_raw`：镜下诊断原文
- `diagnosis_std`：标准化诊断
- `diagnosis_uncertain`：是否不确定（0/1）

### 4.4 图像数量（本次校验重点）
- `wls_count`：白光图像计数
- `eus_count`：EUS 图像计数

---

## 5. 患者目录图像组织与计数规则
单患者目录通常满足：
- `1.png`：报告图（不计入 `wls_count` / `eus_count`）
- `2~n.png`：WLS 或 EUS 或混合

校验逻辑（与 `check_data.py` 一致）：
1. 优先根据文件名关键字识别：
   - 包含 `wls`/`wle`/`white` 记为 WLS
   - 包含 `eus` 记为 EUS
2. 若文件名无法识别（常见纯数字命名）：
   - 序号为 `2` 的图片按 WLS 计数
   - 其余 `3~n` 按 EUS 计数
3. `1.png` 固定视为报告图，不参与 WLS/EUS 数量统计。

---

## 6. check_data.py 当前核验能力（已更新）
当前脚本已支持两类核验：

1. **姓名一致性核验**：
   - 比较 Excel 中患者姓名与目录患者姓名是否一致；
2. **图像数量一致性核验**：
   - 对每位患者比对 `wls_count`、`eus_count` 与目录真实计数是否一致。

脚本输出包含：
- 仅 Excel 存在的姓名；
- 仅目录存在的姓名；
- 计数字段为空/非法行数；
- WLS/EUS 数量不一致患者明细。

示例命令：

```bash
python scripts/data_cleaning/check_data.py \
  --dataset-root /home/Lim/.cache/kagglehub/datasets/eus_dataset/ \
  --excel-path /home/Lim/.cache/kagglehub/datasets/eus_dataset/all_patients.xlsx
```

---

## 7. 当前阶段使用边界（SCI-4）
- 报告截图用于核验与信息整理，不直接作为当前基线模型输入；
- 白光图像用于 WLS 单模态任务；
- EUS 图像用于 EUS 单模态任务；
- 同一患者必须进行患者级划分，避免数据泄漏。
