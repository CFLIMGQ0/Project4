# temp.py 详细说明

## 1. 脚本职责

`temp.py` 用于检查 EUS/GIST/Leiomyoma 数据集的目录结构，并生成两类输出：

1. 数据集整体统计摘要；
2. 患者级有效性表格。

脚本会基于 `configs/path.yaml` 中维护的路径配置读取数据根目录与输出位置；若命令行显式传入路径参数，则以命令行参数为准。

## 2. 路径配置来源

项目中所有默认路径均统一维护在 `configs/path.yaml`：

- `paths.dataset_root`：原始数据集根目录；
- `paths.output_dir`：输出目录；
- `paths.summary_json`：统计摘要 JSON 输出文件；
- `paths.patient_validity_table`：患者有效性表格输出文件。

因此，若后续调整目录结构，优先修改 `configs/path.yaml`，而不是直接修改脚本中的常量。

## 3. 统计逻辑

### 3.1 患者目录识别

- 数据根目录下的一级子目录会被视为患者目录；
- 患者目录名称默认期望以 `ZS` 开头；
- 若命名不符合约定，不会阻止统计，但会被记录为异常命名患者。

### 3.2 检查目录识别

- 每个患者目录下的直接子目录会被视为一次检查；
- 单次检查目录同时存在 `img/` 与 `pdf/` 时，判定为有效检查；
- 缺少任一子目录时，判定为结构异常检查。

### 3.3 统计结果

脚本会输出两份统计结果：

- **原始统计结果**：基于患者目录下全部直接子目录计算；
- **img/pdf 清洗后的统计**：将缺少 `img` 或 `pdf` 的检查目录视为不存在后重新统计。

统计字段包括：

1. 患者总数；
2. 检查总次数；
3. 空患者目录数；
4. 命名异常患者目录数及样例；
5. 缺少 `img/pdf` 的检查目录数量及样例；
6. 患者检查次数分布。

## 4. 患者有效性表格

脚本还会在输出目录中生成三列表格：

| patient_dir | is_valid | invalid_reason |
| --- | --- | --- |
| 患者目录名 | 1/0 | 无效原因说明 |

判定规则如下：

- `is_valid = 1`：患者目录命名正常，且至少存在一个同时包含 `img/` 与 `pdf/` 的检查目录，并且没有结构异常检查目录；
- `is_valid = 0`：存在以下任一情况：
  - 患者目录命名不符合约定；
  - 患者目录下没有检查目录；
  - 所有检查目录都缺少 `img` 或 `pdf`；
  - 存在部分检查目录缺少 `img` 或 `pdf`。

`invalid_reason` 会按患者聚合异常原因，便于后续筛查与人工清洗。

## 5. 使用方式

### 5.1 使用默认路径配置运行

```bash
python temp.py
```

### 5.2 覆盖 path.yaml 中的路径

```bash
python temp.py \
  --dataset-root /your/dataset/root \
  --save-json /your/output/dataset_summary.json \
  --save-table /your/output/patient_validity.csv
```

### 5.3 指定其他路径配置文件

```bash
python temp.py --config /your/configs/path.yaml
```

## 6. 输出文件说明

默认会生成以下文件：

- `outputs/dataset_summary.json`：保存原始统计、清洗后统计以及患者有效性行数据；
- `outputs/patient_validity.csv`：保存患者级三列表格。

## 7. 维护建议

- 如需修改项目默认路径，请优先更新 `configs/path.yaml`；
- 如需新增更多输出文件，也建议继续在 `configs/path.yaml` 中补充路径参数；
- 若文档提到脚本细节，请统一引用本文件，避免在其他 `.md` 文件中重复展开 `temp.py` 的实现说明。
