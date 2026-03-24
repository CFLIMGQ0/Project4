# 数据集结构说明

## 1. 数据集根目录

项目中需要区分“数据集根路径”和“实际数据目录”两个概念（并在 `configs/path.yaml` 中分开配置）：

- 数据集根路径：`paths.dataset_base_root`（可放说明文件与附属统计文件）；
- 实际数据目录：`paths.dataset_root`（患者目录实际所在路径）。

脚本默认读取的路径由 `configs/path.yaml` 的 `paths.dataset_root` 指定。

推荐在 `configs/path.yaml` 中显式区分：

- `paths.dataset_base_root`：数据集根路径（说明文件、统计结果、辅助文件）；
- `paths.dataset_root`：实际数据目录（患者目录，脚本读取该目录）。

因此，若需要切换数据环境、迁移服务器或调整输出位置，请优先修改 `configs/path.yaml`。

## 2. 患者级目录结构

每个患者对应一个目录，目录名当前预期为：

```text
ZSxxxxxxxx
```

其中：

- `ZS` 为当前使用的固定前缀；
- 后续字符通常为患者编号；
- 若历史数据中存在不完全一致的命名，可先保留，后续再根据清洗规则逐步处理。

## 3. 检查级目录结构

每个患者目录下可以包含 `1~n` 个检查目录，每个检查目录表示该患者的一次检查。

根目录与子目录关系可概括为：

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

其中：

- `paths.dataset_base_root` 是配置文件中的“数据集根路径”字段；
- `paths.dataset_root` 是配置文件中的“实际数据目录”字段；
- `paths.dataset_base_root` 下可以同时存放说明文件和统计产物，不建议放在 `paths.dataset_root` 内；
- 这里的 `xxxxxxxx` 表示数字部分；
- 患者目录名与检查目录名中的 `xxxxxxxx` 可能并不相同，应以原始数据中的实际目录名为准；
- `img/`：保存该次检查产生的图片；
- `pdf/`：保存该次检查对应的报告文件。

## 4. 当前工作重点

项目当前主要围绕**数据结构检查与多轮清洗准备**展开，重点包括：

1. 统计患者总数；
2. 统计总检查次数；
3. 观察患者检查次数分布；
4. 识别目录结构不完整的检查数据；
5. 为后续逐轮清洗保留可扩展空间。

## 5. 当前任务边界

- 当前阶段先处理数据集本身；
- 分类任务、建模任务与标签体系暂未确定，统一视为“暂定”；
- 本文档仅描述数据组织方式，不展开具体脚本实现细节。

## 6. 使用建议

- 在进行下一步清洗前，先确认数据目录结构是否一致；
- 若需要调整默认数据根目录或输出路径，请先修改 `configs/path.yaml`；
- 若后续需要新增更多统计维度，可在现有数据处理流程上继续扩展。

## 7. 清洗脚本执行顺序与执行效果

为保证目录结构清洗的一致性，建议固定按以下顺序执行：

1. `python scripts/delete_broken_files.py`
2. `python scripts/delete_empty_dicts.py`
3. `python scripts/delete_broken_dicts.py`
4. `python scripts/delete_empty_patients.py`

执行效果说明：

- 第一步（`delete_broken_files.py`）：在文件级扫描 PDF 与图片完整性，优先定位损坏文件，避免后续目录判断被坏文件干扰。
- 第二步（`delete_empty_dicts.py`）：在检查目录内核查 `img/`、`pdf/` 子目录是否真正包含对应类型文件，清理空目录或类型不匹配目录。
- 第三步（`delete_broken_dicts.py`）：在检查目录级识别“仅有 img 或仅有 pdf”的结构缺损目录，并进行删除，保证检查目录结构完整性。
- 第四步（`delete_empty_patients.py`）：在患者目录级删除已经清洗为空的患者目录，保持患者列表干净。

该顺序对应“先文件、后子目录、再检查目录、最后患者目录”的逐层收敛流程，可降低误删风险并提升清洗可复现性。

## 8. PDF 文件概念说明

- **冗余 PDF**：在同一次检查里，若多个 PDF 表达的信息一致，那么除保留参考用的那一份外，其余都定义为冗余 PDF。
- **冲突 PDF**：在同一次检查里，若不同 PDF 的同名键出现不同的非空值，则这些互相矛盾的 PDF 定义为冲突 PDF。

补充：本节只用于统一“冗余/冲突”术语含义，不展开统计流程与去重实现细节。

## 9. check_pdf 键有效性分组规则（用于后续统计口径）

基于当前 `check_pdf` 统计结果，按“**非空次数是否为 0**”将键分为两类：

- **无效键**：非空次数 = 0（该键在现有样本中始终为空值）；
- **有效键**：非空次数 > 0。

后续若无特殊说明，所有统计数据默认仅统计**有效键**，**无效键不再纳入统计范围**。

### 9.1 无效键（非空次数 = 0）

- `Signal`
- `badnessRemark`
- `bed`
- `biopsy`
- `checkDoctorSign1`
- `codeImage`
- `conditionRemark`
- `dactor`
- `date`
- `department`
- `endoscopy`
- `inspect`
- `inspect_time`
- `markImage`
- `outpatient`
- `pathology`
- `patient`
- `photo`
- `pics`
- `project`
- `project_name`
- `proposal`
- `under_diagnosis`
- `under_see`

### 9.2 有效键（非空次数 > 0）

- `admissionNo`
- `age`
- `anesthesiologistName`
- `applyDeptName`
- `applyNo`
- `archiveTime`
- `badness`
- `bedId`
- `checkTime`
- `condition`
- `doctorName`
- `endoscopeName`
- `hisPatientId`
- `hp`
- `namePatient`
- `narcosisType`
- `operation`
- `operationRemark`
- `operationValue`
- `patientAreaName`
- `patientType`
- `reportTitle`
- `roomName`
- `score`
- `sex`
- `specimen`
- `suggest`
- `watch`
- `watchResult`

## 10. 有效检查目录输出文件（新增）

当前有效目录判定脚本 `scripts/solve_conflicted_pdfs.py` 会输出以下两个 CSV（默认写入 `paths.dataset_base_root`）：

- `valid_dicts_pdf.csv`
  - 第 1 列：检查目录路径；
  - 第 2 列：目录有效性（有效=1，无效=0）；
  - 第 3 列：冲突键数量；
  - 第 4 列：冲突键类别（使用 `|` 连接）。
- `valid_dicts_report.csv`
  - 第 1 列：有效检查目录路径；
  - 第 2 列开始：该有效目录补充完成后的有效键值。

该输出用于后续筛选高置信检查目录与构建键值分析样本。
