# 数据集结构说明

## 1. 数据集根目录

项目中需要区分“数据集根路径”和“实际数据目录”两个概念（并在 `configs/path.yaml` 中分开配置）：

- 数据集根路径：`paths.dataset_base_root`（可放说明文件与附属统计文件）；
- 实际数据目录：`paths.dataset_root`（患者目录实际所在路径）。

脚本默认读取的路径由 `configs/path.yaml` 的 `paths.dataset_root` 指定。

推荐在 `configs/path.yaml` 中显式区分：

- `paths.dataset_base_root`：数据集根路径（说明文件、统计结果、辅助文件）；
- `paths.dataset_root`：实际数据目录（患者目录，脚本读取该目录）。
- `paths.valid_dicts_pdf_csv`：有效检查目录的 PDF 级汇总 CSV（显式路径，建议相对路径）；
- `paths.valid_dicts_report_csv`：有效检查目录的报告级汇总 CSV（显式路径，建议相对路径）；
- `paths.output_dir`：脚本输出根目录；
- `paths.process_cache_dir_name`：`solve_conflicted_pdfs.py` 过程文件目录名（默认 `cache_solve_conflicted_pdfs`）。

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

以下为当前已确认的“有效键 → 页面中文名称”对应关系，并按“重要有效键 / 非重要有效键”拆分。

#### 9.2.1 重要有效键

- `reportTitle` → 页面标题（用于判定报告内容类型；这是标题，不是一个明确打印出来的中文字段名）
- `age` → 年龄
- `badness` → 不良反应
- `condition` → 患者一般情况
- `namePatient` → 姓名
- `operation` → 操作过程
- `operationValue` → 操作名称
- `sex` → 性别
- `suggest` → 注意事项
- `watch` → 内镜所见
- `watchResult` → 诊断
- `specimen` → 活检部位
- `hp` → HP(幽门螺旋杆菌)
- `operationRemark` → 操作过程备注（其值仅当操作过程不顺利时才有可能非空）
- `score` → 波士顿评分

#### 9.2.2 非重要有效键

除上述“重要有效键”外，其余有效键统一归类为“非重要有效键”，具体如下：

- `anesthesiologistName` → 麻醉医生
- `applyDeptName` → 科室
- `applyNo` → 检查号
- `bedId` → 病床号
- `checkTime` → 检查日期
- `doctorName` → 报告医师
- `endoscopeName` → 镜号
- `hisPatientId` → 内镜号
- `narcosisType` → 麻醉方式
- `patientAreaName` → 病区
- `roomName` → 诊间
- `archiveTime` → 报告日期
- `admissionNo` → 由 `patientType` 指定（可为：门诊号、住院号、Z、体检号）
- `patientType` → admissionNo 中文键类型（取值：门诊号、住院号、Z、体检号）

## 10. 唯一性确认输出文件（新增）

当前唯一性确认脚本 `scripts/solve_conflicted_pdfs.py` 默认输出到 `paths.output_dir`（可通过 `--output-dir` 覆盖）。
其中过程文件默认落盘到 `paths.output_dir/paths.process_cache_dir_name`（默认 `cache_solve_conflicted_pdfs`），并按四轮（第一轮 + 第二类 + 第三类 + 第四轮）确认分别落盘：

- 第一轮结果：
  - `valid_dicts_pdf_round1.csv`
  - `valid_dicts_report_round1.csv`
  - `solve_conflicted_pdfs_round1.jsonl`（用于第二轮直接读取，避免重复扫描）
- 第二轮结果：
  - `valid_dicts_pdf_round2.csv`
  - `valid_dicts_report_round2.csv`
  - `solve_conflicted_pdfs_round2.jsonl`
- 第三轮结果：
  - `valid_dicts_pdf_round3.csv`
  - `valid_dicts_report_round3.csv`
  - `solve_conflicted_pdfs_round3.jsonl`
- 第四轮结果（统计 `suggest/watch` 冲突并保留）：
  - `valid_dicts_pdf_round4.csv`
  - `valid_dicts_report_round4.csv`
  - `solve_conflicted_pdfs_round4.jsonl`
- 兼容文件：
  - `valid_dicts_pdf.csv`（写入 `paths.dataset_base_root`，等同第四轮汇总）
  - `valid_dicts_report.csv`（写入 `paths.dataset_base_root`，等同第四轮报告）

第二类唯一性确认（非重要有效键）冲突键规则：

- `archiveTime` / `checkTime`：取最晚时间；
- `roomName`：冲突时置空；
- `anesthesiologistName`：冲突时置空；
- `narcosisType`：冲突时置空；
- `doctorName`：冲突值中先剔除含数字值，再取长度最长者。

第三类唯一性确认（重要有效键）冲突键规则：

- `badness`：冲突时置为 `有`；
- `hp`：按优先级 `阳性 > 阴性 > 待确认 > 未检` 取值。
- `score`：冲突时取分数更大的值；
- `operationValue`：冲突时按逗号拆分多值并合并去重。
- `specimen`：冲突时按部位拆分合并；同一部位出现多个数量时取较大数量并保留全部部位。
- `watchResult`：冲突时按逗号拆分多值并合并去重。

第四轮唯一性确认说明：

- 第四轮统一统计 `suggest` 与 `watch` 冲突：不做唯一值确认，不移除冲突键，仅统计冲突目录数与冲突项数量；
- 第四轮定位为“兼容记录轮次”：完成时会输出“冲突已记录”，将 `suggest/watch` 作为已记录冲突而非未解决冲突；
- 第四轮结束后，若检查目录只剩 `suggest/watch` 冲突，则该检查目录判定为有效检查目录；仅当还有其他键冲突时，才记为“冲突未完全解决”；
- 每轮完成后会生成该轮 `valid_dicts_pdf_roundX.csv` / `valid_dicts_report_roundX.csv` / `solve_conflicted_pdfs_roundX.jsonl`；
- 第四轮输出完成后，脚本会刷新兼容产物 `valid_dicts_pdf.csv` 与 `valid_dicts_report.csv`。

`valid_dicts_pdf_roundX.csv` 新增冲突数量指标：

- `suggest_num`：`suggest` 无冲突时为 `1`，有冲突时记录冲突总数量 `n`（按该键在目录内出现的非空值总次数统计）；
- `watch_num`：`watch` 无冲突时为 `1`，有冲突时记录冲突总数量 `n`（按该键在目录内出现的非空值总次数统计）；
- `conflict_key_types` 中会根据 `suggest_num/watch_num` 展开重复键名，便于直接看到冲突规模。
- `conflict_instance_count`：冲突实例总数（`suggest`/`watch` 按其冲突数量展开计数，其余冲突键按 1 计数）。

该输出用于后续筛选高置信检查目录与构建键值分析样本，并支持按轮次缓存续跑。

## 11. 兼容汇总文件内容说明（`valid_dicts_pdf.csv` / `valid_dicts_report.csv`）

你目前已准备好两份兼容汇总文件：

- `valid_dicts_pdf.csv`：**PDF 级别汇总**，通常是一条 PDF 记录对应一行，主要用于排查冲突来源、定位具体 PDF，以及观察 `suggest_num` / `watch_num` / `conflict_instance_count` 等冲突统计指标。
- `valid_dicts_report.csv`：**检查目录级汇总**，通常是一条“最终确认后的检查目录记录”对应一行，包含用于分析与建模准备的关键字段（例如 `reportTitle`、`namePatient`、`watchResult` 等）。

推荐配置方式（以 `configs/path.yaml` 为准）：

- `paths.valid_dicts_pdf_csv`：显式写入 `valid_dicts_pdf.csv` 路径；
- `paths.valid_dicts_report_csv`：显式写入 `valid_dicts_report.csv` 路径；
- 两者都建议使用**相对路径**（例如 `./dataset/valid_dicts_pdf.csv`），以便迁移环境时不依赖绝对路径。

## 12. reportTitle 数据分布（按报告数量降序）

本节补充当前已统计的胃镜与肠镜 `reportTitle` 分布情况，口径为“报告条数 + 对应图像张数”。

### 12.1 胃（共 17 种 reportTitle）

1. 胃镜手术(住院)报告：报告 842 条，图像 38602 张  
2. 无痛胃镜检查报告：报告 784 条，图像 65056 张  
3. 超声胃镜检查报告：报告 446 条，图像 12844 张  
4. 放大染色胃镜精查报告：报告 291 条，图像 25184 张  
5. 胃镜检查报告：报告 186 条，图像 13800 张  
6. 无痛超声胃镜报告：报告 142 条，图像 6248 张  
7. 胃镜下切除手术报告：报告 124 条，图像 5100 张  
8. 无痛胃镜(含色素内镜)报告：报告 51 条，图像 4293 张  
9. 超声胃镜下手术报告：报告 12 条，图像 534 张  
10. 胃镜下其他手术报告：报告 7 条，图像 246 张  
11. 一诊疗无痛胃镜报告：报告 5 条，图像 381 张  
12. 国际部无痛胃镜检查（含色素内镜）报告：报告 2 条，图像 182 张  
13. 急诊胃镜下取异物报告：报告 2 条，图像 202 张  
14. 急诊胃镜报告：报告 2 条，图像 49 张  
15. 胃镜下静脉曲张手术报告：报告 2 条，图像 92 张  
16. 国际部胃镜检查（含色素内镜）报告：报告 1 条，图像 31 张  
17. 职工体检胃镜(无痛)报告：报告 1 条，图像 82 张

### 12.2 肠（共 11 种 reportTitle）

1. 无痛肠镜检查报告：报告 429 条，图像 33134 张  
2. 肠镜手术(住院)报告：报告 72 条，图像 5470 张  
3. 肠镜检查报告：报告 31 条，图像 2224 张  
4. 肠镜下手术报告：报告 6 条，图像 466 张  
5. 一诊疗无痛肠镜报告：报告 3 条，图像 290 张  
6. 无痛肠镜(含色素内镜)报告：报告 3 条，图像 183 张  
7. 十二指肠镜检查报告：报告 1 条，图像 35 张  
8. 国际部无痛肠镜检查（含色素内镜）报告：报告 1 条，图像 96 张  
9. 国际部肠镜检查（含色素内镜）报告：报告 1 条，图像 45 张  
10. 无痛超声肠镜报告：报告 1 条，图像 37 张  
11. 超声肠镜检查报告：报告 1 条，图像 13 张
- `endoscopeName`：按逗号拆分多值后合并去重；仅剔除“无数字且被更长值完整包含”的泛化项（例如 `肠镜` + `肠镜136` 合并为 `肠镜136`，但 `肠镜13` 与 `肠镜136` 同时保留）。

## 13. reportTitle 统计脚本（新增）

- `scripts/show_reportTitle.py`：统计 `valid_dicts_report.csv` 中 `reportTitle` 的类型及数量。

示例命令：

```bash
python scripts/show_reportTitle.py
```

## 14. reportTitle 同质性分析与分类（基于图像深度特征）

基于 `scripts/check_similarity.py`（ResNet18 提取特征，每类采样上限 300 张），使用三种指标（质心余弦相似度、FID、MMD）对胃镜与肠镜各 reportTitle 类型的图像分布进行两两比较，结果存放于 `outputs/project4/check_similarity/`。

判定标准：
- **高度同质**：余弦相似度 ≥ 0.99 且 MMD ≤ 0.025 且 FID ≤ 0.10
- **同质**：余弦相似度 ≥ 0.97 且 MMD ≤ 0.10 且 FID ≤ 0.20
- **异质**：不满足上述条件

### 15.1 胃镜 reportTitle 同质性分类（17 种 → 4 大类）

#### A 类：常规胃镜检查类

| reportTitle | 样本量 | 与核心的关系 |
|---|---|---|
| 无痛胃镜检查报告 | 300 | 核心 |
| 胃镜检查报告 | 300 | 核心（与无痛胃镜 cos=0.995, MMD=0.017） |
| 无痛胃镜(含色素内镜)报告 | 300 | 核心（与无痛胃镜 cos=0.997, MMD=0.008） |
| 放大染色胃镜精查报告 | 300 | 同质（与无痛胃镜 cos=0.988, MMD=0.031） |
| 一诊疗无痛胃镜报告 | 300 | 同质（与放大染色 cos=0.965, MMD=0.083） |
| 国际部无痛胃镜检查（含色素内镜）报告 | 182 | 同质（与无痛胃镜 cos=0.979, MMD=0.060） |
| 职工体检胃镜(无痛)报告 | 82 | 同质（与国际部无痛 cos=0.978, MMD=0.065） |
| 国际部胃镜检查（含色素内镜）报告 | 31 | 同质（与国际部无痛 cos=0.971, MMD=0.095） |

#### B 类：胃镜手术/治疗类

| reportTitle | 样本量 | 与核心的关系 |
|---|---|---|
| 胃镜手术(住院)报告 | 300 | 核心 |
| 胃镜下切除手术报告 | 300 | 核心（与住院手术 cos=0.997, MMD=0.008） |
| 胃镜下其他手术报告 | 246 | 核心（与住院手术 cos=0.994, MMD=0.018） |
| 急诊胃镜报告 | 49 | 同质（与住院手术 cos=0.971, MMD=0.096） |
| 急诊胃镜下取异物报告 | 202 | 同质（与住院手术 cos=0.963, MMD=0.112） |
| 胃镜下静脉曲张手术报告 | 92 | 边缘（与切除手术 cos=0.947, MMD=0.143；语义属手术类） |

#### C 类：超声胃镜检查类

| reportTitle | 样本量 | 与核心的关系 |
|---|---|---|
| 超声胃镜检查报告 | 300 | 核心 |
| 无痛超声胃镜报告 | 300 | 核心（cos=0.994, MMD=0.015） |

#### D 类：超声胃镜手术类（孤立）

| reportTitle | 样本量 | 说明 |
|---|---|---|
| 超声胃镜下手术报告 | 300 | 与最近邻 C 类 cos=0.958, MMD=0.117；与其他所有类型 cos≤0.91，明显异质 |

#### 胃镜分类小结

- A/B 两大类覆盖 15 种 reportTitle 中的绝大多数报告量；
- C 类为超声检查专用，图像风格与 A/B 均不同（cos ≈ 0.94–0.97）；
- D 类仅 1 种（超声胃镜下手术），与所有其他类型距离最远，需单独处理。

### 15.2 肠镜 reportTitle 同质性分类（11 种 → 4 大类）

#### A 类：常规肠镜检查类

| reportTitle | 样本量 | 与核心的关系 |
|---|---|---|
| 无痛肠镜检查报告 | 300 | 核心 |
| 肠镜检查报告 | 300 | 核心（cos=0.997, MMD=0.011） |
| 无痛肠镜(含色素内镜)报告 | 183 | 核心（与无痛肠镜 cos=0.996, MMD=0.016） |

#### B 类：肠镜手术/治疗类

| reportTitle | 样本量 | 与核心的关系 |
|---|---|---|
| 肠镜手术(住院)报告 | 300 | 核心 |
| 肠镜下手术报告 | 300 | 核心（cos=0.995, MMD=0.017） |
| 国际部肠镜检查（含色素内镜）报告 | 45 | 边缘（与住院手术 cos=0.972, MMD=0.103；虽名含"检查"但三项指标均更接近手术类） |

#### C 类：小样本边缘检查类

| reportTitle | 样本量 | 与核心的关系 |
|---|---|---|
| 一诊疗无痛肠镜报告 | 290 | 核心 |
| 国际部无痛肠镜检查（含色素内镜）报告 | 96 | 核心（与一诊疗 cos=0.980, MMD=0.068） |
| 无痛超声肠镜报告 | 37 | 同质（与一诊疗 cos=0.978, MMD=0.067） |
| 十二指肠镜检查报告 | 35 | 同质（与无痛超声肠镜 cos=0.973, MMD=0.094） |

说明：C 类各成员与 A 类余弦相似度在 0.94–0.96 之间、FID 在 0.16–0.32 之间，分布上已有可区分差异，但样本量普遍偏小（≤290），分类置信度低于 A/B 类。后续若样本量增加，可重新评估是否并入 A 类。

#### D 类：超声肠镜类（孤立）

| reportTitle | 样本量 | 说明 |
|---|---|---|
| 超声肠镜检查报告 | 13 | 与所有类型 cos≤0.876，FID/MMD 因样本过少（<20）被跳过，明显异质 |

#### 肠镜分类小结

- A/B 两大类覆盖主要报告量（无痛肠镜检查 429 + 肠镜手术 72 + 肠镜检查 31 + 肠镜下手术 6 + …）；
- C 类为小样本边缘类型，彼此中度同质但与 A 类有一定差异；
- D 类仅 1 种（超声肠镜检查），样本极少（13 张），需单独处理或后续补充数据后重新评估。
