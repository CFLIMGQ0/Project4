# 数据集结构说明

## 项目目录约定

当前项目根目录固定为 `/home/Lim/Project4`，目录分工如下：

- `src/`：工作环境目录，代码、脚本、说明文档与配置文件位于此目录；
- `datasets/`：数据集根目录；
- `outputs/`：输出根目录；
- `pre_weights/`：预训练模型权重目录；

文档中的 `python scripts/...` 命令默认在 `/home/Lim/Project4/src` 下执行；若在项目根目录执行，请将脚本路径写成 `python src/scripts/...`。

## 1. 数据集根目录

项目中需要区分“数据集根路径”和“实际数据目录”两个概念（并在 `configs/path.yaml` 中分开配置）：

- 数据集根路径：`paths.dataset_base_root`（可放说明文件与附属统计文件）；
- 实际数据目录：`paths.dataset_root`（患者目录实际所在路径）。

脚本默认读取的路径由 `configs/path.yaml` 的 `paths.dataset_root` 指定。

推荐在 `configs/path.yaml` 中显式区分：

- `paths.project_root`：项目根目录（`src`、`datasets`、`outputs`、`pre_weights` 所在位置）；
- `paths.dataset_base_root`：数据集根路径（说明文件、统计结果、辅助文件）；
- `paths.dataset_root`：实际数据目录（患者目录，脚本读取该目录）；
- `paths.valid_dicts_pdf_csv`：有效检查目录的 PDF 级汇总 CSV（显式路径，当前环境建议使用绝对路径）；
- `paths.valid_dicts_report_csv`：有效检查目录的报告级汇总 CSV（显式路径，当前环境建议使用绝对路径）；
- `paths.output_dir`：脚本输出根目录；
- `paths.process_cache_dir_name`：`combine_reports.py` 过程文件目录名（默认 `cache_combine_reports`）。
- `paths.check_similarity_dir_name`：`temp_check_similarity.py` 输出子目录名（位于 `paths.output_dir` 下，默认 `check_similarity`）。

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

## 4. 数据主要处理

项目当前主要围绕**数据结构检查与多轮清洗准备**展开。

### 4.1 数据主要处理内容

1. 统计患者总数；
2. 统计总检查次数；
3. 观察患者检查次数分布；
4. 识别目录结构不完整的检查数据；
5. 为后续逐轮清洗保留可扩展空间。

### 4.2 当前任务边界

- 当前阶段先处理数据集本身；
- 分类任务、建模任务与标签体系暂未确定，统一视为“暂定”；
- 本文档仅描述数据组织方式，不展开具体脚本实现细节。

### 4.3 使用建议

- 在进行下一步清洗前，先确认数据目录结构是否一致；
- 若需要调整默认数据根目录或输出路径，请先修改 `configs/path.yaml`；
- 文档中的命令默认在 `/home/Lim/Project4/src` 下执行；
- 若后续需要新增更多统计维度，可在现有数据处理流程上继续扩展。

## 5. 清洗脚本执行顺序与执行效果

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

## 6. PDF 文件概念说明

- **冗余 PDF**：在同一次检查里，若多个 PDF 表达的信息一致，那么除保留参考用的那一份外，其余都定义为冗余 PDF。
- **冲突 PDF**：在同一次检查里，若不同 PDF 的同名键出现不同的非空值，则这些互相矛盾的 PDF 定义为冲突 PDF。

补充：本节只用于统一“冗余/冲突”术语含义，不展开统计流程与去重实现细节。

## 7. 键有效性分组规则

报告中的键按“**非空次数是否为 0**”将键分为两类：

- **无效键**：非空次数 = 0（该键在现有样本中始终为空值）；
- **有效键**：非空次数 > 0。

后续若无特殊说明，所有统计数据默认仅统计**有效键**，**无效键不再纳入统计范围**。

### 7.1 无效键（非空次数 = 0）

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

### 7.2 有效键（非空次数 > 0）

以下为当前已确认的“有效键 → 页面中文名称”对应关系，并按“重要有效键 / 非重要有效键”拆分。

#### 7.2.1 重要有效键

- `reportTitle` → 页面标题（用于判定报告内容类型；这是标题，不是一个明确打印出来的中文字段名）
- `age` → 年龄
- `badness` → 不良反应
- `condition` → 患者一般情况
- `namePatient` → 姓名
- `operation` → 操作过程
- `operationValue` → 操作名称（多值统一使用 `|` 分隔；清洗后会去除末尾操作编码括号）
- `operationRemark` → 操作过程备注（其值仅当操作过程不顺利时才有可能非空）
- `sex` → 性别
- `suggest` → 注意事项
- `watch` → 内镜所见
- `watchResult` → 诊断
- `specimen` → 活检部位
- `hp` → HP(幽门螺旋杆菌)
- `score` → 波士顿评分

#### 7.2.2 非重要有效键

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

## 8. 唯一性确认输出文件

当前唯一性确认脚本 `scripts/combine_reports.py` 默认输出到 `paths.output_dir`（可通过 `--output-dir` 覆盖）。
其中过程文件默认落盘到 `paths.output_dir/paths.process_cache_dir_name`（默认 `cache_combine_reports`），并按四轮（第一轮 + 第二类 + 第三类 + 第四轮）确认分别落盘：

- 第一轮结果：
  - `valid_dicts_pdf_round1.csv`
  - `valid_dicts_report_round1.csv`
  - `combine_reports_round1.jsonl`（用于第二轮直接读取，避免重复扫描）
- 第二轮结果：
  - `valid_dicts_pdf_round2.csv`
  - `valid_dicts_report_round2.csv`
  - `combine_reports_round2.jsonl`
- 第三轮结果：
  - `valid_dicts_pdf_round3.csv`
  - `valid_dicts_report_round3.csv`
  - `combine_reports_round3.jsonl`
- 第四轮结果（统计 `suggest/watch` 冲突并保留）：
  - `valid_dicts_pdf_round4.csv`
  - `valid_dicts_report_round4.csv`
  - `combine_reports_round4.jsonl`
- 兼容文件：
  - `valid_dicts_pdf.csv`（写入 `paths.output_dir/paths.process_cache_dir_name`，等同第四轮汇总）
  - `valid_dicts_report.csv`（写入 `paths.dataset_base_root`，等同第四轮报告）

第一类唯一性确认（冗余 PDF 信息补全与目录级唯一化）：

- 先在同一检查目录内识别冗余 PDF，并对可互补字段执行信息补全，形成更完整的候选报告；
- 若该检查目录不存在冲突 PDF，则直接将补全后的结果作为该检查目录的唯一确认结果；
- 若存在冲突 PDF，则仅完成冗余信息补全，并将冲突键保留至后续轮次继续处理。

第二类唯一性确认（非重要有效键）冲突键规则：

- `archiveTime` / `checkTime`：取最晚时间；
- `roomName`：冲突时置空；
- `anesthesiologistName`：冲突时置空；
- `narcosisType`：冲突时置空；
- `doctorName`：冲突值中先剔除含数字值，再取长度最长者。
- `endoscopeName`：按逗号拆分多值后合并去重；仅剔除“无数字且被更长值完整包含”的泛化项（例如 `肠镜` + `肠镜136` 合并为 `肠镜136`，但 `肠镜13` 与 `肠镜136` 同时保留）。

第三类唯一性确认（重要有效键）冲突键规则：

- `badness`：冲突时置为 `有`；
- `hp`：按优先级 `阳性 > 阴性 > 待确认 > 未检` 取值。
- `score`：冲突时取分数更大的值；
- `operationValue`：冲突时按逗号拆分多值并合并去重；如后续执行 `clean_values.py`，会进一步统一写成 `操作1|操作2|...`，并去除每个操作末尾的编码括号，仅保留操作名称。
- `specimen`：冲突时按部位拆分合并；同一部位出现多个数量时取较大数量并保留全部部位。
- `watchResult`：冲突时按逗号拆分多值并合并去重。

第四轮唯一性确认说明：

- 第四轮统一统计 `suggest` 与 `watch` 冲突：不做唯一值确认，不移除冲突键，仅统计冲突目录数与冲突项数量；
- 第四轮定位为“兼容记录轮次”：完成时会输出“冲突已记录”，将 `suggest/watch` 作为已记录冲突而非未解决冲突；
- 第四轮结束后，若检查目录只剩 `suggest/watch` 冲突，则该检查目录判定为有效检查目录；仅当还有其他键冲突时，才记为“冲突未完全解决”；
- 每轮完成后会生成该轮 `valid_dicts_pdf_roundX.csv` / `valid_dicts_report_roundX.csv` / `combine_reports_roundX.jsonl`；
- 第四轮输出完成后，脚本会刷新兼容产物：`valid_dicts_pdf.csv`（过程目录）与 `valid_dicts_report.csv`（数据集根目录）。

`valid_dicts_pdf_roundX.csv` 新增冲突数量指标：

- `suggest_num`：`suggest` 无冲突时为 `1`，有冲突时记录冲突总数量 `n`（按该键在目录内出现的非空值总次数统计）；
- `watch_num`：`watch` 无冲突时为 `1`，有冲突时记录冲突总数量 `n`（按该键在目录内出现的非空值总次数统计）；
- `conflict_key_types` 中会根据 `suggest_num/watch_num` 展开重复键名，便于直接看到冲突规模。
- `conflict_instance_count`：冲突实例总数（`suggest`/`watch` 按其冲突数量展开计数，其余冲突键按 1 计数）。

该输出用于后续筛选高置信检查目录与构建键值分析样本，并支持按轮次缓存续跑。

## 9. 兼容汇总文件内容说明（`valid_dicts_pdf.csv` / `valid_dicts_report.csv`）

该项目目前已准备好两份兼容汇总文件：

- `valid_dicts_pdf.csv`：**PDF 级别汇总**，仅保留在过程目录 `paths.output_dir/paths.process_cache_dir_name`，不再写入数据集根目录。
- `valid_dicts_report.csv`：**检查目录级汇总**，写入数据集根目录。第四轮会额外写入 `suggest_num`、`watch_num`、`img_num`，并将多值 `suggest/watch` 以 `watch1 | watch2 | ...` 形式拼接。

推荐配置方式（以 `configs/path.yaml` 为准）：

- `paths.valid_dicts_pdf_csv`：显式写入 `valid_dicts_pdf.csv` 路径；
- `paths.valid_dicts_report_csv`：显式写入 `valid_dicts_report.csv` 路径；
- 两者都建议使用**相对路径**（例如 `./dataset/valid_dicts_pdf.csv`），以便迁移环境时不依赖绝对路径。

## 10. reportTitle 同质化划分结果（含报告数量与图像数量）

根据当前同质化计算结果，按“胃镜/肠镜”分别划分如下：

### 10.1 胃镜同质化划分

1. **常规白光胃镜**
   - 无痛胃镜检查报告（报告 784，图像 65055）
   - 胃镜检查报告（报告 186，图像 13800）
   - 一诊疗无痛胃镜报告（报告 5，图像 381）
   - 职工体检胃镜(无痛)报告（报告 1，图像 82）
   - 急诊胃镜下取异物报告（报告 2，图像 202）

2. **染色胃镜**
   - 放大染色胃镜精查报告（报告 291，图像 25184）
   - 无痛胃镜(含色素内镜)报告（报告 51，图像 4293）
   - 国际部无痛胃镜检查（含色素内镜）报告（报告 2，图像 182）
   - 国际部胃镜检查（含色素内镜）报告（报告 1，图像 31）

3. **手术胃镜**
   - 胃镜手术(住院)报告（报告 842，图像 38602）
   - 胃镜下切除手术报告（报告 124，图像 5100）
   - 胃镜下其他手术报告（报告 7，图像 246）
   - 急诊胃镜报告（报告 2，图像 49）

4. **超声胃镜**
   - 超声胃镜检查报告（报告 446，图像 12844）
   - 无痛超声胃镜报告（报告 142，图像 6248）

5. **其他**
   - 胃镜下静脉曲张手术报告（报告 2，图像 92）
   - 超声胃镜下手术报告（报告 12，图像 534）

### 10.2 肠镜同质化划分

1. **常规白光肠镜**
   - 无痛肠镜检查报告（报告 429，图像 33134）
   - 肠镜检查报告（报告 31，图像 2224）
   - 一诊疗无痛肠镜报告（报告 3，图像 290）
   
2. **染色肠镜**
   - 无痛肠镜(含色素内镜)报告（报告 3，图像 183）
   - 国际部无痛肠镜检查（含色素内镜）报告（报告 1，图像 96）
   - 国际部肠镜检查（含色素内镜）报告（报告 1，图像 45）

3. **手术肠镜**
   - 肠镜手术(住院)报告（报告 72，图像 5470）
   - 肠镜下手术报告（报告 6，图像 466）

4. **超声肠镜**
   - 无痛超声肠镜报告（报告 1，图像 37）
   - 超声肠镜检查报告（报告 1，图像 13）

备注：
- 若后续同质性阈值（余弦/FID/MMD）调整，建议同步更新上述分组与数量。
