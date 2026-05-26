# 多模态字段与三阶段实验计划

本文记录 TASK2 多模态实验当前准备使用的有效关键键、主要风险，以及后续三阶段逐步加入字段的实验安排。当前目标不是一次性把所有文本/结构化字段都塞进模型，而是按泄漏风险和噪声风险分阶段加入，判断哪些字段真正提升性能，哪些字段只是带来噪声或标签泄漏。

## 一、当前有效关键键

目前多模态只考虑下面 9 个键：

| 键 | 类型 | 初步用途 | 主要风险 |
|---|---|---|---|
| `reportTitle` | 检查标题/类别文本 | 区分胃镜、无痛胃镜、超声胃镜、精查等检查场景 | 可能和检查类型、数据来源强相关 |
| `age` | 数值结构化字段 | 提供年龄相关先验 | 可能只学到人群分布偏差 |
| `sex` | 类别结构化字段 | 提供性别相关先验 | 信号可能很弱，容易无贡献 |
| `hp` | 类别结构化字段 | 与胃炎相关，可能提升 `label_gastritis` | 对食管标签可能是噪声 |
| `operationValue` | 操作/检查类型文本 | 区分普通检查、超声内镜、内镜治疗等流程 | 与病种/治疗路径相关，可能产生偏倚 |
| `specimen` | 标本/活检部位文本 | 提供活检区域弱提示 | 可能暗示病灶位置，存在中等泄漏风险 |
| `score` | 评分文本/数值 | 可能反映检查质量或肠道准备评分 | 当前 TASK2 选中样本中几乎不可用，可能主要是缺失噪声 |
| `suggest` | 建议文本 | 可能包含复查、病理、治疗建议 | 高风险，可能直接包含诊断性提示 |
| `watch` | 检查所见文本 | 与图像发现最接近，理论信息量最大 | 高风险，可能直接出现目标标签关键词 |

明确不把 `watchResult` 作为多模态输入。`watchResult` 是 TASK2 标签派生来源，只能用于生成标签和做泄漏审计。

## 二、当前数据管线状态

当前 `/home/Lim/Project4/datasets/task_data/task2/gastro_multilabel_task_datalist.csv` 已包含：

```text
reportTitle, hp, specimen, watch
```

尚未直接保留：

```text
age, sex, operationValue, score, suggest
```

因此第一阶段开始前，需要先修改 datalist 构建逻辑或在 dataset 读取时按 `exam_dir` 回连源报告表，确保 `age` 和 `sex` 能进入训练 batch。

对 TASK2 已选中 `2872` 条样本回连源报告表后的字段缺失情况如下：

| 键 | 缺失率 | 非缺失唯一值数量 | 备注 |
|---|---:|---:|---|
| `reportTitle` ·| `0.00%` | `17` | 可直接作为类别字段 |
| `age` | `0.00%` | `73` | 可标准化为数值字段 |
| `sex` | `0.00%` | `2` | 可作为类别字段 |
| `hp` | `0.00%` | `3` | 可作为类别字段 |
| `operationValue` | `2.54%` | `46` | 缺失较少，可做类别/短文本字段 |
| `specimen` | `26.01%` | `353` | 缺失较多，必须加缺失 mask |
| `score` | `100.00%` | `0` | 当前 TASK2 样本不可用，阶段二中只保留接口或暂不启用 |
| `suggest` | `0.21%` | `1697` | 文本长且强提示风险高 |
| `watch` | `0.00%` | `2867` | 几乎每条不同，信息量高但泄漏风险也高 |

## 三、当前面临的问题

1. 字段贡献未知：`reportTitle`、`age`、`sex` 可能提供稳定先验，但提升幅度不一定大；`watch`、`suggest` 信息量强，但可能让模型变成读报告而不是看图。
2. 字段噪声不同：`score` 在当前 TASK2 选中样本中全缺失，`specimen` 缺失约四分之一，长文本字段还包含模板化内容。
3. 泄漏风险不同：`watch` 和 `suggest` 可能包含 `食管SMT`、`胃炎`、`黏膜病变` 等目标词，必须单独标注为报告辅助或高风险文本实验。
4. 字段相关性强：`reportTitle` 与 `operationValue` 可能高度重叠，`specimen` 与 `watch` 也可能共享病灶部位信息。
5. 部署定义要清晰：如果测试阶段输入 `watch` 或 `suggest`，结果不能和 image-only 或低泄漏结构化模型放在同一主排名里。
6. 数据管线尚未完整：第一阶段需要的 `age`、`sex` 当前不在 datalist 中，需要先补齐字段传递。

## 四、三阶段安排

三个阶段采用逐步累加方式：后一阶段默认继承前一阶段字段，再加入新字段。每阶段都要和 `exp6_long_mil_64_no_roi` image-only 基线比较，并记录字段来源、推理输入和泄漏级别。

### 阶段 1：低风险基础字段

| 项目 | 内容 |
|---|---|
| 阶段名 | `phase1_basic_demographic_title` |
| 使用键 | `reportTitle`、`age`、`sex` |
| 推理输入 | 图像 + 低风险结构化/标题字段 |
| 泄漏级别 | 低 |
| 目标 | 先验证最干净、最容易部署的多模态信号是否能提升稳定性 |

推荐实验：

```text
exp8_p1_reportTitle_age_sex
```

建议编码方式：

| 键 | 编码 |
|---|---|
| `reportTitle` | 类别 embedding 或短文本 embedding |
| `age` | 标准化数值 + 缺失 mask |
| `sex` | 类别 embedding |

阶段 1 的判断标准：

- 如果 `macro_f1`、`subset accuracy` 均提升，可以把这三个字段作为后续默认基础多模态字段。
- 如果只提升验证集、不提升测试集，需要检查 `reportTitle` 是否带来数据分布偏差。
- 如果无提升，后续阶段仍保留阶段 1 作为对照，但不强制作为最终模型输入。

### 阶段 2：中风险临床/操作/标本字段

| 项目 | 内容 |
|---|---|
| 阶段名 | `phase2_clinical_operation_specimen` |
| 使用键 | 阶段 1 + `hp`、`operationValue`、`specimen`、`score` |
| 推理输入 | 图像 + 标题/人口学 + 临床结构化/半结构化字段 |
| 泄漏级别 | 低到中 |
| 目标 | 判断临床上下文、操作类型、活检部位是否能提升三标签分类 |

推荐实验：

```text
exp8_p2_basic_hp_operation_specimen_score
```

注意事项：

- `hp` 可能主要影响 `label_gastritis`，需要单独看每标签收益。
- `operationValue` 可能和检查路径强相关，若提升异常大，需要做字段置乱实验。
- `specimen` 缺失较多，必须使用缺失 mask，不能把空值简单当作普通文本。
- `score` 当前 TASK2 选中样本中为全缺失，第一轮可以只保留解析接口，不参与主训练；如果后续数据补齐，再单独启用。

建议阶段 2 内部做两个消融：

```text
exp8_p2_no_specimen
exp8_p2_no_operationValue
```

这样可以判断提升来自较稳的结构化字段，还是来自可能有泄漏风险的标本/操作字段。

### 阶段 3：高信息量报告文本字段

| 项目 | 内容 |
|---|---|
| 阶段名 | `phase3_report_text` |
| 使用键 | 阶段 2 + `suggest`、`watch` |
| 推理输入 | 图像 + 全部候选多模态字段 |
| 泄漏级别 | 中到高 |
| 目标 | 评估报告所见和建议文本能提供多少上限收益，并识别文本泄漏风险 |

推荐实验：

```text
exp8_p3_all_keys_watch_suggest
```

阶段 3 必须分两类汇报：

| 类型 | 说明 |
|---|---|
| `report_assist` | 测试阶段允许输入 `watch`、`suggest`，作为图像+报告辅助任务 |
| `train_time_distill` | 训练阶段使用 `watch`、`suggest` 作为 teacher 或对齐信号，测试阶段只输入图像 |

建议文本处理：

- 对 `watch`、`suggest` 做目标关键词 mask，对比 mask 前后性能差异。
- 保留一份原文实验作为报告辅助上限，但不能作为 image-only 主结果。
- 抽样检查高置信预测，确认模型没有简单依赖目标标签原词。

## 五、每阶段统一评估方式

每个阶段至少保存：

```text
config.yaml
log.csv
test_result.csv
test_report.csv
field_audit.csv
missing_rate.json
checkpoints/
```

每个阶段记录：

| 项目 | 要求 |
|---|---|
| 主比较基线 | `exp6_long_mil_64_no_roi` |
| 主指标 | `macro_f1` |
| 辅助指标 | `micro_f1`、`macro_auc`、`subset_accuracy`、每标签 F1/AUC/PR-AUC |
| 字段审计 | 记录每个字段的缺失率、唯一值数量、是否参与推理 |
| 泄漏审计 | 记录是否使用 `watch`、`suggest`，是否做关键词 mask |
| 消融方式 | 阶段全字段实验 + 关键字段 leave-one-out |

建议汇总表：

| 阶段 | 实验名 | 字段 | 推理输入 | 泄漏级别 | `macro_f1` | `subset_accuracy` | 结论 |
|---|---|---|---|---|---:|---:|---|
| baseline | `exp6_long_mil_64_no_roi` | 无 | image | none | 待填 | 待填 | 待填 |
| phase1 | `exp8_p1_reportTitle_age_sex` | `reportTitle,age,sex` | image+structured | low | 待填 | 待填 | 待填 |
| phase2 | `exp8_p2_basic_hp_operation_specimen_score` | phase1 + `hp,operationValue,specimen,score` | image+structured/text | low-mid | 待填 | 待填 | 待填 |
| phase3 | `exp8_p3_all_keys_watch_suggest` | phase2 + `suggest,watch` | image+report text | mid-high | 待填 | 待填 | 待填 |

## 六、当前优先事项

1. 先补齐 datalist 或 dataset 的字段传递，让 `reportTitle`、`age`、`sex` 能进入 batch。
2. 先实现阶段 1，验证低风险多模态管线是否稳定。
3. 阶段 2 再加入 `hp`、`operationValue`、`specimen`，`score` 暂时作为缺失字段记录。
4. 阶段 3 单独处理 `suggest`、`watch`，并把结果标注为报告辅助或训练期蒸馏，避免和严格可部署模型混排。

