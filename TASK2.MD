# TASK2: 报告引导的层次化多标签MIL内镜诊断框架

## 目标期刊

二区SCI期刊，候选：*Computers in Biology and Medicine*、*Artificial Intelligence in Medicine*、*IEEE Journal of Biomedical and Health Informatics* 等。

## 硬件配置

3 × NVIDIA RTX 3060 12GB，使用 `DataParallel` 多卡训练，AMP 混合精度。

---

## 一、数据概况

### 1.1 数据规模

| 维度 | 数量 |
|---|---|
| 有效检查目录（样本数） | 3,447 |
| 总图像数 | 约 20+ 万张 |
| 患者数 | 2,275（存在同一患者多次检查） |
| 多次检查患者 | 953（其中 525 人同时有胃镜+肠镜） |
| 数据来源 | 中山医院消化内镜中心 |

### 1.2 报告类型分布（按 reportTitle 同质化划分）

**胃镜**（2,899 份报告，约 17 万张图像）：
- 常规白光胃镜：~980 份（无痛胃镜 784、胃镜 186、其他少量）
- 染色胃镜：~345 份（放大染色 291、含色素内镜 51、其他少量）
- 超声胃镜：~600 份（超声检查 446、无痛超声 142、超声下手术 12）
- 手术胃镜：~986 份（住院手术 841、切除手术 124、其他少量）

**肠镜**（548 份报告，约 4.2 万张图像）：
- 常规白光肠镜：~463 份
- 手术肠镜：~78 份
- 染色/超声肠镜：~7 份

### 1.3 结构化字段（重要有效键）

| 字段 | 中文名 | 内容说明 | 对任务的价值 |
|---|---|---|---|
| `watchResult` | 诊断 | 最终诊断结论，多值用逗号/分号分隔 | **主标签来源**，用于派生多标签分类目标 |
| `watch` | 内镜所见 | 医生对每个解剖部位的详细描述（位置、大小、形态等） | **核心未利用资源**，包含实例级定位信息 |
| `specimen` | 活检部位 | 活检部位和数量（如"胃窦 *1, 食管 *1"） | **关键区域金标准**，2,138/2,899 胃镜有活检记录 |
| `hp` | 幽门螺杆菌 | 阳性 176 / 阴性 973 / 未检 1,750 | 辅助标签，HP阳性是胃炎活动性的强指标 |
| `score` | 波士顿评分 | 肠镜准备质量评分（仅 109 份肠镜有） | 肠镜质量评估标签 |
| `suggest` | 注意事项 | 后续建议（复查、手术、专科就诊等） | 可推断病变严重程度 |
| `age` / `sex` | 年龄 / 性别 | 患者基本信息 | 辅助临床特征 |
| `operationValue` | 操作名称 | 标准化操作名称（多值用 `\|` 分隔） | 区分检查类型（诊断/手术/超声） |
| `reportTitle` | 页面标题 | 报告类型标题 | 用于同质化分组 |

### 1.4 原始诊断分布（从 `watchResult` 统计）

**胃镜诊断 Top 15**（watchResult 去重后有 1,779 种独立诊断词条）：

| 诊断 | 出现次数 | 占胃镜比例 |
|---|---|---|
| 食管SMT（含黏膜下隆起等变体） | 1,385 | 47.8% |
| 食管黏膜病变/肿物（含 SESCC） | 1,509 | 52.1% |
| 胃炎（所有类型） | 1,199 | 41.4% |
| └─ 慢性非活动性胃炎 | 729 | 25.1% |
| └─ 慢性活动性胃炎 | 201 | 6.9% |
| └─ 京都 C-1（轻度萎缩） | 426 | 14.7% |
| └─ 京都 C-2（中度萎缩） | 396 | 13.7% |
| └─ 京都 C-3（重度萎缩） | 102 | 3.5% |
| └─ 京都 O 系列（开放型萎缩） | 50 | 1.7% |
| 胃息肉（含贲门息肉） | 171 | 5.9% |
| 反流性食管炎（LA-A 80 / LA-B 28 / LA-C 6） | 113 | 3.9% |
| 溃疡 | 93 | 3.2% |
| 胆汁反流 | 42 | 1.4% |
| 食管裂孔疝 | 36 | 1.2% |

**手术类型分布**（ESD 710, STER 183, 其他手术少量）

**肠镜诊断**：正常 221 / 息肉 287 / 炎症 22 / 肿物 13 / 憩室 11 / 黑变病 9

**HP 分布**：阳性 176（15.3%）/ 阴性 973（84.7%）/ 未检 1,750

### 1.5 数据的独特性（被现有方法忽略的信息）

1. **`watch` 文本中包含实例级空间定位**：如"距门齿20cm可见一0.8cm黏膜下隆起"——这本质上是对图片内容的**文字级标注**，但现有所有MIL方法都只用了bag-level标签
2. **`specimen` 是活检部位的金标准**：73.7%（2,138/2,899）的胃镜检查有活检记录，医生选择活检的部位就是最有诊断价值的区域
3. **诊断体系本身具有层次结构**：器官→病变类型→严重程度分级（如胃炎→活动/非活动→京都C1/C2/C3），这种层次本身就蕴含了丰富的标签间依赖
4. **同一患者多次检查**：953 名患者有多次检查，481 名有多种胃镜类型
5. **多种检查类型混合**：白光/染色/超声/手术，同一患者可能有多种检查

### 1.6 数据目录结构

```text
datasets/main_data/
└── ZSxxxxxxxx/                  # 患者目录
    ├── ZSxxxxxxxx/              # 检查目录（一次检查）
    │   ├── img/                 # 该次检查的图像文件（几十到上百张）
    │   └── pdf/                 # 该次检查的 PDF 报告
    └── ZSxxxxxxxx/              # 另一次检查
        ├── img/
        └── pdf/
```

### 1.7 CSV 文件字段说明

`valid_dicts_report.csv` 是检查目录级汇总，每行对应一次检查。关键字段：
- `exam_dir`：检查目录绝对路径
- `suggest_num` / `watch_num`：suggest/watch 的冲突版本数
- `img_num`：该次检查的图像数量
- `suggest` / `watch`：多版本用 ` | ` 分隔
- `watchResult`：最终诊断（已做唯一性确认）

---

## 二、任务定义

### 2.1 为什么不是原来的3标签任务

原来的3标签任务（食管SMT / 食管黏膜病变 / 胃炎）对二区投稿有明显不足：

1. **标签数太少（仅3个）**：审稿人会质疑任务难度不够，多标签问题的复杂性体现不出来
2. **胃炎标签过于笼统**：数据中明确有京都分级（C-1/C-2/C-3/O-1/O-2/O-3）和活动性/非活动性区分，这些分级信息被完全浪费
3. **大量有价值的诊断类别被忽略**：胃息肉(171例)、反流性食管炎(113例)、溃疡(93例)等
4. **临床故事弱**：仅预测"有没有胃炎"对临床意义有限，无法为医生提供分级信息
5. **创新点2（标签图推理）发挥空间有限**：3个标签的图结构太简单，无法体现条件图推理的优势

### 2.2 新任务：层次化多标签内镜诊断

**核心改进**：将3标签扩展为**8个层次化多标签**（如表所示），标签之间存在明确的层次依赖关系。

#### 2.2.1 标签体系设计（8标签方案）

| # | 标签名 | 中文说明 | 正例数 | 正例率 | 层次关系 |
|---|---|---|---|---|---|
| 1 | `esophageal_smt` | 食管黏膜下肿物 | 1,385 | 47.8% | **食管层** — 独立标签 |
| 2 | `esophageal_neoplasm` | 食管黏膜病变/肿物 | 1,509 | 52.1% | **食管层** — 独立标签 |
| 3 | `gastritis` | 胃炎（任何类型） | 1,199 | 41.4% | **胃层** — 父标签 |
| 4 | `gastritis_active` | 胃炎活动性 | 201 | 6.9% | **胃层** — `gastritis` 的子标签 |
| 5 | `atrophy_moderate_severe` | 萎缩中重度（京都C2+/O2+） | 488 | 16.8% | **胃层** — `gastritis` 的子标签 |
| 6 | `gastric_polyp` | 胃息肉（含贲门息肉） | 171 | 5.9% | **胃层** — 独立标签 |
| 7 | `reflux_esophagitis` | 反流性食管炎 | 113 | 3.9% | **食管-胃交界** — 独立标签 |
| 8 | `ulcer` | 溃疡 | 93 | 3.2% | **胃/十二指肠层** — 独立标签 |

#### 2.2.2 标签层次依赖关系图

```
食管区域
├── esophageal_smt (47.8%)
├── esophageal_neoplasm (52.1%)
└── reflux_esophagitis (3.9%)

胃区域
├── gastritis (41.4%)
│   ├── gastritis_active (6.9%) ← 子标签：只有 gastritis=1 时才有临床意义
│   └── atrophy_moderate_severe (16.8%) ← 子标签：京都分级 C2+/O2+
├── gastric_polyp (5.9%)
└── ulcer (3.2%)
```

#### 2.2.3 标签共现模式（已从数据验证）

| 共现对 | 次数 | 解读 |
|---|---|---|
| esophageal_neoplasm + gastritis | 637 | 食管病变+胃炎频繁共现（同次检查多处发现） |
| esophageal_smt + gastritis | 562 | 食管SMT+胃炎频繁共现 |
| gastritis + atrophy_moderate_severe | 488 | 中重度萎缩必然伴随胃炎（层次依赖） |
| gastritis + gastritis_active | 201 | 活动性胃炎是胃炎的子类 |
| gastritis + gastric_polyp | 148 | 胃炎+息肉共存 |
| gastritis_active + atrophy_moderate_severe | 134 | 活动性+中重度萎缩 |

**关键观察**：标签共现模式在不同解剖区域有显著差异——食管区域的标签(1,2)主要与胃炎(3)共现但彼此功能独立；胃区域的标签(3,4,5,6)存在明确的层次依赖。这正是**条件标签图推理**的动机。

#### 2.2.4 标签不平衡特征

正例率跨度从 3.2%（溃疡）到 52.1%（食管黏膜病变），存在显著的长尾分布。需要使用 ASL/Focal Loss + pos_weight 处理。

### 2.3 为什么8标签是最优选择

1. **数量适中**：8个标签足够展示多标签问题的复杂性，又不至于因为长尾标签太多导致指标不稳定
2. **每个标签样本量充足**：最小的 `ulcer` 也有93例（正例率3.2%），统计上可靠
3. **层次结构有故事**：`gastritis → gastritis_active / atrophy_moderate_severe` 的层次依赖是天然的创新点——需要建模**父标签为正时子标签才有意义**的约束
4. **跨区域依赖有故事**：食管标签和胃标签之间的共现关系随解剖区域变化
5. **比3标签多2.7倍的标签图边数**：8×8=64 vs 3×3=9，条件图推理的创新点更有发挥空间

### 2.4 形式化定义

- **输入**：一个图像 bag $X = \{x_1, x_2, ..., x_N\}$，$N$ 张图像（$N$ 从几张到 200+ 不等）
- **输出**：层次化多标签预测 $\hat{y} \in \{0, 1\}^8$，8 个二分类标签
- **监督信号**：bag-level 弱标签，从 `watchResult` 字段+正则规则派生
- **层次约束**：$\text{gastritis\_active} = 1 \Rightarrow \text{gastritis} = 1$；$\text{atrophy\_moderate\_severe} = 1 \Rightarrow \text{gastritis} = 1$

### 2.5 MIL 范式

采用 **嵌入式 MIL (Embedding-level MIL)**：
1. 特征提取：CNN backbone 将每张图像编码为特征向量
2. 聚合：注意力机制将变长图像序列聚合为固定维度的 bag 表示
3. 分类：基于 bag 表示做层次化多标签预测

### 2.6 现有代码架构

项目已实现完整的训练框架，位于 `/home/Lim/Project4/src/`：

**主模型** — `GastroLabelGraphMIL`（`model/gastro_label_graph_mil/`）：
- `InstanceEncoder`：backbone（ConvNeXt-Tiny 等）+ 2层GELU MLP projector
- `MultiLabelAttentionMIL`：每个标签独立的 gated attention pooling
- `LabelGraphReasoner`：可学习的标签关联图推理（learnable label tokens → dot-product similarity → softmax → propagation → residual MLP）
- 每标签独立 `nn.Linear(feature_dim, 1)` 分类头

**SOTA 复现**（`sotas/gastro_sota/`）：
- `GastroCLAMSBSOTA`：CLAM 单分支（MLP attention + instance clustering）
- `GastroCLAMMBSOTA`：CLAM 多分支（per-label gated attention + attention diversity）
- `GastroDSMILSOTA`：DS-MIL 双流（critical instance QKV attention）
- `GastroTransMILSOTA`：TransMIL（Transformer encoder + learnable label tokens）
- `GastroDTFDMILSOTA`：DTFD-MIL（double-tier pseudo-bag grouping）

**Baseline**（`baselines/gastro_baseline/`）：
- `GastroAttentionMILBaseline`：标准 ABMIL
- `GastroMeanPoolBaseline`：均值池化
- `GastroMaxPoolBaseline`：最大池化
- `GastroTopKMILBaseline`：Top-K 选择
- `GastroTransformerMILBaseline`：Transformer + ABMIL

**训练基础设施**（`training/`）：
- `Trainer`：统一训练器，支持 AMP、梯度累积、多GPU、early stopping
- `MILBagDataset`：MIL数据集，支持内存/磁盘缓存
- 损失函数：BCE / Focal / ASL（非对称损失）
- 评估指标：macro/micro F1、per-label AUROC/AP/F1、混淆矩阵等

> **代码改动说明**：现有代码中 `num_labels=3`，需改为 `num_labels=8`。标签派生逻辑（`training/data.py` 中的 `build_task_records`）需扩充正则规则以支持8标签。模型结构无需大改（`num_labels` 是参数化的）。

---

## 三、领域经典问题与 SOTA 方案

### 3.1 经典问题

#### 问题 A：MIL 弱监督下关键实例定位不准

- **本质**：只有 bag-level 标签，模型需要自己学习哪些 instance 是关键的
- **表现**：注意力权重经常散布在非病变区域，或集中在信息量不大的正常区域
- **现有方案**：
  - ABMIL 的 gated attention（你的 baseline）
  - CLAM 的 instance clustering（top-k/bottom-k 伪标签 + BCE辅助损失）
  - DS-MIL 的 dual-stream（关键实例作 query 的 cross-attention）
  - DTFD-MIL 的 pseudo-bag grouping（分组投票）
- **局限**：所有方案都是纯视觉的，没有利用任何先验知识来引导实例选择

#### 问题 B：多标签之间的关联建模不足

- **本质**：多个诊断标签之间存在共现/互斥关系以及层次依赖，但多数方法对标签做独立分类
- **表现**：忽略"胃炎活动性只有在胃炎为正时才有意义"这样的层次约束
- **现有方案**：
  - LabelGraphReasoner（可学习图传播）
  - TransMIL 的 learnable label tokens（隐式交互）
  - CLAM-MB 的 attention diversity loss（软约束）
- **局限**：
  - 所有标签图方法学的是**静态的**全局共现关系
  - 标签关系实际上是**条件的**——在不同解剖区域，标签关联模式不同
  - **无一方法处理层次化标签依赖**

#### 问题 C：Bag 内大量冗余/噪声实例

- **本质**：一次内镜检查通常有 50-200+ 张图像，其中大量是正常区域或重复角度
- **表现**：长序列导致注意力分散，有效信号被稀释
- **现有方案**：
  - Top-K 选择（硬过滤）
  - DTFD-MIL 的分组（随机分组后各组小 bag 独立聚合）
  - TransMIL 的 Transformer 全局建模（计算量大）
- **局限**：Top-K 缺乏语义指导，DTFD 分组是随机的无语义

#### 问题 D：特征提取器的域偏差

- **本质**：ImageNet 预训练的 backbone 对内镜图像的特征表达不够好
- **现有方案**：
  - 直接微调 backbone（你的方案）
  - 使用病理 VLP 模型做 feature extractor（PLIP、CONCH、BiomedCLIP）
  - 内镜专用模型（EndoFM，自监督视频预训练）
- **局限**：VLP 模型主要针对病理切片，对内镜有域偏差；自主预训练需要大算力（8×A100+）

#### 问题 E：长尾/不平衡标签分布

- **本质**：8标签中正例率从 3.2%（溃疡）到 52.1%（食管黏膜病变），跨度极大
- **现有方案**：Focal Loss、ASL（非对称损失）、pos_weight 调整
- **局限**：只在损失层面处理，缺乏数据或特征层面的平衡策略

### 3.2 SOTA 方案对照

| 方法 | 会议/期刊 | 核心机制 | 优势 | 不足 |
|---|---|---|---|---|
| **ABMIL** | ICML 2018 | Gated attention pooling | 简洁高效 | 缺乏实例间交互 |
| **CLAM** | Nature BME 2021 | Instance clustering + attention | 有实例级伪监督 | 伪标签噪声大 |
| **DS-MIL** | CVPR 2021 | Dual-stream: instance + bag scoring | 关键实例引导聚合 | argmax选择不稳定 |
| **TransMIL** | NeurIPS 2021 | Transformer + learnable label tokens | 全局实例交互 | 长序列计算量大 |
| **DTFD-MIL** | NeurIPS 2022 | Pseudo-bag grouping + feature distillation | 缓解过拟合 | 分组无语义，随机性大 |
| **MI-Zero** | CVPR 2023 | VLP model + similarity-based MIL | 零样本能力 | 需要大VLP模型，精度低于有监督 |
| **CONCH/PLIP** | Nature Med 2023-24 | 视觉-语言对比预训练 | 强特征提取 | 病理专用，内镜域偏差大，算力需求高 |

### 3.3 SOTA 论文中明确提出的未来方向

1. **"利用临床报告文本作为辅助监督信号"** — 多篇论文提到但无人实现在MIL框架中
2. **"更好的实例采样/选择策略"** — DTFD承认随机分组是局限
3. **"标签关系的动态建模"** — 静态图无法捕捉条件依赖
4. **"多模态信息融合"** — 文本+图像的融合在内镜MIL中几乎空白
5. **"解剖结构感知的特征聚合"** — 利用内镜检查的解剖顺序信息
6. **"层次化标签的建模"** — 现有MIL方法无一处理标签间层次依赖

---

## 四、创新方向设计

### 整体框架名称

**Report-Guided Hierarchical Multi-Label MIL with Conditional Label Graph Reasoning (RG-HMIL)**

论文标题方向：
*"Report-Guided Hierarchical Multi-Label Multiple Instance Learning with Conditional Label Graph Reasoning for Endoscopic Diagnosis"*

### 核心创新思路

利用内镜报告中 **已有但被所有现有方法忽视** 的文本信息（`watch`内镜所见 + `specimen`活检部位），结合**层次化标签体系**的天然结构，实现两个层面的创新：
1. **数据利用层面**：报告文本 → 实例级伪标签 → 引导注意力学习（创新点1）
2. **模型架构层面**：解剖区域感知 → 条件层次图推理 → 多粒度聚合（创新点2）

---

### 创新点 1：Text-Guided Instance Relevance Scoring（报告引导的实例相关性评分）

#### 4.1.1 解决的问题

MIL 中最核心的痛点：**关键实例定位不准**。现有方法全靠纯视觉注意力自己学，缺乏先验知识引导。

#### 4.1.2 核心观察

`watch` 字段包含了极其丰富的实例级定位信息。例如：

> "食管距门齿20cm可见一0.8cm黏膜下隆起，表面光滑，齿状线清晰，40cm过贲门，粘液湖清。高位倒转贲门胃底无异常，胃体下部可见到萎缩黏膜，褪色黏膜呈点片状存在……"

这段文字明确告诉了我们：
- **哪个解剖部位有病变**：食管20cm处有隆起，胃体下部有萎缩
- **哪些部位是正常的**：贲门胃底无异常
- **病变的形态特征**：0.8cm，黏膜下隆起，表面光滑

同时 `specimen` 字段如 `"胃窦 *1, 食管 *1"` 直接标注了医生认为最需要活检的关键部位。

**这些信息本质上就是"文字版的实例级标注"**，但现有所有MIL方法都只用了 `watchResult` 的 bag-level 诊断结论。

#### 4.1.3 技术方案

**Step 1：文本结构化解析（规则+正则，无需大模型）**

从 `watch` 字段中提取结构化发现列表：

```python
# 目标解析格式
findings = [
    {"region": "食管", "position": "距门齿20cm", "lesion": "黏膜下隆起", "size": "0.8cm", "surface": "光滑"},
    {"region": "胃体下部", "position": None, "lesion": "萎缩", "size": None, "surface": "褪色点片状"},
    {"region": "贲门胃底", "position": None, "lesion": None, "status": "无异常"},
]
```

实现方式：
- 按句号/分号分割 `watch` 文本为句段
- 用正则匹配解剖部位关键词（食管/贲门/胃底/胃体/胃角/胃窦/十二指肠球部/降部）
- 用正则匹配病变类型关键词（隆起/糜烂/息肉/萎缩/充血/水肿/溃疡/肿物/狭窄/出血等）
- 用正则匹配"无异常/正常/光滑"等阴性描述
- 从 `specimen` 字段直接解析活检部位（按逗号分割，提取"部位 *数量"格式）

**不需要大模型**，内镜报告的文本模式非常规律和固定。

**Step 2：实例级伪标签生成**

为 bag 中的每张图像生成软伪标签：

```
instance_pseudo_labels = {
    "region_label": one-hot(解剖区域),              # 该图片属于哪个解剖部位
    "relevance_score": float,                        # 该图片的诊断相关性（0~1）
}
```

- **解剖区域标签**：内镜图像在检查过程中大致按解剖顺序排列（食管→贲门→胃底→胃体→胃窦→十二指肠），根据图像在序列中的相对位置 + `watch` 中描述的检查路径进行粗略分配
- **相关性得分**：
  - 图像对应区域在 `watch` 中有病变描述 → 高相关性
  - 图像对应区域在 `specimen` 中出现（做了活检） → 最高相关性
  - 图像对应区域被描述为"无异常" → 低相关性

**Step 3：Instance Relevance Predictor 模块（轻量辅助头）**

```
                                    ┌──────────────────────────┐
instance_features (B, N, D) ───────►│ Instance Relevance Head  │
                                    │ Linear(D, hidden) → ReLU │
                                    │ Linear(hidden, R+1)      │──► region_logits (B, N, R)
                                    │   R = 解剖区域数          │    relevance_logits (B, N, 1)
                                    └──────────────────────────┘
```

- 从 `InstanceEncoder` 输出的特征上接一个轻量MLP辅助头
- 预测两个目标：(a) 解剖区域分类（R 类），(b) 诊断相关性评分（二分类/回归）
- **辅助损失**：
  - `L_region = CrossEntropy(region_logits, pseudo_region_labels)` — 区域分类损失
  - `L_relevance = BCE(relevance_logits, pseudo_relevance_scores)` — 相关性回归损失
- **总损失**：`L_total = L_main + λ_region * L_region + λ_relevance * L_relevance`

**Step 4：Relevance-Aware Attention（相关性感知注意力）**

将预测的 relevance score 注入 attention 机制：

```
# 标准 gated attention
attention_raw = W * (V(x) ⊙ U(x))        # (B, L, N)

# 相关性调制
relevance = sigmoid(relevance_logits)      # (B, N, 1) → (B, 1, N)
attention_modulated = attention_raw + α * log(relevance + ε)

# masked softmax
attention_weights = masked_softmax(attention_modulated, mask)
```

关键：`α` 是可学习参数，初始化为 0（确保初期不干扰正常训练，逐步学习利用相关性信息）。

#### 4.1.4 推理时的处理

**训练时**：使用文本解析的伪标签作为辅助监督
**推理时**：Instance Relevance Predictor 已经学会了从视觉特征预测解剖区域和相关性，**不再需要文本输入**。推理时只需图像，不增加部署复杂度。

#### 4.1.5 为什么这是创新

1. **领域空白**：MIL文献中无任何工作利用结构化报告的内镜所见文本做实例级伪标签引导
2. **解决核心痛点**：直接解决"关键实例定位不准"这个MIL最大挑战
3. **零额外标注成本**：伪标签完全从已有报告文本中自动解析
4. **推理时零开销**：文本引导只在训练时使用，推理只需图像
5. **安全消融**：去掉此模块 = 标准 ABMIL，结果不会比baseline差

#### 4.1.6 万一效果不理想的退路

- 如果文本解析质量不够高 → 只使用 `specimen` 字段（最干净的信号）做活检部位相关性引导
- 如果辅助损失干扰主任务 → 减小 λ 或只在前 N 个 epoch 使用（curriculum schedule）
- 如果解剖区域分配不准 → 去掉区域分类，只保留相关性评分
- 最坏情况：此模块做消融实验对比项，证明"有文本引导比没有好"

---

### 创新点 2：Conditional Hierarchical Label Graph Reasoning（解剖条件的层次化动态标签图推理）

#### 4.2.1 解决的问题

现有 `LabelGraphReasoner` 学习一个 **静态的** 标签共现图。但在8标签体系下，标签关系有两层复杂性：

1. **层次依赖**：`gastritis_active` 和 `atrophy_moderate_severe` 是 `gastritis` 的子标签。父标签为负时，子标签无意义
2. **条件依赖**：
   - 在食管区域实例多时，`esophageal_smt` 和 `esophageal_neoplasm` 强相关
   - 在胃窦区域实例多时，`gastritis` 和 `ulcer` 的关联更强
   - 当某些解剖区域完全没有病变实例时，对应标签间关联应被抑制

静态图无法捕捉这种"在不同解剖上下文中，标签关系不同"的条件依赖性，更无法处理层次约束。

#### 4.2.2 技术方案

**Step 1：解剖区域感知的实例分组（Anatomical-Aware Instance Grouping）**

利用创新点1中 Instance Relevance Predictor 预测的解剖区域标签，将 bag 内的实例按区域分组：

```
bag_instances = {x_1, x_2, ..., x_N}
                        │
                        ▼  (region prediction from 创新点1)
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    Region_esophagus  Region_stomach  Region_duodenum  ...
    {x_2, x_5, x_8}  {x_1, x_3,...}  {x_12, x_15}
```

与 DTFD-MIL 的随机分组相比，这是**语义驱动的分组**——基于解剖结构而非随机划分。

**Step 2：区域级聚合（Region-Level Aggregation）**

对每个解剖区域组，独立做 multi-label attention pooling：

```
region_embed_r = MultiLabelAttention(instances_in_region_r, mask_r)
# 输出: (B, num_labels, feature_dim) per region
```

**Step 3：区域分布特征提取 → 条件向量**

```
region_distribution = [count(region_r) / N for r in regions]  # 各区域实例比例
region_features = concat([region_embed_r.mean(dim=1) for r in regions])  # 各区域平均特征
condition_vector = MLP(concat(region_distribution, region_features))
# 输出: (B, condition_dim)
```

**Step 4：条件层次标签图推理（Conditional Hierarchical Label Graph Reasoning）**

在原 `LabelGraphReasoner` 基础上增加两个关键改进：

```python
class ConditionalHierarchicalLabelGraphReasoner(nn.Module):
    def __init__(self, num_labels, feature_dim, condition_dim, hierarchy_mask):
        # 静态部分（与原 LabelGraphReasoner 相同）
        self.label_tokens = nn.Parameter(...)       # (num_labels, feature_dim)

        # 新增1: 条件调制器
        self.graph_modulator = nn.Sequential(
            nn.Linear(condition_dim, num_labels * num_labels),
            nn.Sigmoid()
        )

        # 新增2: 层次掩码（固定，不学习）
        # hierarchy_mask[i][j] = 1 表示 label_j 是 label_i 的父标签
        # 当父标签预测为负时，抑制子标签的图传播权重
        self.register_buffer('hierarchy_mask', hierarchy_mask)  # (L, L)

        self.refine_mlp = nn.Sequential(...)

    def forward(self, bag_embeds, condition_vector, parent_logits=None):
        # 1. 静态标签图
        norm_tokens = F.normalize(self.label_tokens, dim=-1)
        static_graph = softmax(norm_tokens @ norm_tokens.T / sqrt(d))  # (L, L)

        # 2. 条件调制
        modulation = self.graph_modulator(condition_vector).view(B, L, L)
        dynamic_graph = static_graph.unsqueeze(0) * modulation

        # 3. 层次约束（如果有父标签的预测）
        if parent_logits is not None:
            # parent_logits: (B, L)  父标签激活概率
            # 当 gastritis 概率低时，抑制 gastritis_active 和 atrophy 的传播
            hierarchy_gate = self._compute_hierarchy_gate(parent_logits)  # (B, L, L)
            dynamic_graph = dynamic_graph * hierarchy_gate

        dynamic_graph = dynamic_graph / (dynamic_graph.sum(-1, keepdim=True) + 1e-8)

        # 4. 图传播 + 残差融合
        propagated = torch.bmm(dynamic_graph, bag_embeds)
        refined = bag_embeds + self.refine_mlp(cat(bag_embeds, propagated))
        return refined, dynamic_graph
```

**层次掩码设计**：

```
8标签的层次掩码矩阵 hierarchy_parent (8×8):

                      smt  neo  gas  act  atr  pol  ref  ulc
esophageal_smt         -    0    0    0    0    0    0    0
esophageal_neoplasm    0    -    0    0    0    0    0    0
gastritis              0    0    -    0    0    0    0    0
gastritis_active       0    0    1    -    0    0    0    0   ← 父标签 = gastritis
atrophy_mod_severe     0    0    1    0    -    0    0    0   ← 父标签 = gastritis
gastric_polyp          0    0    0    0    0    -    0    0
reflux_esophagitis     0    0    0    0    0    0    -    0
ulcer                  0    0    0    0    0    0    0    -

当 gastritis 的 sigmoid(logit) < threshold 时:
  → gastritis_active 行/列的传播权重被动态衰减
  → atrophy_moderate_severe 行/列的传播权重被动态衰减
```

**Step 5：多粒度聚合（Multi-Granularity Aggregation）**

```
实例特征 (B, N, D)
        │
        ├─── 全局分支：标准 MultiLabelAttention → global_embed (B, L, D)
        │
        └─── 区域分支：
                ├─ 按解剖区域分组
                ├─ 各区域 MultiLabelAttention → region_embed_r (B, L, D)
                ├─ 拼接各区域 embed → condition_vector
                └─ 各区域 embed 加权聚合 → regional_embed (B, L, D)

global_embed + regional_embed
        │
        ▼
ConditionalHierarchicalLabelGraphReasoner(fused_embed, condition_vector, parent_logits)
        │
        ▼
Per-label classifier → logits (B, 8)
        │
        ▼ (可选：层次约束后处理)
如果 sigmoid(logits[gastritis]) < τ:
    logits[gastritis_active] = -∞
    logits[atrophy_moderate_severe] = -∞
```

#### 4.2.3 为什么这是创新

1. **层次化标签图推理在 MIL 中首次提出**：现有 MIL 标签图方法全部是扁平的、静态的；我们同时引入条件调制 + 层次约束
2. **解剖结构驱动的语义分组**：相比 DTFD-MIL 的随机分组，这是有临床含义的分组策略
3. **多粒度聚合符合临床逻辑**：医生诊断流程就是"逐部位观察局部发现 → 综合判断整体"
4. **与创新点1天然耦合**：region prediction 直接提供分组依据
5. **8标签体系让图推理更有价值**：8个标签有64条边，层次依赖+条件依赖远比3标签的9条边丰富
6. **计算量增加微小**：主要是一个小 MLP（graph modulator）和分区 attention（可并行）

#### 4.2.4 万一效果不理想的退路

- 去掉条件调制 = 原始 LabelGraphReasoner（已有方案）
- 去掉区域分组 = 标准全局聚合 + LabelGraphReasoner
- 只保留区域分组但用静态图 → 仍然是对 DTFD-MIL 的语义分组改进
- 去掉层次约束 = 扁平条件图推理
- 可逐模块消融，确保每个组件都有独立贡献

---

## 五、消融实验设计

### 5.1 主消融矩阵

| 实验 | Text-Guided IRS | Conditional LGR | Hierarchy | Regional Grouping | 说明 |
|---|:---:|:---:|:---:|:---:|---|
| Baseline (ABMIL-8L) | ✗ | ✗ | ✗ | ✗ | 8标签 ABMIL |
| + Static LGR | ✗ | Static | ✗ | ✗ | 静态标签图推理 |
| + Text-Guided IRS only | ✓ | ✗ | ✗ | ✗ | 只加创新点1 |
| + Regional Grouping only | ✗ | ✗ | ✗ | ✓ | 只加语义分组 |
| + Conditional LGR | ✗ | Conditional | ✗ | ✓ | 条件图（无层次） |
| + Hierarchical LGR | ✗ | Conditional | ✓ | ✓ | 条件图+层次约束 |
| **Full Model (RG-HMIL)** | ✓ | Conditional | ✓ | ✓ | 完整模型 |

### 5.2 SOTA 对比

所有方法统一在8标签任务上对比：
- CLAM-SB, CLAM-MB, DS-MIL, TransMIL, DTFD-MIL

### 5.3 标签体系对比（附加实验）

| 标签体系 | Full Model 性能 | 说明 |
|---|---|---|
| 3标签（原始） | ? | 证明扩展标签是有价值的 |
| 8标签（新） | ? | 新方案 |

### 5.4 评价指标

**主指标**：
- macro F1（所有标签等权F1均值）
- micro F1（全局TP/FP/FN）
- per-label AUROC 和 AP

**层次约束相关指标**：
- Hierarchical Consistency Rate：$P(\text{child}=1 | \text{parent}=0) = 0$ 的满足率
- 子标签在父标签为正的子集上的 conditional F1

---

## 六、论文故事线摘要

### 问题

现有内镜图像MIL诊断方法面临三个核心挑战：(1) 弱监督下关键实例定位靠纯视觉注意力，准确性有限；(2) 多标签关联建模是静态的，忽略了解剖结构对标签共现的条件影响；(3) 标签体系是扁平的，无法捕捉诊断标签间的层次依赖关系。

### 观察

临床内镜报告中包含了丰富但被忽视的结构化信息：内镜所见（`watch`）详尽描述了每个解剖部位的发现，活检部位（`specimen`）标注了医生认为最关键的区域。同时，内镜诊断天然具有层次结构（器官→类型→分级），标签间的共现关系随解剖上下文而变化。

### 方法

提出 RG-HMIL 框架，在**层次化8标签多标签任务**上：
1. **Text-Guided Instance Relevance Scoring**：从报告文本中自动解析解剖区域和病变信息，生成实例级伪标签，训练辅助头引导注意力（训练时使用文本引导，推理时零额外开销）
2. **Conditional Hierarchical Label Graph Reasoning**：利用解剖区域信息语义分组实例，动态调制标签图边权重，同时引入层次掩码约束父子标签传播，实现"区域聚合→条件层次图传播→最终分类"的多粒度推理

### 亮点

- 首次在 MIL 框架中利用内镜报告文本作为实例级伪标签引导
- 首次在 MIL 中引入层次化标签图推理
- 不需要额外人工标注（伪标签全部从已有报告自动解析）
- 不需要大模型/VLP 预训练（3×RTX 3060 即可训练）
- 推理时不需要文本输入（与纯视觉方法部署成本相同）
- 8标签体系提供临床分级信息，远超传统3标签方案
- 符合医生"逐部位观察→综合判断→分级诊断"的实际流程
- 每个创新点可独立消融，互不依赖

---

## 七、显存估算与可行性

| 组件 | 显存估算 |
|---|---|
| ConvNeXt-Tiny backbone + projector | ~1.5 GB |
| Instance features (50 张 × 768 dim) | ~0.1 GB |
| Text-Guided Instance Relevance Predictor（轻量 MLP） | ~0.1 GB |
| ConditionalHierarchicalLabelGraphReasoner（MLP + label tokens + 8×8 图） | ~0.05 GB |
| Regional Multi-Label Attention（8标签 × R个区域，复用模块） | ~0.15 GB |
| 训练时梯度 + 优化器状态（AdamW） | ~3-4 GB |
| AMP 混合精度节省 | ~30% |
| **单卡估算** | **~6-8 GB** |
| **3×3060 DataParallel** | **充裕（每卡 12 GB）** |

> 8标签比3标签多出的参数量极小：多5个注意力头（`nn.Linear(D, D)` × 5）+ 标签图从 3×3 变为 8×8。总参数量增加 < 1%。

---

## 八、实施计划（建议顺序）

### Phase 1：标签体系扩展（基础工程）
1. **标签派生规则实现**：在 `training/data.py` 中实现 8 标签的正则派生逻辑
2. **数据验证**：生成标签分布统计，验证标签覆盖率和共现矩阵
3. **基础模型适配**：将所有模型的 `num_labels` 从 3 改为 8，运行 baseline 确认可训练

### Phase 2：创新点1（文本引导）
4. **文本解析器开发**：从 `watch` / `specimen` 字段提取结构化发现（规则+正则）
5. **实例级伪标签生成**：将文本发现映射到图像序列的解剖区域标签和相关性得分
6. **Instance Relevance Predictor 模块实现**：辅助 MLP 头 + 辅助损失
7. **Relevance-Aware Attention 实现**：将相关性得分注入注意力机制
8. **验证创新点1**：对比 baseline，确认文本引导的有效性

### Phase 3：创新点2（条件层次图推理）
9. **解剖区域分组实现**：根据 region prediction 动态分组
10. **ConditionalHierarchicalLabelGraphReasoner 实现**：条件调制 + 层次掩码
11. **Multi-Granularity Aggregation 集成**：全局+区域双分支聚合
12. **验证创新点2**：对比静态 LGR，确认条件层次图推理的有效性

### Phase 4：完整实验与论文
13. **完整消融实验**：按 5.1 矩阵跑全部组合
14. **SOTA 对比实验**：5个 SOTA 在 8标签上的对比
15. **可视化**：注意力热力图、标签图可视化、区域分组可视化、层次约束效果
16. **论文撰写**

---

## 九、标签派生规则（技术参考）

以下为从 `watchResult` 字段派生8个标签的具体匹配规则：

```python
import re

def derive_labels(watchResult: str, hp: str) -> dict:
    wr = watchResult or ''
    labels = {}

    # 1. esophageal_smt: 食管SMT
    labels['esophageal_smt'] = int(
        'SMT' in wr and '食管' in wr
        or ('食管' in wr and '黏膜下' in wr and '隆起' in wr)
    )

    # 2. esophageal_neoplasm: 食管黏膜病变/肿物
    labels['esophageal_neoplasm'] = int(
        '食管黏膜病变' in wr
        or '食管粘膜病变' in wr
        or '食管肿物' in wr
        or 'SESCC' in wr
    )

    # 3. gastritis: 胃炎（任何类型）
    labels['gastritis'] = int('胃炎' in wr)

    # 4. gastritis_active: 活动性胃炎
    labels['gastritis_active'] = int(
        '活动性' in wr and '非活动' not in wr and '胃炎' in wr
    )

    # 5. atrophy_moderate_severe: 萎缩中重度（京都C2+/O2+）
    kyoto_matches = re.findall(r'[（(]([CO])-?(\d)[)）]', wr)
    labels['atrophy_moderate_severe'] = int(
        any(int(grade) >= 2 for _, grade in kyoto_matches)
    )

    # 6. gastric_polyp: 胃息肉（含贲门息肉）
    labels['gastric_polyp'] = int(
        '息肉' in wr and ('胃' in wr or '贲门' in wr)
    )

    # 7. reflux_esophagitis: 反流性食管炎
    labels['reflux_esophagitis'] = int('反流性食管炎' in wr)

    # 8. ulcer: 溃疡
    labels['ulcer'] = int('溃疡' in wr)

    return labels
```

> **注意**：以上规则已在 2,899 份胃镜报告上验证，覆盖率 98.7%（仅37份报告无任何标签命中，均为罕见诊断如"胰管支架植入术"等非诊断类操作记录）。
