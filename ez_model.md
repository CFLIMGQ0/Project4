# 四个 Demo 模型说明（2 个 Baseline + 2 个偏 SOTA/Advanced 模型）

## 1. 先说清楚任务抽象

这套 demo 的核心不是“单张图像分类”，而是 **MIL（Multi-Instance Learning，多实例学习）**。

- 一个 `exam_dir`（检查目录）被当作一个 `bag`
- 目录里的每张图像被当作一个 `instance`
- 标签来自报告文本，因此本质上是 **检查级 / 患者级弱监督**

严格地说，代码当前真正落地的样本单位是“检查目录”，不是“整个患者目录”。  
但因为一个检查目录通常对应一次患者检查，所以你问“图像数量不一的患者如何处理”，在实现层面等价于：

> **如何处理每个 bag 内图像数不一致的问题**

当前 demo 一共跑 4 个模型：

- 2 个 baseline
- 2 个偏 SOTA/advanced 的论文化模型

这里的“偏 SOTA”是指结构上更复杂、更接近论文创新方向，不直接声称它们已经是公开榜单上的真实 SOTA。

---

## 2. 四个模型分别是什么

### 2.1 `demo_gastro_mil_baseline`

**定位**：胃镜任务的基础 baseline  
**任务**：胃镜多标签分类  
**标签**：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

**核心结构**：

- `ResNet50` 作为共享实例编码器
- 对每张图提特征后，经过一个共享投影层
- 使用 `DemoMultiLabelAttentionMIL`
- 每个标签一个独立 attention head
- 每个标签再接一个独立线性分类器

**它解决什么问题**：

- 不要求每张图都有单独标签
- 直接从一组异质胃镜图像里学习 bag-level 多标签预测
- attention 权重还能告诉你“这个标签主要是哪些图在起作用”

**为什么它是 baseline**：

- 结构最标准
- 没有引入专家路由、原型库、关系建模等复杂机制
- 是很适合拿来做对照组的 MIL 主线

---

### 2.2 `demo_gastro_proto_moe_former`

**定位**：胃镜任务的 advanced / 偏 SOTA 模型  
**任务**：胃镜多标签分类

**核心结构比 baseline 多了什么**：

- 双分支实例编码：
  - `local_branch`
  - `global_branch`
- 门控融合：
  - 用 `fusion_gate` 融合局部与全局表征
- 专家路由 MoE：
  - `routing_net` 预测每个 bag 该怎么分配给 4 个专家
  - `experts` 是 4 个专家 MLP
- 关系建模：
  - `DemoRelationEncoder`
  - 当前默认是 `transformer`，层数为 2
- prototype bank：
  - 每个标签有 8 个 prototype
  - 先算 token 与 prototype 的相似度，再汇聚成 bag 级证据
- 双路输出融合：
  - `attention logits + prototype logits`

**额外监督 / 约束**：

- `proto`：原型拉近/拉远约束
- `consistency`：attention 分支和 prototype 分支的一致性约束
- `expert_balance`：防止所有样本都挤到同一个 expert，缓解 routing collapse

**一个很关键的设计**：

胃镜样本在构建时会从报告文本里推断一个 `gastro_subtype_id`：

- `white_light`
- `surgery`
- `stain`
- `ultrasound`

这个 subtype 会作为 routing 的弱先验送入模型，帮助专家路由更稳定。  
实现上不是硬分配，而是把 one-hot 先验和路由网络的软分配做加权融合。

**为什么它更像 SOTA 风格模型**：

- 不再只是“编码 + attention pooling”
- 显式加入了专家分工、原型记忆和实例关系建模
- 更适合论文化描述和做消融实验

---

### 2.3 `demo_colo_mil_baseline`

**定位**：肠镜任务的基础 baseline  
**任务**：肠镜二分类（当前默认 `normal / polyp`）

**核心结构**：

- `ResNet50` 共享编码器
- 共享投影层
- `DemoSingleAttentionMIL`
- 单头 attention 做 bag 聚合
- 最后接一个二分类线性头

**它的特点**：

- 比胃镜 baseline 更简单，因为当前肠镜主任务是二分类
- 输出的是 bag 级 `normal / polyp`
- attention 可以直接看哪些图最像病灶证据

**为什么它是 baseline**：

- 结构直接
- 没有 count-aware、去偏、prototype memory 等附加机制
- 适合作为肠镜任务的最小可运行对照

---

### 2.4 `demo_colo_count_aware_debias_mil`

**定位**：肠镜任务的 advanced / 偏 SOTA 模型  
**任务**：肠镜二分类主任务 + 单发/多发辅助任务

**核心结构比 baseline 多了什么**：

- 双分支编码：
  - `lesion_branch`
  - `context_branch`
- 门控融合：
  - 把病灶特征和上下文特征融合
- 实例级病灶打分：
  - `instance_lesion_scorer`
- Top-k 候选选择：
  - 选最像病灶的 `topk_lesion=8`
  - 选最不像病灶、偏上下文的 `topk_context=8`
- Bag 表征：
  - 把 lesion pool 和 context pool 拼接后再分类
- Count head：
  - 额外预测 `single_polyp / multi_polyp`
  - 只对阳性 bag 有效
- Prototype memory：
  - 分别维护 `normal_prototypes` 和 `polyp_prototypes`
  - 用两类 prototype 的相似度差值修正最终二分类 logit

**额外监督 / 去偏约束**：

- `count`：单发 / 多发辅助监督
- `proto`：normal / polyp 原型对比约束
- `hard_negative`：对正常样本里异常高响应的实例做抑制
- `consistency`：病灶分支和上下文分支输出保持一定一致性

**它为什么重要**：

肠镜里最常见的问题不是“有没有图”，而是：

- 正常背景图很多
- 真正病灶图占比可能很低
- 单发息肉和多发息肉在 bag 结构上也有差异

这个模型就是围绕这些问题设计的，所以它不只是分类，还在做：

- 病灶候选提纯
- 正常背景去偏
- count-aware 辅助建模

**为什么它更像 SOTA 风格模型**：

- 显式区分 lesion / context
- 引入 top-k 候选选择
- 引入 count-aware 辅助头
- 引入 prototype memory 和 hard negative 抑制

---

## 3. 四个模型的关系可以怎么理解

可以把它们看成两条主线，每条主线各有一个 baseline 和一个增强版：

### 胃镜线

- `demo_gastro_mil_baseline`：标准多标签 attention MIL
- `demo_gastro_proto_moe_former`：在标准 MIL 上叠加 MoE、prototype、relation encoder

### 肠镜线

- `demo_colo_mil_baseline`：标准二分类 attention MIL
- `demo_colo_count_aware_debias_mil`：在标准 MIL 上叠加 lesion/context 拆分、top-k、count-aware、prototype、去偏约束

如果一句话总结：

- baseline 负责证明 “MIL 主线本身能不能跑通”
- advanced 模型负责证明 “加入针对数据噪声和模态混杂的结构后，是否能更强、更稳、更可解释”

---

## 4. 你是如何处理“图像数量不一的患者”的

这是这套项目里很关键的一部分。你的处理不是简单把所有患者都裁成固定张数，而是：

> **先保留 bag 的变长本质，再通过采样、补齐、mask 和批次控制去兼容训练**

下面按流程讲。

### 4.1 先把每个检查目录当成一个变长 bag

在 `build_task_records` 里：

- 每个 `exam_dir` 都会被递归收集图像路径
- 图像列表保存到 `image_paths`
- 每个样本天然允许有不同的图像数量

也就是说，你的数据层从一开始就是 **变长 bag**，不是固定长度张量。

---

### 4.2 先做最小图像数过滤

构建样本时会检查：

- 如果 `len(image_paths) < min_instances`
- 这个样本会直接跳过

默认配置里：

- `min_instances: 1`

所以当前逻辑的意思是：

- 至少要有 1 张图才能作为有效 bag

---

### 4.3 对图像太多的 bag 做“按策略采样”

真正进入 `Dataset` 后，不会无上限地把某个 bag 里的所有图都读进来，而是会根据模型配置设定 `max_instances`。

默认配置：

- 胃镜 baseline：`24`
- 胃镜 advanced：`20`
- 肠镜 baseline：`24`
- 肠镜 advanced：`20`

采样策略分训练和验证/测试两套：

- 训练：`random`
- 验证/测试：`uniform`

具体含义：

- `random`：如果图像太多，就随机抽取一部分，增强训练随机性
- `uniform`：如果图像太多，就尽量均匀地从头到尾取样，减少评估波动
- `all_if_small`：代码也支持，但当前默认没启用

这一步解决的是：

- 有些患者/检查图像太多，直接全读会爆显存
- bag 太长时，训练和评估都会变慢

---

### 4.4 训练时还会做随机实例丢弃

在选完索引后，训练集还会做一层 `random_instance_dropout`。

默认配置：

- 胃镜 baseline：`0.05`
- 胃镜 advanced：`0.08`
- 肠镜 baseline：`0.03`
- 肠镜 advanced：`0.08`

作用是：

- 进一步减少模型对固定几张图的过拟合
- 强迫模型学会从不同子集里聚合证据
- 提升对 bag 内冗余图像和背景图像的鲁棒性

---

### 4.5 如果图像太少，会自动补到最小实例数

如果经过采样/丢弃后，实例数少于 `min_instances`：

- 会复用已有索引，重复补齐
- 保证每个 bag 至少满足最小实例数约束

当前默认 `min_instances=1`，所以这一层通常不会特别明显；  
但代码已经把更一般的情况考虑进去了。

---

### 4.6 到 `collate_fn` 再做 batch 内 padding

不同 bag 的图像数仍然不一样，所以在一个 batch 里会这样处理：

- 先找到这个 batch 里最大的实例数 `max_n`
- 建一个零张量 `images[bsz, max_n, c, h, w]`
- 图像少的 bag 只填前面有效位置
- 同时生成一个 `mask[bsz, max_n]`

其中：

- `mask=True` 表示这张图是真实存在的
- `mask=False` 表示这是 padding 出来的空位

所以你的实现不是把所有患者粗暴截成同一个固定长度，而是：

- **batch 内动态补齐**
- **全程保留有效位掩码**

---

### 4.7 模型在聚合时用 mask 忽略 padding

这是最关键的一步，否则 padding 会污染注意力和分类结果。

你的模型里有多处专门处理 `mask`：

- `masked_softmax` 会让 padding 位置不参与 attention 归一化
- `RelationEncoder` 在 Transformer 后还会把无效位乘回 0
- 肠镜 advanced 的 `top-k` 选择只在有效实例范围内做
- prototype 相似度计算也会结合 `mask`

所以最终效果是：

- 同一个 batch 里可以混合不同长度的 bag
- 但模型真正“看到”的只有有效图像
- padding 只负责对齐张量形状，不参与语义计算

---

### 4.8 还做了“按实例总数”控制 batch，防止长 bag 挤爆显存

这一步很容易被忽略，但其实非常实用。

项目里不是只按 `batch_size` 控制一个 batch 有多少个样本，还额外用了 `InstanceAwareBatchSampler`：

- 先估计每个 bag 的实例数
- 再限制一个 batch 的总实例数不能超过 `max_instances_per_batch`

默认配置：

- baseline：`72`
- advanced：`60`

这样做的原因是：

- 即使 `batch_size=3`
- 如果 3 个 bag 都特别长，显存峰值仍然可能很高

所以你实际上做了双重控制：

- 控每个 bag 最多多少张图
- 控每个 batch 总共最多多少张图

这让训练稳定很多。

---

## 5. 一句话总结你的“变长患者图像处理方案”

你的方案可以概括成下面这句：

> **以检查目录为变长 bag，训练时按策略采样实例，batch 内动态 padding，并用 mask 保证 attention、关系建模、prototype 和 top-k 选择都只在有效图像上进行，同时再用实例感知采样器控制显存。**

如果再说得更直白一点：

- 图像多的患者：抽样
- 图像少的患者：保留，必要时补齐
- batch 对齐：padding
- 计算时忽略 padding：mask
- 防止显存炸掉：限制每个 batch 的总实例数

---

## 6. 这四个模型各自最适合什么用途

- `demo_gastro_mil_baseline`
  - 用来验证胃镜多标签任务能否被标准 MIL 跑通
- `demo_gastro_proto_moe_former`
  - 用来验证加入 subtype 先验、专家路由、prototype 和关系建模后能否进一步提升
- `demo_colo_mil_baseline`
  - 用来验证肠镜二分类任务的基础性能
- `demo_colo_count_aware_debias_mil`
  - 用来验证病灶候选选择、单发/多发辅助监督和去偏设计是否有效

如果你后面要写论文或做正式实验，最自然的结构就是：

1. 先报告两个 baseline
2. 再报告两个 advanced 模型
3. 最后专门做“变长 bag 处理策略”和“辅助约束”的消融
