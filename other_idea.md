# 已删除模型思路总结

本文档用于保留当前已从代码中移除的模型方案，方便后续回看结构设计和创新点。下面这些方案目前**只保留思路说明，不再保留实现代码**。

## 胃镜方向

### 1. `demo_gastro_proto_moe_former`

原始定位：

- 面向胃镜三标签多标签分类的重型增强模型。

主要结构：

1. 共享 backbone 编码。
2. `local_branch` 与 `global_branch` 双分支特征抽取。
3. `fusion_gate` 对局部和全局特征做门控融合。
4. `routing_net + experts` 组成 MoE 路由结构。
5. `relation_encoder` 做实例间关系建模。
6. `MultiLabelAttentionMIL` 生成标签级 bag 表征。
7. prototype bank 对每个标签提供原型证据。
8. 分类结果由 attention logits 与 prototype logits 相加得到。

附加约束：

- `prototype_pull_push_loss`
- `consistency_loss`
- `expert_balance_loss`

创新理念：

- 希望把“多标签相关性”“胃镜模式差异”“原型证据约束”和“专家分工”全部放进一个统一框架。
- 核心思路是让不同专家去吸收不同病灶模式，同时用原型库约束标签级证据来源。

当前不保留原因：

- 结构过重，变量过多。
- 路由、原型、关系建模和双分支同时存在，训练与调参成本较高。
- 当前阶段更适合保留一个更直接、更聚焦的标签相关模型。

### 2. `demo_gastro_subtype_deconf_mil`

原始定位：

- 在已知胃镜亚型信息时，做共享表示与去混杂建模。

主要结构：

1. backbone 编码。
2. `shared_proj` 统一投影。
3. `relation_encoder` 建模实例关系。
4. `MultiLabelAttentionMIL` 做标签级聚合。
5. `subtype_embed` 与 `subtype_proj` 生成亚型混杂表示。
6. 对 bag 表征做正交投影式去混杂。
7. 多标签分类头输出结果。

附加约束：

- `subtype_orth`，用于抑制 bag 表征对亚型方向的依赖。

创新理念：

- 把白光、染色、超声、手术等胃镜模态差异看作显式 confounder。
- 通过“减去 subtype 投影分量”来弱化亚型偏置，让标签表征更聚焦病灶本身。

当前不保留原因：

- 当前正式任务并未把亚型信息作为稳定监督信号纳入数据流程。
- 相比直接标签关系建模，这条路线对先验依赖更强，落地复杂度更高。

### 3. `demo_gastro_topk_hybrid_mil`

原始定位：

- 同时利用全局 attention 和稀疏 top-k 证据的胃镜增强模型。

主要结构：

1. backbone 编码。
2. `shared_proj` 特征投影。
3. `relation_encoder` 做实例关系建模。
4. `MultiLabelAttentionMIL` 产生全局标签级聚合。
5. `topk_pool` 从每个标签的高响应实例中抽取稀疏证据。
6. `mix_gate` 融合全局表征与 top-k 表征。
7. 多标签分类头输出结果。

附加约束：

- `sparse_attn`，鼓励 attention 更聚焦。

创新理念：

- 假设真正正证据只集中在少量关键帧。
- 用 top-k 稀疏证据补足纯全局 attention 的“平均化”问题。
- 通过门控把全局鲁棒性和局部强证据结合起来。

当前不保留原因：

- 这条线更偏“证据聚合方式增强”，但当前阶段更想优先保留标签空间建模最明确的方案。
- 与 `gastro_label_graph_mil` 相比，问题指向没有那么集中。

## 肠镜方向

### 4. `demo_colo_count_aware_debias_mil`

原始定位：

- 面向肠镜二分类的复杂增强模型，同时兼顾息肉计数偏差与背景混杂。

主要结构：

1. backbone 编码。
2. `lesion_branch` 与 `context_branch` 双分支建模病灶与背景。
3. `instance_lesion_scorer` 计算实例病灶分数。
4. top-k 病灶候选池与 top-k 背景候选池。
5. `binary_head` 输出主二分类结果。
6. `count_head` 额外预测单发 / 多发。
7. 正常原型与息肉原型做 prototype similarity 约束。

附加约束：

- `count`
- `proto`
- `hard_negative`
- `consistency`

创新理念：

- 一方面用显式 lesion/context 分离减弱背景误导；
- 另一方面用 prototype 和记数监督提升稀疏病灶识别能力。

当前不保留原因：

- 当前项目正式任务只保留肠镜二分类。
- 该结构里“计数感知”部分天然与后续三分类 / 多发性建模更强绑定，当前阶段不适合作为主线。

### 5. `demo_colo_noisy_or_mil`

原始定位：

- 面向稀疏阳性 bag 的肠镜二分类增强模型。

主要结构：

1. backbone 编码。
2. `shared_proj` 特征投影。
3. `relation_encoder` 建模实例关系。
4. `instance_head` 输出每张图的阳性概率。
5. 使用 Noisy-OR 公式将实例概率聚合成 bag 概率。

关键公式：

- `P(B=1) = 1 - Π_i (1 - p_i)`

创新理念：

- 假设“只要有极少数关键病灶帧为真，整个 bag 就应为阳性”。
- 用概率层面的 Noisy-OR 替代普通 attention/平均池化，更偏向提升召回。

当前不保留原因：

- 方案很干净，但当前阶段肠镜只保留 baseline 作为主线。
- 后续若要重新做“稀疏召回优先”实验，这个方向仍然值得单独拿出来。

### 6. `demo_colo_ds_mil`

原始定位：

- DSMIL 风格的关键实例锚定模型。

主要结构：

1. backbone 编码。
2. `shared_proj` 特征投影。
3. `relation_encoder` 建模实例关系。
4. `instance_head` 先找出最关键实例。
5. 用关键实例生成 query，其余实例生成 key。
6. 通过 query-key 相似度做关系加权聚合。
7. bag logit 与关键实例 logit 融合输出。

附加约束：

- `consistency`，约束 bag 级判断与关键实例判断一致。

创新理念：

- 不是直接对整包做平均权重，而是先锚定最可疑实例，再围绕它聚合上下文。
- 更符合“息肉只出现在少数关键帧”的实际观察。

当前不保留原因：

- 思路明确，但仍属于肠镜增强分支，不是当前保留主线。

### 7. `demo_colo_counterfactual_mil`

原始定位：

- 基于病灶 top-k 与背景 top-k 显式相减的反事实模型。

主要结构：

1. backbone 编码。
2. `shared_proj` 特征投影。
3. `relation_encoder` 建模实例关系。
4. `instance_head` 给出每帧病灶响应。
5. 从高分区域抽 lesion top-k，从低分区域抽 context top-k。
6. 构造 `counterfactual = lesion - lambda * context`。
7. 主分类头基于 lesion / context / counterfactual 联合输出。
8. `count_head` 兼容单发 / 多发计数监督。

附加约束：

- `separation`
- `count`

创新理念：

- 明确把背景看成一种需要被减掉的反事实分量。
- 通过“正证据减背景证据”压制正常黏膜和无关纹理带来的伪阳性。

当前不保留原因：

- 方案更适合在肠镜增强路线中单独展开验证。
- 现阶段项目只保留 baseline，不继续保留这条复杂支线实现。

## 当前结论

- 胃镜方向当前保留 `gastro_baseline + gastro_label_graph_mil`。
- 肠镜方向当前只保留 `colonoscopy_baseline`。
- 上述 7 个方案的结构与创新点已经保留在本文档中，代码实现已从项目中移除。
