# TODO_LIST: RG-HMIL 模型实现指南

> 本文档用于指导 Codex/后续对话逐步完成 TASK2.MD 中设计的 RG-HMIL 模型。
> 每个章节相对独立，可按顺序逐步完成。

---

## 一、我们的模型：RG-HMIL 完整架构规格

### 1.0 总览

**模型全称**：Report-Guided Hierarchical Multi-Label MIL with Conditional Label Graph Reasoning (RG-HMIL)

**核心数据流**（前向传播路径）：

```
输入: images (B, N, 3, H, W) + mask (B, N)
  │
  ▼
[InstanceEncoder] ── backbone(ConvNeXt-Tiny) + 2层MLP projector
  │
  ▼
instance_features: (B, N, D=512)
  │
  ├─── [InstanceRelevancePredictor] ──→ region_logits (B, N, R)
  │         │                            relevance_logits (B, N, 1)
  │         │
  │         ▼
  │    辅助损失: L_region + L_relevance (仅训练时，用文本伪标签监督)
  │
  ├─── [全局分支] ──→ [RelevanceAwareMultiLabelAttention]
  │         │              (注意力被 relevance_logits 调制)
  │         ▼
  │    global_embed: (B, L=8, D=512)
  │    global_attn: (B, L=8, N)
  │
  └─── [区域分支]
          │
          ├─ 用 region_logits 做 soft/hard 分组 → R 个区域子集
          │
          ├─ 每个区域: [MultiLabelAttention] → region_embed_r: (B, L, D)
          │
          ├─ 区域分布特征 → condition_vector: (B, condition_dim)
          │
          └─ 各区域 embed 加权聚合 → regional_embed: (B, L, D)

  global_embed + regional_embed → fused_embed: (B, L=8, D=512)
  │
  ▼
[ConditionalHierarchicalLabelGraphReasoner]
  │   输入: fused_embed + condition_vector + parent_logits(可选)
  │   静态图 × 条件调制 × 层次掩码 → 动态图传播
  │
  ▼
refined_embed: (B, L=8, D=512)
  │
  ▼
[Per-label Classifiers] ── 8 × nn.Linear(512, 1)
  │
  ▼
logits: (B, 8)
```

**总参数约束**：在现有 `GastroLabelGraphMIL` 基础上新增参数 < 5%，显存 < 8GB/卡。

---

### 1.1 模块 1：InstanceEncoder（已有，无需改动）

**位置**：`model/gastro_label_graph_mil/modules.py` — `InstanceEncoder` 类

**功能**：将 bag 中每张图像编码为特征向量。

**结构规格**：
```
输入: images (B, N, 3, H, W)
  → reshape → (B*N, 3, H, W)
  → backbone: ConvNeXt-Tiny (pretrained, out_dim=768)
       → nn.Linear(768, 512) → ReLU → Dropout(0.1)
  → reshape → (B, N, 512)
  → projector: Linear(512, 512) → GELU → Dropout(0.2) → Linear(512, 512) → GELU
  → 输出: (B, N, 512)
```

**参数**：`backbone_name="convnext_tiny"`, `pretrained=True`, `freeze_stages=0 或 1`, `feature_dim=512`, `dropout=0.2`

**无需任何改动**，新模型继承复用此模块。

---

### 1.2 模块 2：InstanceRelevancePredictor（创新点1 — 新增模块）

**功能**：从 instance 特征预测两个辅助目标——解剖区域分类 + 诊断相关性评分。训练时用文本伪标签监督，推理时无需文本。

**结构规格**：

```python
class InstanceRelevancePredictor(nn.Module):
    """
    轻量辅助头，从 instance_features 预测解剖区域和诊断相关性。
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256,
                 num_regions: int = 6, dropout: float = 0.1):
        # num_regions = 6: 食管 / 贲门胃底 / 胃体 / 胃角胃窦 / 十二指肠 / 其他
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.region_head = nn.Linear(hidden_dim, num_regions)     # 区域分类
        self.relevance_head = nn.Linear(hidden_dim, 1)            # 相关性评分

    def forward(self, instance_features, mask):
        """
        输入:
            instance_features: (B, N, feature_dim=512)
            mask: (B, N) bool
        输出:
            region_logits: (B, N, num_regions=6)
            relevance_logits: (B, N, 1)
        """
        h = self.shared(instance_features)          # (B, N, hidden_dim)
        region_logits = self.region_head(h)          # (B, N, 6)
        relevance_logits = self.relevance_head(h)    # (B, N, 1)
        return region_logits, relevance_logits
```

**解剖区域定义**（R=6）：

| 索引 | 区域 | 中文 | 胃镜检查中的典型顺序位置 |
|---|---|---|---|
| 0 | esophagus | 食管 | 序列前段（~0%–20%） |
| 1 | cardia_fundus | 贲门胃底 | ~20%–30% |
| 2 | gastric_body | 胃体 | ~30%–50% |
| 3 | antrum_angle | 胃角胃窦 | ~50%–70% |
| 4 | duodenum | 十二指肠 | ~70%–85% |
| 5 | other | 其他/不明 | 回退类别 |

**辅助损失规格**：

```python
# 在模型 forward 中计算，通过 outputs["aux_losses"] 返回给 Trainer
aux_losses = {}

# 1. 区域分类损失 (CrossEntropy，忽略无标签的 instance)
#    pseudo_region_labels: (B, N) int, 值 0-5, 无标签位置填 -100
region_loss = F.cross_entropy(
    region_logits.reshape(-1, num_regions),
    pseudo_region_labels.reshape(-1),
    ignore_index=-100,
)
aux_losses["region_cls"] = region_loss

# 2. 相关性损失 (BCE，忽略无标签的 instance)
#    pseudo_relevance: (B, N) float, 值 0-1, 无标签位置填 -1
valid_rel = pseudo_relevance >= 0
if valid_rel.any():
    rel_loss = F.binary_cross_entropy_with_logits(
        relevance_logits.squeeze(-1)[valid_rel],
        pseudo_relevance[valid_rel],
    )
    aux_losses["relevance"] = rel_loss
```

**Trainer 配置**：
```yaml
aux_loss_weights:
  region_cls: 0.5     # λ_region
  relevance: 0.5      # λ_relevance
```

**关键设计约束**：
- `hidden_dim` 不超过 256（保持轻量）
- 推理时此模块仍然执行 forward 以提供 `region_logits` 和 `relevance_logits` 给后续模块使用，但不再计算辅助损失（因为推理时没有伪标签）
- 训练时伪标签通过 `batch["pseudo_region_labels"]` 和 `batch["pseudo_relevance"]` 传入（需扩展 `MILBagDataset` 和 `mil_collate_fn`）

---

### 1.3 模块 3：RelevanceAwareMultiLabelAttention（创新点1 — 改造模块）

**功能**：在标准 GatedAttention 的基础上，用 relevance_logits 调制注意力分数。

**改造位置**：基于现有 `model/common/pooling.py` 的 `GatedAttention`，新建子类或增加参数。

**结构规格**：

```python
class RelevanceAwareMultiLabelAttention(nn.Module):
    """
    在 MultiLabelAttentionMIL 基础上增加 relevance 调制。
    """
    def __init__(self, in_dim: int, attn_dim: int, num_labels: int,
                 dropout: float = 0.1):
        self.attn = GatedAttention(in_dim, attn_dim, num_heads=num_labels, dropout=dropout)
        # 可学习调制强度，每个标签一个，初始化为 0（=不干扰）
        self.alpha = nn.Parameter(torch.zeros(num_labels))

    def forward(self, x, mask, relevance_logits=None):
        """
        输入:
            x: (B, N, in_dim=512)
            mask: (B, N) bool
            relevance_logits: (B, N, 1) 或 None
        输出:
            bag_embed: (B, num_labels, in_dim)
            attn_weights: (B, num_labels, N)
        """
        # 标准 gated attention 计算 raw logits
        gate = self.attn.v(x) * self.attn.u(x)          # (B, N, attn_dim)
        attn_logits = self.attn.w(gate).transpose(1, 2)  # (B, num_labels, N)

        # relevance 调制
        if relevance_logits is not None:
            relevance = torch.sigmoid(relevance_logits.squeeze(-1))  # (B, N)
            relevance_bias = torch.log(relevance + 1e-6)             # (B, N)
            # alpha: (num_labels,) → (1, num_labels, 1) 广播
            attn_logits = attn_logits + self.alpha.unsqueeze(0).unsqueeze(-1) * relevance_bias.unsqueeze(1)

        attn_weights = masked_softmax(attn_logits, mask.unsqueeze(1), dim=-1)
        bag_embed = torch.einsum("bhn,bnd->bhd", attn_weights, x)
        return bag_embed, attn_weights
```

**关键设计约束**：
- `alpha` 初始化为 0，确保训练初期不干扰正常注意力学习
- 当 `relevance_logits=None` 时，等价于标准 `MultiLabelAttentionMIL`（安全消融）
- `num_labels=8`

---

### 1.4 模块 4：AnatomicalInstanceGrouper + RegionAggregator（创新点2 — 新增模块）

**功能**：根据 region_logits 将 bag 内实例按解剖区域分组，每组独立做 multi-label attention，然后聚合为区域级 bag embedding + 条件向量。

**结构规格**：

```python
class AnatomicalInstanceGrouper(nn.Module):
    """
    根据 region prediction 将实例分组，对每组执行独立的 multi-label attention。
    """
    def __init__(self, in_dim: int = 512, attn_dim: int = 256,
                 num_labels: int = 8, num_regions: int = 6,
                 condition_dim: int = 128, dropout: float = 0.1):
        # 每个区域的独立注意力（共享参数 or 独立参数？建议独立：各区域关注不同的特征模式）
        # 为节省参数，使用共享注意力 + 区域嵌入偏置
        self.region_attn = MultiLabelAttentionMIL(in_dim, attn_dim, num_labels, dropout)

        # 区域嵌入（可学习）
        self.region_embeddings = nn.Embedding(num_regions, in_dim)

        # condition_vector 生成: 各区域的聚合特征 + 区域分布比例 → MLP
        self.condition_mlp = nn.Sequential(
            nn.Linear(num_regions * in_dim + num_regions, condition_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
        )

        # 区域聚合权重（可学习，each region gets a learnable importance weight）
        self.region_weight = nn.Linear(in_dim, 1)

    def forward(self, instance_features, mask, region_logits):
        """
        输入:
            instance_features: (B, N, D=512)
            mask: (B, N) bool
            region_logits: (B, N, R=6)
        输出:
            regional_embed: (B, L=8, D=512)  — 区域聚合后的标签级表示
            condition_vector: (B, condition_dim=128) — 用于条件图推理
        """
        B, N, D = instance_features.shape
        R = region_logits.shape[-1]
        L = self.region_attn.attn.w.out_features  # num_labels

        # soft assignment: 每个 instance 属于各区域的概率
        region_probs = F.softmax(region_logits, dim=-1)     # (B, N, R)

        # 区域分布比例
        region_dist = (region_probs * mask.unsqueeze(-1).float()).sum(dim=1)  # (B, R)
        region_dist = region_dist / (region_dist.sum(dim=-1, keepdim=True) + 1e-8)

        # 对每个区域做 soft-masked attention
        region_embeds = []      # 存储各区域的 (B, L, D)
        region_mean_feats = []  # 存储各区域的平均特征 (B, D)
        for r in range(R):
            # soft region mask: 原始 mask × 区域概率
            region_mask_soft = mask.float() * region_probs[:, :, r]  # (B, N)
            # 阈值化避免全零: 如果某区域概率都低则跳过
            region_valid = region_mask_soft.sum(dim=1) > 0.5  # (B,)

            # 加区域嵌入偏置
            r_emb = self.region_embeddings(
                torch.tensor(r, device=instance_features.device)
            )  # (D,)
            region_features = instance_features + r_emb.unsqueeze(0).unsqueeze(0)  # (B, N, D)

            # 用 soft mask 作为 attention 的额外 mask 权重
            # 这里把 region_mask_soft > threshold 作为 hard mask 传给 attention
            hard_mask = region_mask_soft > 0.1  # (B, N) bool
            # 确保至少有1个 instance
            hard_mask = hard_mask | (~hard_mask.any(dim=1, keepdim=True)).expand_as(hard_mask) & mask

            r_embed, _ = self.region_attn(region_features, hard_mask)  # (B, L, D)
            region_embeds.append(r_embed)

            # 区域平均特征
            weighted_feat = (instance_features * region_mask_soft.unsqueeze(-1)).sum(dim=1)  # (B, D)
            weighted_feat = weighted_feat / (region_mask_soft.sum(dim=1, keepdim=True) + 1e-8)
            region_mean_feats.append(weighted_feat)

        region_embeds = torch.stack(region_embeds, dim=1)      # (B, R, L, D)
        region_mean_feats = torch.stack(region_mean_feats, dim=1)  # (B, R, D)

        # 计算 condition_vector
        condition_input = torch.cat([
            region_mean_feats.reshape(B, -1),  # (B, R*D)
            region_dist,                        # (B, R)
        ], dim=-1)
        condition_vector = self.condition_mlp(condition_input)  # (B, condition_dim)

        # 聚合各区域 embed → 单一 regional_embed
        # 方式: 可学习加权 (attention over regions)
        region_weights = self.region_weight(region_mean_feats).squeeze(-1)  # (B, R)
        region_weights = F.softmax(region_weights, dim=-1)                  # (B, R)
        regional_embed = torch.einsum("br,brld->bld", region_weights, region_embeds)  # (B, L, D)

        return regional_embed, condition_vector
```

**关键设计约束**：
- 使用 **soft grouping**（基于 region_probs 的概率分组），避免 hard argmax 的不可微问题
- `condition_dim=128`，保持轻量
- 区域 attention **共享参数**（只有一个 `region_attn`），通过 `region_embeddings` 区分不同区域的特征偏好，节省参数
- 如果 soft grouping 效果不佳，可替换为 hard argmax 分组 + straight-through estimator

---

### 1.5 模块 5：ConditionalHierarchicalLabelGraphReasoner（创新点2 — 改造模块）

**功能**：在原 `LabelGraphReasoner` 基础上增加 (a) 条件调制、(b) 层次掩码。

**结构规格**：

```python
class ConditionalHierarchicalLabelGraphReasoner(nn.Module):
    def __init__(self, num_labels: int = 8, feature_dim: int = 512,
                 condition_dim: int = 128, dropout: float = 0.2):
        # ---- 与原 LabelGraphReasoner 相同的部分 ----
        self.label_tokens = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- 新增: 条件调制器 ----
        self.graph_modulator = nn.Sequential(
            nn.Linear(condition_dim, num_labels * num_labels),
            nn.Sigmoid(),
        )

        # ---- 新增: 层次掩码 (固定，不学习) ----
        # hierarchy_mask[i][j] = factor ∈ (0, 1]
        # 子标签 i 的传播权重受父标签 j 的 logit 影响
        # 非层次关系的位置 = 1.0（不影响）
        hierarchy_mask = torch.ones(num_labels, num_labels)
        # 具体赋值在 _build_hierarchy_mask() 中完成
        self.register_buffer('hierarchy_parent_map', self._build_parent_map())

        self.num_labels = num_labels
        self.feature_dim = feature_dim

    @staticmethod
    def _build_parent_map():
        """
        返回 (num_labels,) 的 int tensor:
          parent_map[i] = j 表示标签 i 的父标签是 j
          parent_map[i] = -1 表示标签 i 没有父标签

        8标签索引约定:
          0: esophageal_smt
          1: esophageal_neoplasm
          2: gastritis              ← 父标签
          3: gastritis_active       ← 子标签, parent=2
          4: atrophy_moderate_severe← 子标签, parent=2
          5: gastric_polyp
          6: reflux_esophagitis
          7: ulcer
        """
        parent_map = torch.full((8,), -1, dtype=torch.long)
        parent_map[3] = 2   # gastritis_active 的父标签 = gastritis
        parent_map[4] = 2   # atrophy_moderate_severe 的父标签 = gastritis
        return parent_map

    def forward(self, bag_embeds, condition_vector=None, all_logits=None):
        """
        输入:
            bag_embeds: (B, L=8, D=512) — 融合后的标签级表示
            condition_vector: (B, condition_dim=128) 或 None — 条件信号
            all_logits: (B, L=8) 或 None — 分类器的初步logits（用于层次门控）
        输出:
            refined: (B, L, D)
            dynamic_graph: (B, L, L) — 用于可视化和分析
        """
        B = bag_embeds.shape[0]
        L = self.num_labels
        D = self.feature_dim

        # 1. 静态标签图（与原 LabelGraphReasoner 相同）
        norm_tokens = F.normalize(self.label_tokens, dim=-1)
        static_graph = F.softmax(
            norm_tokens @ norm_tokens.T / (D ** 0.5),
            dim=-1,
        )  # (L, L)

        # 2. 条件调制
        if condition_vector is not None:
            modulation = self.graph_modulator(condition_vector)  # (B, L*L)
            modulation = modulation.view(B, L, L)               # (B, L, L)
            dynamic_graph = static_graph.unsqueeze(0) * modulation
        else:
            dynamic_graph = static_graph.unsqueeze(0).expand(B, -1, -1)

        # 3. 层次门控
        if all_logits is not None:
            # parent_map: (L,), 值为父标签索引或 -1
            hierarchy_gate = torch.ones(B, L, L, device=bag_embeds.device)
            for child_idx in range(L):
                parent_idx = self.hierarchy_parent_map[child_idx].item()
                if parent_idx >= 0:
                    # 父标签激活概率
                    parent_prob = torch.sigmoid(all_logits[:, parent_idx])  # (B,)
                    # 衰减子标签的入边权重
                    hierarchy_gate[:, child_idx, :] = parent_prob.unsqueeze(-1)
                    # 也衰减子标签的出边权重
                    hierarchy_gate[:, :, child_idx] = parent_prob.unsqueeze(-1)
            dynamic_graph = dynamic_graph * hierarchy_gate

        # 4. 重归一化
        dynamic_graph = dynamic_graph / (dynamic_graph.sum(dim=-1, keepdim=True) + 1e-8)

        # 5. 图传播 + 残差融合（与原 LabelGraphReasoner 相同）
        propagated = torch.bmm(dynamic_graph, bag_embeds)  # (B, L, D)
        refined = bag_embeds + self.refine(
            torch.cat([bag_embeds, propagated], dim=-1)
        )
        return refined, dynamic_graph
```

**层次门控的直觉**：当 `gastritis` 的 logit 较低（模型认为没有胃炎）时，`gastritis_active` 和 `atrophy_moderate_severe` 在标签图中的传播权重被自动衰减，避免无关标签间的错误信息传播。

**与原 LabelGraphReasoner 的关系**：
- `condition_vector=None` 且 `all_logits=None` 时 → 退化为原始静态 LabelGraphReasoner
- 仅 `condition_vector!=None` → 条件图推理（无层次）
- 两者都有 → 完整的条件层次图推理

---

### 1.6 模块 6：顶层网络 RG-HMIL（新模型类）

**功能**：组装上述所有模块，替代原 `GastroLabelGraphMIL` 作为主模型。

**建议文件位置**：`model/rg_hmil/network.py` + `model/rg_hmil/modules.py`

**结构规格**：

```python
class RGHMIL(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        num_regions: int = 6,
        condition_dim: int = 128,
        dropout: float = 0.2,
        use_text_guidance: bool = True,          # 开关：创新点1
        use_region_grouping: bool = True,        # 开关：创新点2-分组
        use_conditional_graph: bool = True,      # 开关：创新点2-条件调制
        use_hierarchy: bool = True,              # 开关：创新点2-层次约束
    ):
        # 1. InstanceEncoder（复用现有）
        self.instance_encoder = InstanceEncoder(
            backbone_name, pretrained, freeze_stages, feature_dim, dropout
        )

        # 2. InstanceRelevancePredictor（新增）
        self.relevance_predictor = InstanceRelevancePredictor(
            feature_dim, hidden_dim=256, num_regions=num_regions, dropout=0.1
        )

        # 3. 全局注意力 — 有/无 relevance 调制
        if use_text_guidance:
            self.global_attn = RelevanceAwareMultiLabelAttention(
                feature_dim, attn_dim, num_labels, dropout=0.1
            )
        else:
            self.global_attn = MultiLabelAttentionMIL(
                feature_dim, attn_dim, num_labels, dropout=0.1
            )

        # 4. 区域分组聚合器（创新点2）
        self.use_region_grouping = use_region_grouping
        if use_region_grouping:
            self.region_aggregator = AnatomicalInstanceGrouper(
                feature_dim, attn_dim, num_labels, num_regions,
                condition_dim, dropout=0.1
            )

        # 5. 标签图推理
        if use_conditional_graph:
            self.label_graph_reasoner = ConditionalHierarchicalLabelGraphReasoner(
                num_labels, feature_dim, condition_dim, dropout
            )
        else:
            self.label_graph_reasoner = LabelGraphReasoner(
                num_labels, feature_dim, dropout
            )

        # 6. 分类头
        self.classifiers = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(num_labels)
        ])

        # 开关
        self.use_text_guidance = use_text_guidance
        self.use_region_grouping = use_region_grouping
        self.use_conditional_graph = use_conditional_graph
        self.use_hierarchy = use_hierarchy
        self.num_labels = num_labels

    def forward(self, images, mask, pseudo_region_labels=None,
                pseudo_relevance=None):
        """
        输入:
            images: (B, N, 3, H, W)
            mask: (B, N)
            pseudo_region_labels: (B, N) int, 训练时由 dataloader 提供, 推理时为 None
            pseudo_relevance: (B, N) float, 训练时由 dataloader 提供, 推理时为 None
        """
        # 1. 实例特征提取
        features = self.instance_encoder(images)  # (B, N, D)

        # 2. 区域 & 相关性预测
        region_logits, relevance_logits = self.relevance_predictor(features, mask)

        # 3. 全局分支
        if self.use_text_guidance:
            global_embed, global_attn = self.global_attn(features, mask, relevance_logits)
        else:
            global_embed, global_attn = self.global_attn(features, mask)

        # 4. 区域分支
        condition_vector = None
        if self.use_region_grouping:
            regional_embed, condition_vector = self.region_aggregator(
                features, mask, region_logits
            )
            fused_embed = global_embed + regional_embed  # 残差融合
        else:
            fused_embed = global_embed

        # 5. 标签图推理
        # 先做一次初步分类以获取 parent_logits（用于层次门控）
        pre_logits = torch.stack([
            clf(fused_embed[:, i, :]).squeeze(-1)
            for i, clf in enumerate(self.classifiers)
        ], dim=1)  # (B, num_labels)

        if self.use_conditional_graph:
            refined_embed, dynamic_graph = self.label_graph_reasoner(
                fused_embed,
                condition_vector=condition_vector,
                all_logits=pre_logits if self.use_hierarchy else None,
            )
        else:
            refined_embed, dynamic_graph = self.label_graph_reasoner(fused_embed)

        # 6. 最终分类
        logits = torch.stack([
            clf(refined_embed[:, i, :]).squeeze(-1)
            for i, clf in enumerate(self.classifiers)
        ], dim=1)  # (B, num_labels)

        # 7. 辅助损失（仅训练时）
        aux_losses = {}
        if self.training and pseudo_region_labels is not None:
            region_loss = F.cross_entropy(
                region_logits.reshape(-1, region_logits.shape[-1]),
                pseudo_region_labels.reshape(-1),
                ignore_index=-100,
            )
            aux_losses["region_cls"] = region_loss

        if self.training and pseudo_relevance is not None:
            valid = pseudo_relevance >= 0
            if valid.any():
                rel_loss = F.binary_cross_entropy_with_logits(
                    relevance_logits.squeeze(-1)[valid],
                    pseudo_relevance[valid],
                )
                aux_losses["relevance"] = rel_loss

        return {
            "logits": logits,
            "attention": global_attn,
            "instance_features": features,
            "label_graph": dynamic_graph,
            "region_logits": region_logits,
            "relevance_logits": relevance_logits,
            "aux_losses": aux_losses,
        }
```

**消融开关对照表**：

| 开关组合 | 等价模型 | 消融实验名 |
|---|---|---|
| 全 False | ABMIL-8L + Static LGR | Baseline+LGR |
| `use_text_guidance=True` only | + IRS | 创新点1 only |
| `use_region_grouping=True` + `use_conditional_graph=True` | + Conditional LGR | 创新点2(无层次) |
| 全 True | **Full RG-HMIL** | 完整模型 |

---

### 1.7 数据管线扩展：伪标签生成

**文本解析器**（新文件：`preprocessing/text_parser.py`）：

需要实现的核心函数：

```python
def parse_watch_text(watch: str) -> list[dict]:
    """
    解析 watch 字段文本，返回结构化发现列表。

    输入: "食管距门齿20cm可见一0.8cm黏膜下隆起，表面光滑，齿状线清晰..."
    输出: [
        {"region": "esophagus", "lesion": "黏膜下隆起", "positive": True},
        {"region": "cardia_fundus", "lesion": None, "positive": False},
        ...
    ]

    实现策略:
    1. 按句号/分号/逗号分割为句段
    2. 正则匹配解剖部位关键词 → 确定 region
    3. 正则匹配病变关键词 → 确定 lesion type
    4. 正则匹配阴性词（"无异常", "正常", "光滑", "未见"） → 判断 positive/negative
    """

def parse_specimen(specimen: str) -> list[dict]:
    """
    解析 specimen 字段（活检部位）。

    输入: "胃窦 *1\n胃体部 *2"  或  "升结肠 *1"
    输出: [
        {"region": "antrum_angle", "count": 1},
        {"region": "gastric_body", "count": 2},
    ]
    """

def generate_pseudo_labels(
    watch: str,
    specimen: str,
    num_images: int,
) -> dict:
    """
    生成实例级伪标签。

    输出: {
        "region_labels": [int] * num_images,      # 每张图的区域索引 (0-5)，不确定的填 -100
        "relevance_scores": [float] * num_images,  # 每张图的诊断相关性 (0.0-1.0)，不确定的填 -1.0
    }

    分配策略:
    1. 假设图像大致按解剖顺序排列（这是胃镜检查的标准流程）
    2. 将图像序列均匀分为若干段，映射到 watch 中提到的解剖区域顺序
    3. 有病变描述的区域图像 → relevance=0.8
    4. 有 specimen 的区域图像 → relevance=1.0
    5. 明确"无异常"的区域图像 → relevance=0.1
    6. 未提及的区域 → relevance=0.3（中性）
    """
```

**解剖部位关键词映射表**：

```python
REGION_KEYWORDS = {
    "esophagus": ["食管", "食道", "门齿", "齿状线"],
    "cardia_fundus": ["贲门", "胃底"],
    "gastric_body": ["胃体", "大弯", "小弯"],
    "antrum_angle": ["胃窦", "胃角", "幽门"],
    "duodenum": ["十二指肠", "球部", "降部"],
}

POSITIVE_LESION_KEYWORDS = [
    "隆起", "糜烂", "息肉", "萎缩", "充血", "水肿",
    "溃疡", "肿物", "狭窄", "出血", "发红", "褪色",
    "增生", "肿胀", "变薄", "蛇行", "结节",
]

NEGATIVE_KEYWORDS = ["无异常", "正常", "光滑", "未见", "无殊"]
```

**数据管线改动**：
- `MILBagDataset.__getitem__` 增加返回 `pseudo_region_labels` 和 `pseudo_relevance`
- `mil_collate_fn` 增加对这两个字段的 padding
- `build_task_records` 中调用 `generate_pseudo_labels` 预计算并缓存

---

## 二、对比实验模型列表

### 2.1 Baseline 模型（5个）

| # | 模型 | 文献 | 简述 |
|---|---|---|---|
| 1 | **MeanPool-MIL** | Dietterich et al., 1997 | 均值池化聚合，最简单的MIL baseline |
| 2 | **MaxPool-MIL** | — | 最大值池化（hard attention） |
| 3 | **Top-K MIL** | — | 选择注意力最高的 K 个 instance 聚合 |
| 4 | **ABMIL** | Ilse et al., ICML 2018, "Attention-based Deep Multiple Instance Learning" | Gated attention pooling，经典MIL注意力基准 |
| 5 | **Transformer-MIL** | — | Transformer encoder + attention pooling |

> 项目中已有实现：`GastroMeanPoolBaseline`, `GastroMaxPoolBaseline`, `GastroTopKMILBaseline`, `GastroAttentionMILBaseline`, `GastroTransformerMILBaseline`。需更新为 8 标签。

### 2.2 SOTA 模型（5个）

| # | 模型 | 文献 | 简述 |
|---|---|---|---|
| 1 | **CLAM** | Lu et al., *Nature Biomedical Engineering*, 2021, "Data-efficient and weakly supervised computational pathology on whole-slide images" | Instance-level clustering + attention |
| 2 | **DS-MIL** | Li et al., *CVPR*, 2021, "Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning" | 双流 critical instance QKV attention |
| 3 | **TransMIL** | Shao et al., *NeurIPS*, 2021, "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification" | Transformer + PPEG + learnable class tokens |
| 4 | **DTFD-MIL** | Zhang et al., *NeurIPS*, 2022, "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification" | Pseudo-bag grouping + feature distillation |
| 5 | **WIKG-MIL** | Li et al., *CVPR*, 2024, "Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis" | 动态知识图谱构建 + 实例交互图 |

> 项目中已有 CLAM-SB/MB, DS-MIL, TransMIL, DTFD-MIL 的实现（`sotas/gastro_sota/`），需更新为 8 标签。WIKG-MIL 需新增实现。

---

## 三、训练与验证策略

### 3.1 整体训练阶段规划

训练分为**三个阶段**，循序渐进：

```
Phase 0: 标签扩展验证
  └─ 确认 8 标签在所有已有模型上可正常训练

Phase 1: 创新点1 验证（Text-Guided IRS）
  └─ 在 ABMIL-8L 基础上验证文本引导的有效性

Phase 2: 创新点2 验证（Conditional Hierarchical LGR）
  └─ 在创新点1 基础上验证条件层次图推理的有效性

Phase 3: 完整对比实验
  └─ Full model vs 所有 baseline & SOTA
```

---

### 3.2 Phase 0：标签扩展验证

**目标**：确认 8 标签体系可以正常训练，不引入 bug。

| 实验 | 模型 | num_labels | 说明 |
|---|---|---|---|
| 0-A | GastroAttentionMILBaseline | 8 | ABMIL-8L baseline |
| 0-B | GastroLabelGraphMIL | 8 | 现有主模型 8标签版 |

**检查点**：
- 训练可以正常收敛
- 8 个标签的 per-label AUROC/AP 合理
- macro F1 / micro F1 有意义

---

### 3.3 Phase 1：创新点1 消融实验

**目标**：验证 Text-Guided Instance Relevance Scoring 的每个组件的贡献。

| 实验编号 | 模型配置 | 消融说明 |
|---|---|---|
| 1-A | ABMIL-8L (baseline) | 纯注意力，无任何引导 |
| 1-B | ABMIL-8L + IRP辅助头 (只加辅助损失，不调制注意力) | 测试区域预测本身是否有用 |
| 1-C | ABMIL-8L + IRP + Relevance-Aware Attn | **完整创新点1** |
| 1-D | ABMIL-8L + IRP + Relevance-Aware Attn (只用 specimen) | 测试只用活检信号的效果 |

**需要调参的超参数**：

| 超参数 | 搜索范围 | 说明 |
|---|---|---|
| `λ_region` (region_cls 辅助损失权重) | {0.1, 0.3, 0.5, 1.0} | 太大会抢走主任务梯度 |
| `λ_relevance` (relevance 辅助损失权重) | {0.1, 0.3, 0.5, 1.0} | 同上 |
| `num_regions` | {4, 6, 8} | 区域划分粒度 |
| `alpha` 初始化 | {0.0, 0.1} | 调制强度初始值 |

**评价指标**：
- **主要对比**：macro F1, micro F1（1-C vs 1-A 的提升幅度）
- **辅助分析**：per-label AUROC/AP 提升分布（哪些标签受益最多）
- **注意力质量**：可视化 attention heatmap，看创新点1是否让注意力更聚焦在病变区域

---

### 3.4 Phase 2：创新点2 消融实验

**目标**：在 Phase 1 最佳配置基础上，逐步叠加创新点2 的各组件。

| 实验编号 | 在 Phase1 最佳基础上叠加 | 消融说明 |
|---|---|---|
| 2-A | Phase1 最佳 (= 1-C) | 创新点1 only baseline |
| 2-B | + Regional Grouping (区域分组聚合，无条件图) | 测试语义分组本身的效果 |
| 2-C | + Static LGR (静态标签图推理) | 对照：加上原始LGR |
| 2-D | + Conditional LGR (条件调制图，无层次) | 测试条件调制的增量 |
| 2-E | + Conditional Hierarchical LGR (条件+层次) | **完整创新点2** |
| 2-F | **Full RG-HMIL** (= 2-E) | = 最终完整模型 |

**需要调参的超参数**：

| 超参数 | 搜索范围 | 说明 |
|---|---|---|
| `condition_dim` | {64, 128, 256} | 条件向量维度 |
| 区域 attention 参数共享策略 | {共享, 独立} | 独立参数更强但更大 |
| 层次门控阈值 | {无阈值用 sigmoid, 有阈值 0.5} | 层次门控的严格程度 |
| 全局/区域融合方式 | {相加, 拼接+MLP, 可学习加权} | 两个分支怎么合并 |

**评价指标**：
- **主要对比**：macro F1, micro F1（2-E vs 2-A 的提升幅度，以及 vs 2-C 静态LGR 的提升）
- **层次约束指标**：Hierarchical Consistency Rate = $1 - P(\hat{y}_{child}=1 | \hat{y}_{parent}=0)$
- **条件图分析**：可视化不同样本的 dynamic_graph，看是否在不同解剖上下文中呈现不同的标签关联模式
- **子标签条件指标**：`gastritis_active` 和 `atrophy_moderate_severe` 在 `gastritis=1` 子集上的 F1

---

### 3.5 Phase 3：完整对比实验

**目标**：Full RG-HMIL vs 所有 baseline 和 SOTA。

| 类别 | 模型列表 |
|---|---|
| Baseline | MeanPool, MaxPool, TopK, ABMIL, Transformer-MIL |
| SOTA | CLAM-SB, CLAM-MB, DS-MIL, TransMIL, DTFD-MIL, WIKG-MIL |
| Ours (消融) | ABMIL-8L, +StaticLGR, +IRS, +RegionGroup, +CondLGR, +CondHierLGR |
| **Ours (Full)** | **RG-HMIL** |

**实验设置**（统一条件）：
- Backbone: ConvNeXt-Tiny (pretrained)
- 数据划分: 60/20/20, seed=42, 按患者分层
- 评价指标: macro F1, micro F1, per-label AUROC, per-label AP
- 训练配置: lr=1e-4, AdamW, 160 epochs, patience=20, AMP
- 报告 3 次不同 seed 的均值 ± 标准差

---

### 3.6 附加实验

#### 3.6.1 标签体系对比

| 标签体系 | Full Model macro F1 | 说明 |
|---|---|---|
| 3标签（原始） | ？ | 证明 8 标签更有意义 |
| 8标签（新设计） | ？ | 完整标签体系 |

#### 3.6.2 伪标签质量分析

- 随机抽取 50 个 bag，人工检查伪区域标签的准确率
- 分析文本解析器在不同报告类型上的解析成功率
- 如果准确率 < 70% → 只使用 specimen 信号

#### 3.6.3 可视化

- **注意力热力图**：选取典型样本，对比 ABMIL vs RG-HMIL 的注意力分布
- **动态标签图**：展示不同样本（食管主导 vs 胃部主导）的条件图差异
- **区域分组**：可视化实例被分配到各解剖区域的结果，叠加在图像序列上
- **层次约束效果**：对比有/无层次约束时，子标签的 false positive 案例

---

### 3.7 训练细节参考（统一配置）

```yaml
# 统一训练配置
trainer:
  max_epochs: 160
  patience: 20
  lr: 1e-4
  optimizer: adamw
  weight_decay: 0.01
  warmup_ratio: 0.05
  grad_accum_steps: 2
  amp: true
  monitor_metric: macro_f1    # 或 macro_auc
  monitor_mode: max
  loss_name: asymmetric        # ASL
  use_multi_gpu: true

data:
  max_instances: 80            # bag 最大图像数
  min_instances: 5
  bag_sampling_strategy: random
  image_size: 224
  random_instance_dropout: 0.1
  batch_size: 4                # 每卡
  num_workers: 4

# RG-HMIL 特有配置
model:
  backbone_name: convnext_tiny
  feature_dim: 512
  attn_dim: 256
  num_labels: 8
  num_regions: 6
  condition_dim: 128
  dropout: 0.2
  use_text_guidance: true
  use_region_grouping: true
  use_conditional_graph: true
  use_hierarchy: true

aux_loss_weights:
  region_cls: 0.5
  relevance: 0.5
```
