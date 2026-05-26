"""
Anchor-Preserved Tail MIL

问题动机：
- 你当前最核心的目标不是“整体换一个模型赌运气”
- 而是保住前三个强标签的性能，同时给后五个弱标签更多局部证据通道

核心想法：
- 一条 anchor 分支继续走稳定的标准注意力 MIL，负责保住强标签
- 一条 tail 分支只看 top-k 局部证据，专门照顾尾标签
- 训练时再加上已有文本解析伪标签带来的区域/相关性辅助监督
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_2.common import Exp2AttentionMILBase, masked_topk_softmax_pool, resolve_head_label_indices


class AnchorMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        head_label_indices: list[int] | tuple[int, ...] | None = None,
        tail_topk: int = 4,
        num_regions: int = 6,
        region_aux_weight: float = 0.2,
        relevance_aux_weight: float = 0.1,
        head_anchor_weight: float = 0.25,
        tail_branch_weight: float = 0.15,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.head_label_indices = resolve_head_label_indices(num_labels, head_label_indices)
        self.tail_topk = int(tail_topk)
        self.region_aux_weight = float(region_aux_weight)
        self.relevance_aux_weight = float(relevance_aux_weight)
        self.head_anchor_weight = float(head_anchor_weight)
        self.tail_branch_weight = float(tail_branch_weight)

        self.tail_scorer = nn.Linear(feature_dim, num_labels)
        self.tail_classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])
        self.region_head = nn.Linear(feature_dim, num_regions)
        self.relevance_head = nn.Linear(feature_dim, 1)

        init_mix = torch.tensor([0.97, 0.97, 0.95, 0.45, 0.4, 0.35, 0.3, 0.3], dtype=torch.float32)
        if init_mix.numel() < num_labels:
            init_mix = torch.cat([init_mix, init_mix.new_full((num_labels - init_mix.numel(),), 0.35)], dim=0)
        self.mix_logits = nn.Parameter(torch.logit(init_mix[:num_labels].clamp(1e-4, 1.0 - 1e-4)))

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        pseudo_region_labels: torch.Tensor | None = None,
        pseudo_relevance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        anchor_bag, anchor_attention = self.mil_pool(features, mask)

        tail_scores = self.tail_scorer(features).transpose(1, 2)
        tail_bag, tail_attention = masked_topk_softmax_pool(features, tail_scores, mask, self.tail_topk)

        anchor_logits = self.classify_embeddings(anchor_bag)
        tail_logits = torch.stack(
            [head(tail_bag[:, label_index, :]).squeeze(-1) for label_index, head in enumerate(self.tail_classifiers)],
            dim=1,
        )

        mix = torch.sigmoid(self.mix_logits).view(1, self.num_labels)
        logits = mix * anchor_logits + (1.0 - mix) * tail_logits
        attention = mix.unsqueeze(-1) * anchor_attention + (1.0 - mix).unsqueeze(-1) * tail_attention

        aux_losses: dict[str, torch.Tensor] = {}
        if self.training and labels is not None:
            if self.head_label_indices:
                head_index = torch.tensor(self.head_label_indices, device=logits.device)
                aux_losses["anchor_head"] = (
                    F.binary_cross_entropy_with_logits(anchor_logits[:, head_index], labels[:, head_index])
                    * self.head_anchor_weight
                )

            tail_indices = [index for index in range(self.num_labels) if index not in set(self.head_label_indices)]
            if tail_indices:
                tail_index = torch.tensor(tail_indices, device=logits.device)
                aux_losses["tail_branch"] = (
                    F.binary_cross_entropy_with_logits(tail_logits[:, tail_index], labels[:, tail_index])
                    * self.tail_branch_weight
                )

        if self.training and pseudo_region_labels is not None:
            valid = pseudo_region_labels >= 0
            if valid.any():
                region_logits = self.region_head(features)
                aux_losses["region_aux"] = (
                    F.cross_entropy(region_logits[valid], pseudo_region_labels[valid].long()) * self.region_aux_weight
                )

        if self.training and pseudo_relevance is not None:
            valid = pseudo_relevance >= 0
            if valid.any():
                relevance_logits = self.relevance_head(features).squeeze(-1)
                aux_losses["relevance_aux"] = (
                    F.binary_cross_entropy_with_logits(relevance_logits[valid], pseudo_relevance[valid].float())
                    * self.relevance_aux_weight
                )

        outputs = {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
            "anchor_logits": anchor_logits,
            "tail_logits": tail_logits,
            "anchor_attention": anchor_attention,
            "tail_attention": tail_attention,
        }
        if aux_losses:
            outputs["aux_losses"] = aux_losses
        return outputs


__all__ = ["AnchorMILModel"]
