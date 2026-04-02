from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .demo_backbones import build_backbone
from .demo_losses import consistency_loss, expert_balance_loss, prototype_pull_push_loss
from .demo_mil_pooling import DemoMultiLabelAttentionMIL, DemoRelationEncoder


class DemoGastroProtoMoEFormer(nn.Module):
    """胃镜 advanced：双分支实例编码 + MoE + prototype bank + 关系建模。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 3,
        num_experts: int = 4,
        proto_per_label: int = 8,
        relation_type: str = "transformer",
        relation_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.num_experts = num_experts

        self.encoder, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )

        self.local_branch = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_branch = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )

        self.routing_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_experts),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, feature_dim),
                )
                for _ in range(num_experts)
            ]
        )

        self.relation_encoder = DemoRelationEncoder(
            d_model=feature_dim,
            nhead=8,
            num_layers=relation_layers,
            dropout=dropout,
            relation_type=relation_type,
        )

        self.mil_pool = DemoMultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.attn_classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

        self.prototypes = nn.Parameter(torch.randn(num_labels, proto_per_label, feature_dim) * 0.02)
        self.proto_scale = nn.Parameter(torch.ones(num_labels))
        self.proto_bias = nn.Parameter(torch.zeros(num_labels))

    def _encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        x = images.reshape(b * n, c, h, w)
        base = self.encoder(x).reshape(b, n, -1)

        z_local = self.local_branch(base)
        z_global = self.global_branch(base)

        gate = self.fusion_gate(torch.cat([z_local, z_global], dim=-1))
        fused = gate * z_local + (1.0 - gate) * z_global
        return fused

    def _route_experts(self, tokens: torch.Tensor, mask: torch.Tensor, subtype_ids: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # bag-level summary for routing
        bag_feat = (tokens * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        soft_w = torch.softmax(self.routing_net(bag_feat), dim=-1)  # [B, E]

        if subtype_ids is not None:
            valid = subtype_ids >= 0
            if valid.any():
                one_hot = F.one_hot(subtype_ids.clamp(min=0), num_classes=self.num_experts).float()
                soft_w = torch.where(valid.unsqueeze(-1), 0.7 * one_hot + 0.3 * soft_w, soft_w)
                soft_w = soft_w / soft_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(tokens))
        stacked = torch.stack(expert_outs, dim=2)  # [B, N, E, D]

        mixed = torch.einsum("bneD,be->bnD", stacked, soft_w)
        return mixed, soft_w

    def _prototype_scores(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        attn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: [B, N, D], attn: [B, L, N]
        return:
          bag_proto_score: [B, L]
          token_proto_score: [B, L, N]
        """
        tok = F.normalize(tokens, dim=-1)
        prot = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bnd,lkd->blnk", tok, prot)  # [B, L, N, K]
        token_proto_score = sim.max(dim=-1).values  # [B, L, N]
        token_proto_score = token_proto_score * mask.unsqueeze(1).float()

        # 用标签 attention 对 prototype evidence 做汇聚
        bag_proto_score = (token_proto_score * attn).sum(dim=-1)  # [B, L]
        return bag_proto_score, token_proto_score

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        subtype_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        tokens = self._encode_instances(images)
        tokens, expert_w = self._route_experts(tokens=tokens, mask=mask, subtype_ids=subtype_ids)
        tokens = self.relation_encoder(tokens, mask)

        bag_embeds, attn = self.mil_pool(tokens, mask)

        attn_logits = []
        for i in range(self.num_labels):
            attn_logits.append(self.attn_classifiers[i](bag_embeds[:, i, :]).squeeze(-1))
        attn_logits = torch.stack(attn_logits, dim=1)  # [B, L]

        bag_proto_score, token_proto_score = self._prototype_scores(tokens=tokens, mask=mask, attn=attn)
        proto_logits = self.proto_scale.unsqueeze(0) * bag_proto_score + self.proto_bias.unsqueeze(0)
        logits = attn_logits + proto_logits

        aux_losses: dict[str, torch.Tensor] = {}
        if labels is not None:
            aux_losses["proto"] = prototype_pull_push_loss(bag_proto_score, labels).reshape(1)
            aux_losses["consistency"] = consistency_loss(torch.sigmoid(attn_logits), torch.sigmoid(proto_logits)).reshape(1)
            aux_losses["expert_balance"] = expert_balance_loss(expert_w).reshape(1)

        return {
            "logits": logits,
            "attention": attn,
            "attn_logits": attn_logits,
            "proto_logits": proto_logits,
            "prototype_scores": bag_proto_score,
            "token_prototype_scores": token_proto_score,
            "expert_weights": expert_w,
            "instance_features": tokens,
            "aux_losses": aux_losses,
        }
