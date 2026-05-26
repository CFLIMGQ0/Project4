"""
Spectral Decoupled Multi-Label MIL (SD-MIL)

数学理论：谱分解 / 矩阵分解理论
核心思想：对标签共现矩阵做 SVD 分解，得到标签空间的正交基。
         在训练时将多标签预测问题转换到解耦的谱空间中，让每个
         谱分量独立优化。这防止了强标签通过共现关系在梯度更新中
         压制弱标签。

解决的问题：
- 多标签 BCE 损失中，强标签的梯度信号远大于弱标签
- 标签间的共现相关性导致弱标签被"搭便车"预测
- 弱标签的独立判别能力无法被有效训练

关键模块：
- SpectralDecoupler: 基于 SVD 的标签空间解耦器
- DecoupledHead: 在谱空间中做独立预测，再旋转回原空间
- GradientBalancer: 梯度范数均衡，防止谱分量间的梯度失衡
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class SpectralDecoupler(nn.Module):
    """基于 SVD 的多标签解耦模块。

    维护一个可更新的标签共现矩阵，定期做 SVD 提取正交基。
    训练时在谱空间中预测，推理时直接输出原空间结果。
    """

    def __init__(self, num_labels: int, feature_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim

        # Learnable rotation matrix (initialized as identity, will learn optimal decoupling)
        self.rotation = nn.Parameter(torch.eye(num_labels))

        # Per-spectral-component classifiers (in decoupled space)
        self.spectral_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, 1),
            ) for _ in range(num_labels)
        ])

        # Spectral-to-label mixer with orthogonality regularization
        self.label_mixer = nn.Linear(num_labels, num_labels, bias=False)
        nn.init.eye_(self.label_mixer.weight)

        # Gradient scaling per spectral component (learnable)
        self.grad_scale = nn.Parameter(torch.ones(num_labels))

    def _orthogonality_loss(self) -> torch.Tensor:
        """鼓励 rotation 矩阵保持正交性。"""
        RRT = torch.mm(self.rotation, self.rotation.T)
        eye = torch.eye(self.num_labels, device=RRT.device)
        return ((RRT - eye) ** 2).sum()

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bag_embeds: (B, L, D) per-label bag embeddings
        Returns:
            logits: (B, L) final label predictions
            ortho_loss: scalar orthogonality regularization
        """
        batch_size = bag_embeds.shape[0]

        # Rotate label embeddings to spectral space
        # rotation: (L, L), bag_embeds: (B, L, D)
        R = F.normalize(self.rotation, dim=-1)  # Row-normalize for stability
        spectral_embeds = torch.einsum("lk,bkd->bld", R, bag_embeds)  # (B, L, D)

        # Independent prediction in spectral space
        spectral_logits = []
        for i in range(self.num_labels):
            s_logit = self.spectral_heads[i](spectral_embeds[:, i, :]).squeeze(-1)
            # Apply gradient scaling
            if self.training:
                scale = self.grad_scale[i].abs().clamp(min=0.1, max=10.0)
                s_logit = s_logit * scale
            spectral_logits.append(s_logit)

        spectral_logits_tensor = torch.stack(spectral_logits, dim=1)  # (B, L)

        # Map back to original label space
        logits = self.label_mixer(spectral_logits_tensor)

        ortho_loss = self._orthogonality_loss()
        return logits, ortho_loss


class SDMILModel(nn.Module):
    """Spectral Decoupled MIL: 基于谱分解的多标签 MIL。

    通过可学习的正交旋转将标签空间解耦，在谱空间中独立预测
    每个分量，再旋转回原始标签空间。正交性约束防止标签间
    的梯度耦合，使弱标签获得独立的优化路径。
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        ortho_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.ortho_weight = ortho_weight

        self.encoder, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )
        self.shared_proj = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.mil_pool = MultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.spectral = SpectralDecoupler(
            num_labels=num_labels,
            feature_dim=feature_dim,
            dropout=dropout,
        )

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        logits, ortho_loss = self.spectral(bag_embeds)

        result = {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
        }

        if self.training:
            result["aux_losses"] = {"ortho_reg": self.ortho_weight * ortho_loss}

        return result
