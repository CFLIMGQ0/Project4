"""
EQLv2 MIL

参考来源：
- Tan et al., CVPR 2021
- Equalization Loss v2: A New Gradient Balance Approach for Long-Tailed Object Detection

项目改写说明：
- 原论文用于检测，这里把“正负梯度失衡”思想改写成多标签 BCE 的逐标签负梯度抑制
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exp_2.common import Exp2AttentionMILBase


class EQLV2MILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        eql_gamma: float = 0.7,
        eql_momentum: float = 0.9,
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
        self.eql_gamma = float(eql_gamma)
        self.eql_momentum = float(eql_momentum)
        self.register_buffer("pos_grad", torch.ones(num_labels, dtype=torch.float32))
        self.register_buffer("neg_grad", torch.ones(num_labels, dtype=torch.float32))

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del criterion, current_epoch
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)

        if train_mode:
            with torch.no_grad():
                batch_pos_grad = (labels * (1.0 - probs)).sum(dim=0)
                batch_neg_grad = ((1.0 - labels) * probs).sum(dim=0)
                self.pos_grad.mul_(self.eql_momentum).add_(batch_pos_grad * (1.0 - self.eql_momentum))
                self.neg_grad.mul_(self.eql_momentum).add_(batch_neg_grad * (1.0 - self.eql_momentum))

        grad_ratio = (self.pos_grad + 1e-6) / (self.neg_grad + 1e-6)
        neg_weight = (grad_ratio / self.eql_gamma).clamp(min=0.0, max=1.0).unsqueeze(0)

        pos_loss = -labels * torch.log(probs.clamp(min=1e-6))
        neg_loss = -(1.0 - labels) * neg_weight * torch.log((1.0 - probs).clamp(min=1e-6))
        return (pos_loss + neg_loss).mean()


__all__ = ["EQLV2MILModel"]
