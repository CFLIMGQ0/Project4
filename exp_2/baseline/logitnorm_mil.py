"""
LogitNorm MIL

参考来源：
- Wei et al., ICML 2022
- Mitigating Neural Network Overconfidence with Logit Normalization

项目改写说明：
- 原方法用于抑制过置信输出
- 这里主要借它来压住尾标签“概率抬得太高”的现象，减少假阳性
"""

from __future__ import annotations

import torch

from exp_2.common import Exp2AttentionMILBase


class LogitNormMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        norm_temperature: float = 1.5,
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
        self.norm_temperature = float(norm_temperature)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(images, mask)
        raw_logits = outputs["logits"]
        norm = raw_logits.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        norm_logits = self.norm_temperature * raw_logits / norm
        outputs["raw_logits"] = raw_logits
        outputs["logits"] = norm_logits
        return outputs

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del current_epoch, train_mode
        loss = criterion(outputs["logits"], labels)
        return loss.mean() if isinstance(loss, torch.Tensor) and loss.ndim > 0 else loss


__all__ = ["LogitNormMILModel"]
