"""2026 年三种论文方法的检查级基础适配；非原论文完整复现。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from exp_4.models import Exp4BaseModel
from sotas.task2.multimodal_sotas import Task2MultimodalSOTABase, _masked_mean


class UnifiedMultimodal2026(Task2MultimodalSOTABase):
    """双编码器、跨模态交互及配对对比目标的分类路径。"""

    def __init__(self, *, hidden_dim: int = 1024, contrast_temperature: float = .07, **kwargs: Any):
        super().__init__(**kwargs)
        self.image_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.text_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.cross_attention = nn.MultiheadAttention(self.feature_dim, self.num_heads,
                                                      dropout=self.dropout_rate, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(3 * self.feature_dim),
                                        nn.Linear(3 * self.feature_dim, hidden_dim), nn.GELU(),
                                        nn.Dropout(self.dropout_rate), nn.Linear(hidden_dim, self.num_labels))
        self.contrast_temperature = contrast_temperature

    def forward(self, images: torch.Tensor, mask: torch.Tensor,
                watch_token_ids: torch.Tensor | None = None,
                watch_token_mask: torch.Tensor | None = None, **_: Any) -> dict[str, Any]:
        image_tokens, _, attention, text_tokens, text_mask, text_pooled, _ = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask)
        image_pooled = _masked_mean(image_tokens, mask)
        image_embed = self.image_projection(image_pooled)
        text_embed = self.text_projection(text_pooled)
        safe_text_mask = text_mask.bool().clone()
        empty = ~safe_text_mask.any(1)
        safe_text_mask[empty, 0] = True
        attended, _ = self.cross_attention(image_embed[:, None], text_tokens, text_tokens,
                                           key_padding_mask=~safe_text_mask, need_weights=False)
        fused = torch.cat((image_embed, text_embed, attended[:, 0]), dim=-1)
        logits = self.classifier(fused)
        if self.training and len(images) > 1:
            scores = F.normalize(image_embed, dim=-1) @ F.normalize(text_embed, dim=-1).T
            scores = scores / self.contrast_temperature
            labels = torch.arange(len(images), device=images.device)
            contrastive = .5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))
        else:
            contrastive = logits.sum() * 0
        return self.build_outputs(logits=logits, attention=attention, image_tokens=image_tokens,
                                  extra={"aux_losses": {"contrastive": contrastive}})


class AdaptiveMultimodalFusion2026(Task2MultimodalSOTABase):
    """图像与报告的门控多模态单元。"""

    def __init__(self, *, hidden_dim: int = 1024, **kwargs: Any):
        super().__init__(**kwargs)
        self.visual = nn.Linear(self.feature_dim, self.feature_dim)
        self.clinical = nn.Linear(self.feature_dim, self.feature_dim)
        self.gate = nn.Linear(2 * self.feature_dim, self.feature_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(self.feature_dim),
                                        nn.Linear(self.feature_dim, hidden_dim), nn.GELU(),
                                        nn.Dropout(self.dropout_rate), nn.Linear(hidden_dim, self.num_labels))

    def forward(self, images: torch.Tensor, mask: torch.Tensor,
                watch_token_ids: torch.Tensor | None = None,
                watch_token_mask: torch.Tensor | None = None, **_: Any) -> dict[str, Any]:
        image_tokens, _, attention, _, _, text_pooled, active = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask)
        image_pooled = _masked_mean(image_tokens, mask)
        visual = torch.tanh(self.visual(image_pooled))
        clinical = torch.tanh(self.clinical(text_pooled))
        gate = torch.sigmoid(self.gate(torch.cat((image_pooled, text_pooled), dim=-1)))
        gate = gate * active[:, None].to(gate.dtype)
        fused = (1 - gate) * visual + gate * clinical
        return self.build_outputs(logits=self.classifier(fused), attention=attention,
                                  image_tokens=image_tokens)


class GCFNet2026(Exp4BaseModel):
    """图像特征、训练集标签共现图与标签条件交互；不使用检查报告。"""

    def __init__(self, *, label_adjacency: list[list[float]] | None = None,
                 hidden_dim: int = 1024, **kwargs: Any):
        for key in ("text_vocab_size", "text_embed_dim", "textcnn_kernel_sizes", "num_heads", "num_layers"):
            kwargs.pop(key, None)
        kwargs["use_label_graph"] = False
        super().__init__(**kwargs)
        matrix = torch.tensor(label_adjacency if label_adjacency is not None else
                              torch.eye(self.num_labels).tolist(), dtype=torch.float32)
        if matrix.shape != (self.num_labels, self.num_labels):
            raise ValueError("标签共现矩阵维度不符")
        matrix = .5 * (matrix + matrix.T)
        matrix.fill_diagonal_(1.)
        degrees = matrix.sum(-1).clamp_min(1e-6).rsqrt()
        self.register_buffer("label_graph", degrees[:, None] * matrix * degrees[None, :])
        self.label_embedding = nn.Parameter(torch.randn(self.num_labels, self.feature_dim) * .02)
        self.graph_layer_1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.graph_layer_2 = nn.Linear(self.feature_dim, self.feature_dim)
        self.image_projection = nn.Sequential(nn.LayerNorm(self.feature_dim),
                                              nn.Linear(self.feature_dim, self.feature_dim))
        self.interaction = nn.Sequential(nn.LayerNorm(2 * self.feature_dim),
                                         nn.Linear(2 * self.feature_dim, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor, **_: Any) -> dict[str, Any]:
        tokens, _ = self.encode_instances(images, mask)
        image = self.image_projection(_masked_mean(tokens, mask))
        nodes = F.relu(self.graph_layer_1(self.label_graph @ self.label_embedding))
        nodes = self.graph_layer_2(self.label_graph @ nodes)
        image_labels = image[:, None, :].expand(-1, self.num_labels, -1)
        logits = self.interaction(torch.cat((image_labels, nodes[None].expand(len(images), -1, -1)),
                                            dim=-1)).squeeze(-1)
        return {"logits": logits, "instance_features": tokens, "aux_losses": {}}


REGISTRY = {
    "unified_multimodal_framework_2026": UnifiedMultimodal2026,
    "adaptive_multimodal_fusion_2026": AdaptiveMultimodalFusion2026,
    "gcf_net_2026": GCFNet2026,
}


def build_paper2026_adapted(name: str, **kwargs: Any) -> nn.Module:
    return REGISTRY[name](**kwargs)
