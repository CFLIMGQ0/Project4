"""让 ACPE 图像路径在训练期学习抵抗连续缺失观测。"""

from __future__ import annotations

import torch
from training.losses import AsymmetricLossMultiLabel

from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel


def contiguous_deletion_view(images, mask, instance_indices, deletion_fraction: float = 0.5):
    """随机移除一个连续图像块，再将剩余有效图像压到序列前端。"""
    batch, length = mask.shape
    reduced_images = torch.zeros_like(images)
    reduced_mask = torch.zeros_like(mask)
    reduced_indices = None if instance_indices is None else torch.full_like(instance_indices, -1)
    for row in range(batch):
        count = int(mask[row].sum())
        if count < 3:
            chosen = torch.arange(count, device=images.device)
        else:
            removed = min(count - 2, max(1, int(round(count * deletion_fraction))))
            start = int(torch.randint(0, count - removed + 1, (), device=images.device))
            chosen = torch.cat(
                [torch.arange(start, device=images.device),
                 torch.arange(start + removed, count, device=images.device)]
            )
        retained = chosen.numel()
        reduced_images[row, :retained] = images[row, chosen]
        reduced_mask[row, :retained] = True
        if reduced_indices is not None:
            reduced_indices[row, :retained] = instance_indices[row, chosen]
    return reduced_images, reduced_mask, reduced_indices


class BlockAugmentationMixin:
    """在原多模态前向之外，仅给连续删片视图的图像 logits 加监督。"""

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels=None,
        watch_token_ids=None,
        watch_token_mask=None,
        instance_indices=None,
        original_image_counts=None,
        **kwargs,
    ):
        del kwargs
        context, label_embeds, attention, extra = self.encode_long_mil(
            images, mask, instance_indices, original_image_counts
        )
        output = self._build_watch_cross_attention_outputs(
            images=images,
            label_embeds=label_embeds,
            attention=attention,
            features=context,
            extra_outputs=extra,
            labels=labels,
            watch_token_ids=watch_token_ids,
            watch_token_mask=watch_token_mask,
            use_gate=True,
        )
        if self.training and labels is not None:
            dropped_images, dropped_mask, dropped_indices = contiguous_deletion_view(
                images, mask, instance_indices
            )
            _, dropped_embeds, _, _ = self.encode_long_mil(
                dropped_images, dropped_mask, dropped_indices, original_image_counts
            )
            dropped_logits = self.classify(dropped_embeds)
            if isinstance(labels, tuple):
                targets, known = labels
                loss = (
                    AsymmetricLossMultiLabel()(dropped_logits[known], targets[known])
                    if known.any() else dropped_logits.sum() * 0
                )
            else:
                loss = AsymmetricLossMultiLabel()(dropped_logits, labels)
            output["aux_losses"]["block_aug"] = loss
            output["block_aug_retained_fraction"] = (
                dropped_mask.sum(dim=1).float() / mask.sum(dim=1).clamp_min(1)
            )
        return output


class BlockAugmentationModel(BlockAugmentationMixin, Exp12AProCoPEWatchCrossAttentionTextCNNModel):
    """完整标签监督数据集的连续缺失训练候选。"""
