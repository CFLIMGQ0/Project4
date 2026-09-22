"""纯图像知识迁移：独立学生分类头、外部冻结教师与四组监督。"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as functional

from exp_8.models import Exp13LCCFAblationModel, _soft_binary_distillation
from training.losses import AsymmetricLossMultiLabel


VARIANTS = {
    "A": {"lambda_img": 0.5, "lambda_mm": 0.0, "lambda_kd": 0.0},
    "B": {"lambda_img": 0.5, "lambda_mm": 0.0, "lambda_kd": 1.0},
    "C": {"lambda_img": 0.5, "lambda_mm": 1.0, "lambda_kd": 0.0},
    "D": {"lambda_img": 0.5, "lambda_mm": 1.0, "lambda_kd": 1.0},
}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ImageTransferStudent(Exp13LCCFAblationModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_classifiers = nn.ModuleList(
            [nn.Linear(self.feature_dim, 1) for _ in range(self.num_labels)]
        )

    def configure_supervision(self, variant):
        enabled = bool(VARIANTS[variant]["lambda_mm"])
        for module in (self.text_encoder, self.text_cross_attn, self.text_gate, self.classifiers):
            module.requires_grad_(enabled)
        self.label_query_bias.requires_grad_(enabled)

    def forward(self, images, mask, instance_indices=None, original_image_counts=None,
                labels=None, known=None, watch_token_ids=None, watch_token_mask=None,
                enable_mm=False):
        _, image_embeds, _, _ = self.encode_long_mil(
            images, mask, instance_indices, original_image_counts
        )
        image_logits = torch.stack([
            classifier(image_embeds[:, label_index]).squeeze(-1)
            for label_index, classifier in enumerate(self.image_classifiers)
        ], dim=1)
        output = {"s_img": image_logits}
        if not self.training or not enable_mm:
            return output
        if watch_token_ids is None or watch_token_mask is None:
            raise ValueError("图文监督必须提供配对掩码报告")
        tokens, text_mask, _, active = self.text_encoder(
            watch_token_ids, watch_token_mask, batch_size=images.shape[0], device=images.device
        )
        retrieved, _ = self._lccf_retrieve_text(image_embeds, tokens, text_mask)
        retrieved = retrieved * active[:, None, None].to(retrieved.dtype)
        fused, _ = self._lccf_fuse(image_embeds, retrieved, active)
        output["s_mm_student"] = self.classify(fused)
        observed_labels = (labels, known) if known is not None else labels
        output["l_lqd"] = self._lccf_consistency_loss(
            output["s_mm_student"], image_embeds, retrieved, active, observed_labels
        )
        return output


class FrozenMultimodalTeacher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        return self.model(
            images=batch["images"], mask=batch["mask"],
            instance_indices=batch["instance_indices"],
            original_image_counts=batch["original_image_counts"],
            watch_token_ids=batch["watch_token_ids"], watch_token_mask=batch["watch_token_mask"],
        )["logits"]


def masked_asl(logits, labels, known):
    if not known.any():
        return logits.sum() * 0.0
    return AsymmetricLossMultiLabel()(logits[known], labels[known])


def masked_kd(student_logits, teacher_logits, valid, tau=2.0):
    if student_logits.shape != teacher_logits.shape or valid.shape != student_logits.shape:
        raise ValueError("教师、学生和有效标签mask维度不一致")
    if not valid.any():
        return student_logits.sum() * 0.0
    return _soft_binary_distillation(student_logits[valid], teacher_logits[valid], tau)


def transfer_losses(student, batch, variant, teacher=None, lambda_q=0.01, tau=2.0):
    weights = VARIANTS[variant]
    use_mm, use_kd = bool(weights["lambda_mm"]), bool(weights["lambda_kd"])
    if not student.training:
        raise ValueError("训练损失仅允许在学生train模式调用")
    if use_kd != (teacher is not None):
        raise ValueError("仅B/D必须提供独立冻结教师；A/C禁止教师")
    inputs = {name: batch[name] for name in (
        "images", "mask", "instance_indices", "original_image_counts", "labels", "known"
    )}
    if use_mm:
        inputs.update({name: batch[name] for name in ("watch_token_ids", "watch_token_mask")})
    output = student(**inputs, enable_mm=use_mm)
    losses = {"l_img": masked_asl(output["s_img"], batch["labels"], batch["known"]),
              "l_mm": None, "l_kd": None, "l_lqd": None}
    total = weights["lambda_img"] * losses["l_img"]
    if use_mm:
        losses["l_lqd"] = output["l_lqd"]
        losses["l_mm"] = masked_asl(output["s_mm_student"], batch["labels"], batch["known"]) + lambda_q * output["l_lqd"]
        total = total + weights["lambda_mm"] * losses["l_mm"]
    if use_kd:
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise ValueError("教师参数未冻结")
        valid = batch["known"] & batch["mask"].any(1, keepdim=True) & batch["watch_token_mask"].any(1, keepdim=True)
        losses["l_kd"] = masked_kd(output["s_img"], teacher(batch), valid, tau)
        total = total + weights["lambda_kd"] * losses["l_kd"]
    losses["l_total"] = total
    return output, losses
