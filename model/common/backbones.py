from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _configure_pretrained_weights_dir() -> Path:
    """将 torchvision 预训练权重统一缓存到项目根目录。"""
    weights_dir = Path(__file__).resolve().parents[3] / "pre_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(weights_dir)
    torch.hub.set_dir(str(weights_dir))
    return weights_dir


_configure_pretrained_weights_dir()


class _ResNetFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class _ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.norm = model.classifier[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)


class _EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def _safe_load_model(builder, weights, fallback_builder):
    try:
        return builder(weights=weights)
    except Exception:
        return fallback_builder(weights=None)


def _freeze_modules(modules: list[nn.Module], freeze_stages: int) -> None:
    if freeze_stages <= 0:
        return
    for module in modules[:freeze_stages]:
        for parameter in module.parameters():
            parameter.requires_grad = False


def build_backbone(
    backbone_name: str,
    pretrained: bool = True,
    out_dim: int = 512,
    freeze_stages: int = 0,
    projector_dropout: float = 0.1,
) -> tuple[nn.Module, int]:
    """构建图像编码器并输出统一维度特征。"""
    name = backbone_name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = _safe_load_model(models.resnet50, weights, models.resnet50)
        feature_extractor = _ResNetFeatureExtractor(model)
        _freeze_modules(
            [
                feature_extractor.stem,
                feature_extractor.layer1,
                feature_extractor.layer2,
                feature_extractor.layer3,
                feature_extractor.layer4,
            ],
            freeze_stages,
        )
        in_dim = 2048
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load_model(models.convnext_tiny, weights, models.convnext_tiny)
        feature_extractor = _ConvNeXtFeatureExtractor(model)
        _freeze_modules(list(feature_extractor.features.children()), freeze_stages)
        in_dim = 768
    elif name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = _safe_load_model(models.efficientnet_b3, weights, models.efficientnet_b3)
        feature_extractor = _EfficientNetFeatureExtractor(model)
        _freeze_modules(list(feature_extractor.features.children()), freeze_stages)
        in_dim = 1536
    else:
        raise ValueError(f"不支持的 backbone: {backbone_name}")

    encoder = nn.Sequential(
        feature_extractor,
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(projector_dropout),
    )
    return encoder, out_dim
