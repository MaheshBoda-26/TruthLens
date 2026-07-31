"""
Shared model definition for training and inference.
"""

import os

import torch
import torch.nn as nn
from torchvision import models


def _load_resnet18_backbone():
    """Prefer pretrained weights for fine-tuning, but fall back offline."""
    use_pretrained = os.getenv("USE_PRETRAINED", "1") == "1"
    if not use_pretrained:
        return models.resnet18(weights=None)

    try:
        weights = models.ResNet18_Weights.DEFAULT
        return models.resnet18(weights=weights)
    except Exception as exc:
        print(f"Warning: pretrained weights unavailable, using random init ({exc})")
        return models.resnet18(weights=None)


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = _load_resnet18_backbone()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def create_model(device: torch.device | None = None) -> nn.Module:
    model = DeepfakeDetector()
    if device is not None:
        model = model.to(device)
    return model
