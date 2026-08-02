"""
Shared model definitions and factory function for TruthLens training and inference.
Supports baseline ResNet18 as well as EfficientNet-B2 (with optional Frequency Stream).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FrequencyStream(nn.Module):
    """Extracts High-Pass Filter / Laplacian residual features to highlight compression grid artifacts."""

    def __init__(self):
        super().__init__()
        # 3x3 High-pass Laplacian filter kernel
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("hp_kernel", kernel.repeat(3, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.hp_kernel, padding=1, groups=3)


class EfficientNetB2Detector(nn.Module):
    """EfficientNet-B2 backbone with optional DCT high-pass frequency stream integration."""

    def __init__(self, num_classes=1, dropout_rate=0.4, use_freq_stream=True):
        super().__init__()
        self.use_freq_stream = use_freq_stream
        if use_freq_stream:
            self.freq_stream = FrequencyStream()

        try:
            weights = models.EfficientNet_B2_Weights.DEFAULT
            self.backbone = models.efficientnet_b2(weights=weights)
        except Exception as exc:
            print(f"Warning: pretrained EfficientNet weights unavailable, using random init ({exc})")
            self.backbone = models.efficientnet_b2(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if self.use_freq_stream:
            freq_features = self.freq_stream(x)
            x = x + 0.3 * freq_features
        features = self.backbone(x)
        return self.head(features)


class ResNet18Detector(nn.Module):
    """Legacy baseline ResNet18 model definition."""

    def __init__(self):
        super().__init__()
        use_pretrained = os.getenv("USE_PRETRAINED", "1") == "1"
        try:
            weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# Backward compatibility alias
DeepfakeDetector = ResNet18Detector


def create_model(arch: str = "efficientnet_b2", device: torch.device | None = None, **kwargs) -> nn.Module:
    """Factory function for instantiating model architectures."""
    arch_lower = arch.lower()
    if arch_lower in ("efficientnet_b2", "efficientnet_b2_ns", "efficientnet_b2_v10"):
        model = EfficientNetB2Detector(**kwargs)
    elif "resnet18" in arch_lower or "resnet-18" in arch_lower:
        model = ResNet18Detector()
    else:
        # Fallback to EfficientNetB2
        print(f"Warning: Architecture '{arch}' unknown, defaulting to EfficientNetB2 Detector")
        model = EfficientNetB2Detector(**kwargs)

    if device is not None:
        model = model.to(device)
    return model
