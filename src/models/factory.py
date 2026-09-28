"""CNN backbones with transfer learning (timm).

All models are ImageNet-pretrained CNNs from the timm library. timm replaces
the original 1000-class ImageNet classifier with a new, randomly
initialised 5-class head (with dropout before it).

Transfer learning is done in two stages (see src/training/train.py):
1. Backbone frozen, only the new head is trained. The pretrained features
   are kept intact while the random head settles.
2. Everything unfrozen and fine-tuned with a lower learning rate, so the
   features adapt to retinal images without being destroyed.
"""

from __future__ import annotations

import timm
import torch.nn as nn

# Short names used in the config / CLI -> timm model names.
BACKBONES = {
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
    "mobilenetv3": "mobilenetv3_large_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
}


def build_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create a pretrained CNN with a new ``num_classes`` output head.

    Args:
        backbone: Key of :data:`BACKBONES`.
        num_classes: Number of output classes (5 DR stages).
        pretrained: Load ImageNet weights (downloaded on first use).
        dropout: Dropout rate before the classification layer.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONES)}")
    return timm.create_model(
        BACKBONES[backbone], pretrained=pretrained, num_classes=num_classes, drop_rate=dropout
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every layer except the classification head."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    model.backbone_frozen = not trainable


def keep_frozen_batchnorm_fixed(model: nn.Module) -> None:
    """Put BatchNorm layers in eval mode while the backbone is frozen.

    Freezing the weights is not enough: in training mode BatchNorm layers
    still update their running mean/variance from the new data, which
    changes the pretrained features. Call this after ``model.train()``.
    """
    if getattr(model, "backbone_frozen", False):
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
