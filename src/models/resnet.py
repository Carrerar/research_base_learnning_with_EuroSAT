"""ResNet-50 cho EuroSAT, conv1 sửa cho N-channel input (CLAUDE.md mục 5).

Giữ ImageNet prior trên 3 channel RGB đầu (B04,B03,B02 phải nằm ở 3 vị trí đầu
của input — loader xếp bands theo thứ tự caller truyền, nên caller chịu trách
nhiệm đặt RGB lên đầu khi muốn tận dụng prior). Các channel còn lại khởi tạo
"ấm" = mean của trọng số RGB × 0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def _adapt_conv1(old_conv1: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Tạo conv1 mới cho ``in_channels`` channel, copy prior từ conv1 cũ (3-ch)."""
    new_conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    with torch.no_grad():
        n_copy = min(3, in_channels)
        new_conv1.weight[:, :n_copy] = old_conv1.weight[:, :n_copy]
        if in_channels > 3:
            mean_w = old_conv1.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
            for i in range(3, in_channels):
                new_conv1.weight[:, i : i + 1] = mean_w * 0.5
    return new_conv1


def build_resnet50(
    in_channels: int,
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    """ResNet-50 với conv1 nhận ``in_channels`` và fc -> ``num_classes``.

    ``pretrained=True`` dùng weights ImageNet (IMAGENET1K_V2). Với ``in_channels==3``
    conv1 giữ nguyên trọng số RGB pretrained.
    """
    if in_channels < 1:
        raise ValueError(f"in_channels phải >= 1, nhận {in_channels}")

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if in_channels != 3:
        model.conv1 = _adapt_conv1(model.conv1, in_channels)

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2048 -> num_classes
    return model
