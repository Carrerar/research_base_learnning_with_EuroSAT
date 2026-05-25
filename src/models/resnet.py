"""ResNet-50 cho EuroSAT, conv1 sửa cho N-channel input (CLAUDE.md mục 5).

Khi input ≠ 3 channels, conv1 mới khởi tạo "ấm" = mean trọng số RGB ImageNet × 0.5
cho mọi channel. Nếu caller truyền ``rgb_positions`` (3 int chỉ vị trí Red/Green/Blue
trong input), conv1 mới sẽ COPY trọng số RGB ImageNet đúng vào các vị trí đó —
nhờ vậy prior align với band thực, không phụ thuộc thứ tự band trong loader.

Lịch sử: trước 2026-05-24 code copy weight[:,:3] vào 3 channel đầu vô điều kiện
=> với ``bands: all`` ([B01,B02,B03,B04,...]) prior R/G/B bị áp lên Aerosol/Blue/Green
(sai), B04 Red không nhận prior. Bug đã fix bằng API rgb_positions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

RGB_BAND_NAMES: tuple[str, str, str] = ("B04", "B03", "B02")  # Red, Green, Blue


def _adapt_conv1(
    old_conv1: nn.Conv2d,
    in_channels: int,
    rgb_positions: Sequence[int] | None = None,
) -> nn.Conv2d:
    """Tạo conv1 mới cho ``in_channels`` channel.

    Default khởi tạo mọi channel = mean(weight_RGB) × 0.5 ("ấm"). Nếu truyền
    ``rgb_positions`` (3 int, vị trí của R/G/B trong input), 3 vị trí đó nhận
    đúng trọng số R/G/B ImageNet thay vì mean.
    """
    new_conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    with torch.no_grad():
        mean_w = old_conv1.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
        for i in range(in_channels):
            new_conv1.weight[:, i : i + 1] = mean_w * 0.5
        if rgb_positions is not None:
            if len(rgb_positions) != 3:
                raise ValueError(
                    f"rgb_positions phải có 3 phần tử (R,G,B), nhận {rgb_positions}"
                )
            for new_pos, old_pos in zip(rgb_positions, (0, 1, 2)):
                if not 0 <= new_pos < in_channels:
                    raise ValueError(
                        f"rgb_positions[{old_pos}]={new_pos} ngoài [0,{in_channels})"
                    )
                new_conv1.weight[:, new_pos : new_pos + 1] = old_conv1.weight[
                    :, old_pos : old_pos + 1
                ]
    return new_conv1


def build_resnet50(
    in_channels: int,
    num_classes: int = 10,
    pretrained: bool = True,
    bands: Sequence[str] | None = None,
) -> nn.Module:
    """ResNet-50 với conv1 nhận ``in_channels`` và fc -> ``num_classes``.

    ``pretrained=True`` dùng weights ImageNet (IMAGENET1K_V2). Với ``in_channels==3``
    conv1 giữ nguyên trọng số RGB pretrained (giả định caller đã đặt R/G/B đúng thứ
    tự). Với ``in_channels!=3``: nếu truyền ``bands`` chứa đủ B04/B03/B02, conv1 mới
    sẽ map RGB ImageNet prior vào đúng vị trí của 3 band đó; còn lại khởi tạo "ấm".
    """
    if in_channels < 1:
        raise ValueError(f"in_channels phải >= 1, nhận {in_channels}")

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if pretrained and in_channels == 3 and bands is not None:
        band_list = list(bands)
        if band_list != list(RGB_BAND_NAMES):
            raise ValueError(
                "pretrained=True + in_channels==3 assumes RGB band order "
                f"{list(RGB_BAND_NAMES)}. Received bands={band_list}. "
                "For non-RGB 3-channel inputs, use in_channels != 3 with explicit "
                "adapter logic or set pretrained=False."
            )

    if in_channels != 3:
        rgb_positions = None
        if pretrained and bands is not None:
            try:
                rgb_positions = [list(bands).index(b) for b in RGB_BAND_NAMES]
            except ValueError:
                rgb_positions = None
        model.conv1 = _adapt_conv1(model.conv1, in_channels, rgb_positions=rgb_positions)

    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2048 -> num_classes
    return model
