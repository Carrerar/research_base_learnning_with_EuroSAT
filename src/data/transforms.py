"""Augmentation cho EuroSAT (CLAUDE.md mục 5).

Chỉ các phép biến đổi hình học giữ nguyên quan hệ phổ giữa các channel:
random flip H/V và random xoay 90°. **KHÔNG** color jitter (phá spectral relations).
Val/Test không augment (chỉ normalize, đã làm trong dataset).

Transform nhận và trả tensor ``(C, H, W)`` (đã normalize) — khớp chỗ
``EuroSATDataset(transform=...)``.
"""

from __future__ import annotations

import torch


class GeometricAugment:
    """Random horizontal/vertical flip + random 90° rotation (k ∈ {0,1,2,3})."""

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.5, rot90: bool = True):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.rot90 = rot90

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_hflip:
            x = torch.flip(x, dims=[2])  # W
        if torch.rand(1).item() < self.p_vflip:
            x = torch.flip(x, dims=[1])  # H
        if self.rot90:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
        return x


def build_transform(train: bool):
    """Trả về transform: GeometricAugment khi train, None khi val/test."""
    return GeometricAugment() if train else None
