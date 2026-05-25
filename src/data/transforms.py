"""Augmentation cho EuroSAT (CLAUDE.md mục 5 + Tier 3 T3-C aug sweep).

Sample-level transforms áp dụng trong ``EuroSATDataset(transform=...)``:
- ``GeometricAugment``: random flip H/V + random 90° rotation. KHÔNG color jitter
  (phá spectral relations). Mặc định cho mọi train run.
- ``RandomResizedCrop``: crop ngẫu nhiên scale ∈ [0.7,1.0] ratio ∈ [3/4,4/3] rồi
  resize về 64×64. Tier 3 T3-C variant.

Batch-level mixing (MixUp/CutMix) áp dụng trong training loop, không phải transform.
Xem ``src/training/mixaug.py``.

Transform nhận và trả tensor ``(C, H, W)`` (đã normalize) — khớp chỗ
``EuroSATDataset(transform=...)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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


class RandomResizedCrop:
    """Crop ngẫu nhiên (scale, ratio) rồi bilinear resize về ``size``×``size``."""

    def __init__(
        self,
        size: int = 64,
        scale: tuple[float, float] = (0.7, 1.0),
        ratio: tuple[float, float] = (3 / 4, 4 / 3),
        p: float = 1.0,
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        _, h, w = x.shape
        area = h * w
        log_r_lo = math.log(self.ratio[0])
        log_r_hi = math.log(self.ratio[1])
        for _ in range(10):
            target_area = area * (self.scale[0] + (self.scale[1] - self.scale[0]) * torch.rand(1).item())
            aspect = math.exp(log_r_lo + (log_r_hi - log_r_lo) * torch.rand(1).item())
            ch = int(round(math.sqrt(target_area / aspect)))
            cw = int(round(math.sqrt(target_area * aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                i = int(torch.randint(0, h - ch + 1, (1,)).item())
                j = int(torch.randint(0, w - cw + 1, (1,)).item())
                crop = x[:, i : i + ch, j : j + cw]
                return F.interpolate(
                    crop.unsqueeze(0), size=(self.size, self.size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
        # Fallback: center crop to min(h, w) then resize
        side = min(h, w)
        i = (h - side) // 2
        j = (w - side) // 2
        crop = x[:, i : i + side, j : j + side]
        return F.interpolate(
            crop.unsqueeze(0), size=(self.size, self.size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)


class Compose:
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_transform(train: bool, spatial: str | None = None):
    """Transform cho ``EuroSATDataset``.

    ``train=True``: GeometricAugment + optional spatial aug. ``spatial='rrc'`` thêm
    ``RandomResizedCrop(size=64)``. None hoặc 'none' = chỉ geometric.
    ``train=False``: None (val/test không augment).
    """
    if not train:
        return None
    ts = [GeometricAugment()]
    if spatial in ("rrc", "random_resized_crop"):
        ts.append(RandomResizedCrop(size=64))
    elif spatial in (None, "none", ""):
        pass
    else:
        raise ValueError(f"spatial aug không hỗ trợ: {spatial!r}")
    return Compose(ts)
