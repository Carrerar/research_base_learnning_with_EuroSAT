"""MixUp / CutMix batch-level augmentation cho Tier 3 T3-C sweep.

MixUp (Zhang et al. 2017): convex combination ảnh + labels theo λ~Beta(α,α).
CutMix (Yun et al. 2019): cắt patch ngẫu nhiên từ ảnh khác dán vào, label mix theo
diện tích patch.

Cả hai trả về ``(x_mixed, y_a, y_b, lam)``. Caller tính loss:
    lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

Khi α <= 0 hoặc với xác suất (1-p) → no-op (lam=1, y_b=y_a).
"""

from __future__ import annotations

import math

import torch


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Convex combination 2 batch theo λ~Beta(α,α).

    ``x`` shape (N,C,H,W), ``y`` shape (N,) int64. Trả (x_mix, y_a, y_b, lam).
    """
    if alpha <= 0.0 or torch.rand(1).item() >= p:
        return x, y, y, 1.0
    lam = float(_sample_beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    return x_mixed, y, y[perm], lam


def cutmix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Cắt patch ngẫu nhiên từ x[perm] dán vào x. ``lam`` = 1 − area(patch)/area(image).

    ``x`` shape (N,C,H,W), ``y`` shape (N,) int64.
    """
    if alpha <= 0.0 or torch.rand(1).item() >= p:
        return x, y, y, 1.0
    lam_init = float(_sample_beta(alpha, alpha))
    _, _, H, W = x.shape
    cut_rat = math.sqrt(1.0 - lam_init)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    if cut_w == 0 or cut_h == 0:
        return x, y, y, 1.0
    cx = int(torch.randint(0, W, (1,)).item())
    cy = int(torch.randint(0, H, (1,)).item())
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    perm = torch.randperm(x.size(0), device=x.device)
    x_out = x.clone()
    x_out[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(H * W))
    return x_out, y, y[perm], lam


def _sample_beta(a: float, b: float) -> float:
    """Sample từ Beta(a,b) bằng torch (không phụ thuộc numpy)."""
    g1 = torch._standard_gamma(torch.tensor([a])).item()
    g2 = torch._standard_gamma(torch.tensor([b])).item()
    return g1 / (g1 + g2 + 1e-12)


def apply_batch_aug(
    name: str | None,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float | None = None,
    p: float = 1.0,
):
    """Dispatcher gọi mixup/cutmix theo tên. ``name`` ∈ {None,'none','mixup','cutmix'}."""
    if name in (None, "none", ""):
        return x, y, y, 1.0
    if name == "mixup":
        return mixup_batch(x, y, alpha=alpha if alpha is not None else 0.2, p=p)
    if name == "cutmix":
        return cutmix_batch(x, y, alpha=alpha if alpha is not None else 1.0, p=p)
    raise ValueError(f"batch aug không hỗ trợ: {name!r}")
