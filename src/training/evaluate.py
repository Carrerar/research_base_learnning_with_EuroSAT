"""Evaluation: chạy model trên 1 loader, trả loss/acc/macro-F1 + nhãn dự đoán."""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    criterion: nn.Module | None = None,
) -> dict:
    """Đánh giá model.

    Trả dict: ``loss`` (None nếu không truyền criterion), ``acc``, ``macro_f1``,
    ``y_true``, ``y_pred`` (list int). Dùng cho cả val (mỗi epoch) và test (cuối).
    """
    model.eval()
    total_loss, n = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return {
        "loss": (total_loss / n) if (criterion is not None and n) else None,
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }
