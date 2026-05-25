"""Train loop EuroSAT theo recipe cố định (CLAUDE.md mục 5).

AdamW + CosineAnnealingLR, label smoothing 0.1, gradient clipping, mixed precision
(AMP khi có CUDA), early stopping theo val accuracy. Log mỗi epoch qua W&B helper,
lưu best checkpoint theo val acc.

Chỉ chứa logic train; việc dựng data/model/run do ``scripts/train.py`` lo.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate
from src.training.mixaug import apply_batch_aug
from src.utils import wandb_utils


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device | str,
    run=None,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    max_grad_norm: float = 1.0,
    patience: int = 10,
    use_amp: bool | None = None,
    ckpt_path: str = "outputs/best.pth",
    batch_aug: str | None = None,
    batch_aug_alpha: float | None = None,
    batch_aug_p: float = 1.0,
    optimizer_name: str = "adamw",
    sgd_momentum: float = 0.9,
    sgd_nesterov: bool = True,
) -> dict:
    """Train + early stopping. Trả dict: best_val_acc, best_epoch, ckpt_path.

    ``run`` là W&B run (hoặc None để bỏ log). Best checkpoint = val accuracy cao nhất.
    ``batch_aug`` ∈ {None, 'mixup', 'cutmix'}: bật batch-level mixing.
    ``optimizer_name`` ∈ {'adamw', 'sgd'}: SGD dùng momentum + nesterov.
    """
    device = torch.device(device)
    model.to(device)
    if use_amp is None:
        use_amp = device.type == "cuda"

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=sgd_momentum,
            nesterov=sgd_nesterov, weight_decay=weight_decay)
    else:
        raise ValueError(f"optimizer_name không hỗ trợ: {optimizer_name!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            x_in, y_a, y_b, lam = apply_batch_aug(
                batch_aug, x, y, alpha=batch_aug_alpha, p=batch_aug_p)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x_in)
                if lam >= 1.0:
                    loss = criterion(logits, y_a)
                else:
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            # train accuracy tính trên hard label y (gốc), không phải y mixed — để monitor convergence
            running_correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

        scheduler.step()
        train_loss = running_loss / seen
        train_acc = running_correct / seen

        val = evaluate(model, val_loader, device, criterion)
        if run is not None:
            wandb_utils.log_epoch(
                run,
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val["loss"],
                val_acc=val["acc"],
                val_macro_f1=val["macro_f1"],
            )
        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val_loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['macro_f1']:.4f}"
        )

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"model": best_state, "epoch": epoch, "val_acc": best_val_acc},
                ckpt_path,
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping ở epoch {epoch} (patience={patience}).")
                break

    # Khôi phục best weights để caller chạy test.
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "ckpt_path": ckpt_path,
    }
