"""W&B helpers — đóng gói convention logging của dự án (CLAUDE.md mục 7).

Mục tiêu: mọi script training gọi cùng một bộ hàm, đảm bảo run nào cũng có
naming / tags / config / notes / metrics đúng chuẩn để reviewer đọc được, và
KHÔNG run nào bị thiếu metadata.

Quy ước (mục 7):
- Project:    eurosat-refine
- Run name:   tier{N}-{exp_id}-{config_desc}-seed{S}
- Tags:       <tier-tag>, <exp_id>, "resnet50"
- Notes:      Question / Hypothesis / Baseline reference / Expected outcome
- Metrics/epoch:  train/loss, train/acc, val/loss, val/acc, val/macro_f1, lr, epoch
- Metrics cuối:   test/acc, test/macro_f1, test/per_class_f1, test/confusion_matrix
- Checkpoint:     wandb.Artifact (best theo val accuracy)
- KHÔNG xoá run thất bại — tag "tier-failed" nếu cần.
"""

from __future__ import annotations

from typing import Sequence

import wandb

PROJECT = "eurosat-refine"
MODEL_TAG = "resnet50"

TIER_TAGS = {
    1: "tier1-baseline",
    2: "tier2-hypothesis",
    3: "tier3-tuning",
}


def make_run_name(tier: int, exp_id: str, config_desc: str, seed: int) -> str:
    """`tier{N}-{exp_id}-{config_desc}-seed{S}` — vd tier2-H3a-13bands+NDVI-seed42."""
    return f"tier{tier}-{exp_id}-{config_desc}-seed{seed}"


def make_notes(question: str, hypothesis: str, baseline_ref: str, expected: str) -> str:
    """Notes theo template mục 7."""
    return (
        f"Question: {question}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Baseline reference: {baseline_ref}\n"
        f"Expected outcome: {expected}"
    )


def init_run(
    *,
    tier: int,
    exp_id: str,
    config_desc: str,
    seed: int,
    config: dict,
    question: str,
    hypothesis: str,
    baseline_ref: str,
    expected: str,
    extra_tags: Sequence[str] | None = None,
    project: str = PROJECT,
    mode: str | None = None,
):
    """Khởi tạo 1 W&B run đúng convention. Trả về ``wandb.Run``.

    ``config`` là dict bắt buộc của mục 7 (model, input_channels, input_type, ...).
    ``mode`` = "online" | "offline" | "disabled" (None = mặc định của wandb/env).
    Hàm tự thêm seed và run_name vào config để khỏi quên.
    """
    if tier not in TIER_TAGS:
        raise ValueError(f"tier phải thuộc {list(TIER_TAGS)}, nhận {tier}")

    name = make_run_name(tier, exp_id, config_desc, seed)
    tags = [TIER_TAGS[tier], exp_id, MODEL_TAG]
    if extra_tags:
        tags.extend(extra_tags)

    full_config = {**config, "seed": seed, "run_name": name}

    return wandb.init(
        project=project,
        name=name,
        tags=tags,
        notes=make_notes(question, hypothesis, baseline_ref, expected),
        config=full_config,
        mode=mode,
    )


def log_epoch(
    run,
    *,
    epoch: int,
    lr: float,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_macro_f1: float,
) -> None:
    """Log metrics 1 epoch (mục 7). ``epoch`` dùng làm step."""
    run.log(
        {
            "epoch": epoch,
            "lr": lr,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/macro_f1": val_macro_f1,
        },
        step=epoch,
    )


def log_test_results(
    run,
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Log metrics cuối training: acc, macro-F1, per-class F1, confusion matrix.

    Trả về dict các con số đã tính (để script in/lưu thêm nếu cần).
    """
    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None,
                         labels=list(range(len(class_names))), zero_division=0)
    per_class_f1 = {name: float(per_class[i]) for i, name in enumerate(class_names)}

    run.log(
        {
            "test/acc": acc,
            "test/macro_f1": macro_f1,
            "test/per_class_f1": per_class_f1,
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=list(y_true),
                preds=list(y_pred),
                class_names=list(class_names),
            ),
        }
    )
    return {"acc": acc, "macro_f1": macro_f1, "per_class_f1": per_class_f1}


def log_checkpoint(
    run,
    *,
    ckpt_path: str,
    name: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Lưu best checkpoint dưới dạng wandb.Artifact (mục 7)."""
    art = wandb.Artifact(
        name=name or f"{run.name}-best",
        type="model",
        metadata=metadata or {},
    )
    art.add_file(ckpt_path)
    run.log_artifact(art)
