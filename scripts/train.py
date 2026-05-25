"""CLI chạy 1 experiment EuroSAT từ YAML config (1 run / seed).

Nối: config -> data (band/indices/normalize) -> model (conv1 N-ch) -> train
(W&B log mỗi epoch, early stop, best ckpt) -> test (acc/macro-F1/per-class/CM).

Ví dụ:
    python -m scripts.train --config configs/tier1_rgb.yaml
    python -m scripts.train --config configs/tier1_rgb.yaml --seed 42      # chỉ 1 seed
    # smoke test nhanh (CPU, ít mẫu, không đụng W&B, không tải weights):
    python -m scripts.train --config configs/tier1_rgb.yaml --seed 42 \
        --epochs 1 --limit 64 --device cpu --wandb-mode disabled --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader

from src.data.eurosat_dataset import (
    CLASSES,
    EuroSATDataset,
    load_split,
    stats_to_channel_vectors,
)
from src.data.transforms import build_transform
from src.models.resnet import build_resnet50
from src.training.evaluate import evaluate
from src.training.train import train
from src.utils import wandb_utils
from src.utils.config import load_config, resolve_bands
from src.utils.seed import make_generator, seed_worker, set_seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_loader(samples, *, root, bands, indices, norm_mean, norm_std,
                 train, batch_size, num_workers, seed, spatial_aug=None):
    ds = EuroSATDataset(
        root=root, samples=samples, bands=bands, indices=indices,
        norm_mean=norm_mean, norm_std=norm_std,
        transform=build_transform(train, spatial=spatial_aug),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=True, drop_last=False,
        worker_init_fn=seed_worker, generator=make_generator(seed),
    ), ds


def run_one_seed(cfg, seed, args, channel_stats):
    tr = cfg["training"]
    bands = resolve_bands(cfg["bands"])
    indices = list(cfg.get("indices") or [])
    in_channels = len(bands) + len(indices)
    norm_mean, norm_std = stats_to_channel_vectors(channel_stats, bands, indices)

    batch_size = args.batch_size or tr["batch_size"]
    epochs = args.epochs or tr["epochs"]
    num_workers = tr["num_workers"] if args.num_workers is None else args.num_workers

    aug_cfg = cfg.get("augmentation") or {}
    spatial_aug = aug_cfg.get("spatial")
    batch_aug = aug_cfg.get("batch")
    batch_aug_alpha = aug_cfg.get("batch_alpha")
    batch_aug_p = float(aug_cfg.get("batch_p", 1.0))
    aug_desc = "flip+rot"
    if spatial_aug: aug_desc += f"+{spatial_aug}"
    if batch_aug: aug_desc += f"+{batch_aug}(a={batch_aug_alpha},p={batch_aug_p})"

    def split_samples(name):
        s = load_split(f"{args.splits_dir}/{name}.txt")
        return s[: args.limit] if args.limit else s

    common = dict(root=args.data_root, bands=bands, indices=indices,
                  norm_mean=norm_mean, norm_std=norm_std,
                  batch_size=batch_size, num_workers=num_workers, seed=seed)
    train_loader, train_ds = build_loader(split_samples("train"), train=True,
                                          spatial_aug=spatial_aug, **common)
    val_loader, _ = build_loader(split_samples("val"), train=False, **common)
    test_loader, _ = build_loader(split_samples("test"), train=False, **common)

    print(f"[seed {seed}] channels={in_channels} ({train_ds.channel_names}) "
          f"| train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    set_seed(seed)
    pretrained = cfg["model"].get("pretrained", True) and not args.no_pretrained
    model = build_resnet50(in_channels=in_channels, num_classes=len(CLASSES),
                           pretrained=pretrained, bands=bands)

    wb = cfg["wandb"]
    run = wandb_utils.init_run(
        tier=cfg["tier"], exp_id=cfg["exp_id"], config_desc=cfg["config_desc"],
        seed=seed,
        config={
            "model": cfg["model"]["name"], "input_channels": in_channels,
            "input_type": cfg["input_type"], "indices_used": indices,
            "bands": bands, "optimizer": "adamw", "lr": tr["lr"],
            "weight_decay": tr["weight_decay"], "batch_size": batch_size,
            "epochs": epochs, "augmentation": aug_desc,
            "optimizer": tr.get("optimizer", "adamw"),
            "train_split": "80/10/10-stratified-seed42", "pretrained": pretrained,
        },
        question=wb["question"], hypothesis=wb["hypothesis"],
        baseline_ref=wb["baseline_ref"], expected=wb["expected"],
        mode=args.wandb_mode,
    )

    ckpt = f"outputs/{run.name}.pth"
    summary = train(
        model, train_loader, val_loader, device=args.device, run=run,
        epochs=epochs, lr=tr["lr"], weight_decay=tr["weight_decay"],
        label_smoothing=tr["label_smoothing"], max_grad_norm=tr["max_grad_norm"],
        patience=tr["patience"], ckpt_path=ckpt,
        batch_aug=batch_aug, batch_aug_alpha=batch_aug_alpha, batch_aug_p=batch_aug_p,
        optimizer_name=tr.get("optimizer", "adamw"),
        sgd_momentum=tr.get("sgd_momentum", 0.9),
        sgd_nesterov=tr.get("sgd_nesterov", True),
    )

    test = evaluate(model, test_loader, args.device)
    metrics = wandb_utils.log_test_results(
        run, y_true=test["y_true"], y_pred=test["y_pred"], class_names=CLASSES)
    wandb_utils.log_checkpoint(run, ckpt_path=ckpt, metadata={
        "best_val_acc": summary["best_val_acc"], "best_epoch": summary["best_epoch"],
        "test_acc": metrics["acc"], "test_macro_f1": metrics["macro_f1"]})
    print(f"[seed {seed}] DONE best_val_acc={summary['best_val_acc']:.4f} "
          f"test_acc={metrics['acc']:.4f} test_macro_f1={metrics['macro_f1']:.4f}")
    run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="chỉ chạy 1 seed này")
    p.add_argument("--data-root", default="dataset/allbands")
    p.add_argument("--splits-dir", default="dataset/splits")
    p.add_argument("--stats", default="stats/channel_stats.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-mode", default=None,
                   choices=[None, "online", "offline", "disabled"])
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap mẫu/split (smoke test)")
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    with open(args.stats, "r", encoding="utf-8") as f:
        channel_stats = json.load(f)

    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    print(f"Config {args.config} | exp {cfg['exp_id']} | seeds {seeds} | device {args.device}")
    for seed in seeds:
        run_one_seed(cfg, seed, args, channel_stats)


if __name__ == "__main__":
    main()
