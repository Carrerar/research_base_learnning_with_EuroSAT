"""Tính per-channel mean/std trên TRAIN split, lưu stats/channel_stats.json.

Tính cho CẢ 13 bands GỐC và CẢ 4 spectral indices (NDVI, NDWI, NDBI, NDMI),
mỗi cái lưu theo tên. Loader chọn ra subset cần dùng tuỳ thí nghiệm
(xem ``eurosat_dataset.stats_to_channel_vectors``).

Quan trọng: chỉ dùng TRAIN split để tránh leak (CLAUDE.md mục 8). Một lần quét,
tích luỹ sum/sumsq/count cho 17 channel, rồi suy ra mean/std.

Cách chạy:
    python -m src.data.compute_stats \
        --data-root dataset/allbands --split dataset/splits/train.txt \
        --out stats/channel_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# Windows console (cp1252) không in được tiếng Việt -> ép UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.eurosat_dataset import (
    ALL_BANDS,
    ALL_INDICES,
    compute_indices,
    load_split,
    read_patch,
)


def compute_stats(data_root: str, split_path: str) -> dict:
    """Quét train split, trả về dict {'bands':..., 'indices':..., 'meta':...}."""
    samples = load_split(split_path)
    n_band = len(ALL_BANDS)
    n_idx = len(ALL_INDICES)

    # Tích luỹ theo channel (float64 để ổn định số học).
    band_sum = np.zeros(n_band, np.float64)
    band_sumsq = np.zeros(n_band, np.float64)
    idx_sum = np.zeros(n_idx, np.float64)
    idx_sumsq = np.zeros(n_idx, np.float64)
    count = 0  # số pixel mỗi channel

    for relpath, _ in tqdm(samples, desc="stats", unit="img"):
        full = read_patch(os.path.join(data_root, relpath)).astype(np.float64)  # (13,H,W)
        npx = full.shape[1] * full.shape[2]
        count += npx

        flat = full.reshape(n_band, -1)
        band_sum += flat.sum(1)
        band_sumsq += (flat ** 2).sum(1)

        ind = compute_indices(full, ALL_INDICES).reshape(n_idx, -1)
        idx_sum += ind.sum(1)
        idx_sumsq += (ind ** 2).sum(1)

    def finalize(s, ss):
        mean = s / count
        var = np.maximum(ss / count - mean ** 2, 0.0)
        return mean, np.sqrt(var)

    band_mean, band_std = finalize(band_sum, band_sumsq)
    idx_mean, idx_std = finalize(idx_sum, idx_sumsq)

    return {
        "bands": {
            b: {"mean": float(band_mean[i]), "std": float(band_std[i])}
            for i, b in enumerate(ALL_BANDS)
        },
        "indices": {
            ix: {"mean": float(idx_mean[i]), "std": float(idx_std[i])}
            for i, ix in enumerate(ALL_INDICES)
        },
        "meta": {
            "split": os.path.basename(split_path),
            "n_samples": len(samples),
            "n_pixels_per_channel": int(count),
            "band_order": ALL_BANDS,
            "note": "Stats CHỈ tính trên train split. B8A ở index 12, B11 ở index 10.",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="dataset/allbands")
    p.add_argument("--split", default="dataset/splits/train.txt")
    p.add_argument("--out", default="stats/channel_stats.json")
    args = p.parse_args()

    if not os.path.exists(args.split):
        raise SystemExit(f"Chưa có split {args.split}. Chạy create_splits.py trước.")

    stats = compute_stats(args.data_root, args.split)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Đã lưu {args.out} ({stats['meta']['n_samples']} mẫu train).")
    for b, v in stats["bands"].items():
        print(f"  {b:4s} mean={v['mean']:10.2f} std={v['std']:10.2f}")
    for ix, v in stats["indices"].items():
        print(f"  {ix:4s} mean={v['mean']:10.4f} std={v['std']:10.4f}")


if __name__ == "__main__":
    main()
