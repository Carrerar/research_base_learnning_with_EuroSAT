# EuroSAT Refine → Research

Dự án Research-Based Learning trên dataset EuroSAT (Sentinel-2), đi theo chu trình:

```text
Read → Reproduce → Refine → Research → Report
```

## Trạng thái hiện tại

- **Reproduce:** DONE — RGB baseline tái lập gần paper gốc.
- **Refine:** DONE — kiểm chứng 13 spectral bands, spectral indices, augmentation/optimizer tuning.
- **Research:** chuẩn bị RQ1-A — low-label self-supervised learning.

## Kết luận Refine

Refine ban đầu kiểm chứng giả thuyết:

> 13 bands + spectral indices có thể vượt RGB baseline và cải thiện nhóm lớp khó.

Sau bug fix conv1 RGB-prior alignment, kết luận chính là:

- Winner cuối: **T1-R2 fixed = 13-band raw + AdamW default recipe**.
- Spectral indices **không cải thiện thêm** khi đã có 13 raw bands + conv1 alignment đúng.
- EuroSAT supervised 80/10/10 gần saturation; research value còn lại nằm ở:
  - low-label regime,
  - representation learning,
  - SSL.

## Research protocol

Research phase giữ nguyên global split:

```text
Train / Val / Test = 21600 / 2700 / 2700
```

Không chuyển sang literal 10/90 split. Chỉ subsample từ TRAIN để tạo label fractions 1% / 5% / 10% / 100%.

Toàn bộ context, decisions log, protocol freeze và guard rails nằm trong [CLAUDE.md](CLAUDE.md) và [updated.md](updated.md).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```text
dataset/        rgb/ (visualization), allbands/ (training), splits/ (train/val/test)
src/data/       dataset loader, preprocessing, spectral indices
src/models/     model setup, conv1 modification cho N-channel input
src/training/   train loop, evaluation
src/utils/      seed, logging, W&B helpers
configs/        YAML config cho mỗi run
scripts/        CLI để chạy thí nghiệm
notebooks/      EDA, visualization
outputs/        checkpoints, logs (gitignored)
stats/          channel_stats.json
```

## Research low-label splits

Tạo nested train subsets:

```bash
python -m scripts.create_label_fraction_splits --train-split dataset/splits/train.txt --out-dir dataset/splits --seed 42
```

Sau đó chạy Condition B baseline curve, ví dụ:

```bash
python -m scripts.train --config configs/research/rq1_B_1pct.yaml --seed 42
```
