# EuroSAT Refine

Dự án Research-Based Learning trên dataset EuroSAT (Sentinel-2): khai thác đầy đủ
13 spectral bands + 4 spectral indices (NDVI, NDWI, NDBI, NDMI) để vượt baseline
RGB của bài báo gốc, đo bằng cả overall accuracy và macro-F1 trên 4 lớp khó.

Toàn bộ context, phương án Refine và cấu trúc thí nghiệm 3 tầng nằm trong
[CLAUDE.md](CLAUDE.md) — đọc file đó để biết chi tiết.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
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

Chi tiết thiết kế: xem [CLAUDE.md](CLAUDE.md).
