# CLAUDE.md — EuroSAT Refine Project

> File này là memory của dự án. Claude Code đọc tự động khi bắt đầu phiên làm việc.
> Cập nhật file này khi có thay đổi quan trọng về thiết kế hoặc khi hoàn thành tầng.

---

## 1. Bối cảnh dự án

Đây là dự án **Research-Based Learning** dựa trên bài báo:

> Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. IEEE JSTARS. arXiv:1709.00029.

Dự án chia thành 5 giai đoạn theo chu trình **Read → Reproduce → Refine → Research → Report** (8 tuần). File này tập trung vào giai đoạn **Refine** (tuần 4-5) và chuẩn bị cho **Research** (tuần 6-7).

**Yêu cầu nộp bài:** log W&B đầy đủ + paper kèm theo ghi nhận toàn bộ quá trình training. Vì vậy mọi run cần trung thực, có giả thuyết rõ, và không xoá run thất bại.

---

## 2. Bài báo gốc — tóm tắt phân tích

### Đóng góp chính
- Dataset 27,000 ảnh Sentinel-2, 10 lớp LULC, 13 spectral bands, resolution 10m.
- Benchmark CNN (ResNet-50, GoogLeNet) đạt 98.57% accuracy trên RGB split 80/20.
- So sánh 3 band combinations: RGB (98.57%), CI/Color Infrared (98.30%), SWIR (97.05%).

### Khe hở (gap) chính của bài
| Khe hở | Mức độ khai thác được |
|---|---|
| Không có model "13-band native" — chỉ test 3-band combinations | **Cao — chính là target của Refine** |
| Không dùng spectral indices (NDVI, NDWI, NDBI, NDMI) — domain knowledge bỏ phí | **Cao — chính là target của Refine** |
| Confusion matrix yếu trên 4 lớp đồng cỏ/nông nghiệp (Annual Crop, Permanent Crop, Pasture, Herbaceous Vegetation) | Trung bình — là metric phụ |
| Low-data regime: chỉ 75% accuracy ở 10% train data | Cao — dành cho Research (SSL) |
| Không có temporal modeling | Dành cho Research |
| Bias địa lý (chỉ châu Âu) | Dành cho Research (cross-domain) |
| Không có statistical significance (single number, no std) | Phải fix trong reproduce |

### 10 lớp và nhóm độ khó
- **Dễ:** Forest, Sea/Lake, Residential, Industrial, Highway.
- **Trung bình:** River.
- **Khó (dễ nhầm lẫn nhau):** Annual Crop, Permanent Crop, Pasture, Herbaceous Vegetation.

### 13 bands Sentinel-2

| Loại | Resolution | Bands | Mục đích |
|---|---|---|---|
| Visible + NIR | 10m | B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) | Chi tiết bề mặt |
| Red-edge + SWIR | 20m | B05, B06, B07, B8A, B11, B12 | Thực vật, độ ẩm |
| Atmospheric | 60m | B01, B09, B10 | Hiệu chỉnh khí quyển (có thể là noise) |

---

## 3. Phương án Refine đã chốt

**Tiêu đề:** Multi-spectral fusion với 13 bands + 4 spectral indices, đo bằng cả accuracy tổng thể *và* macro-F1 trên 4 lớp khó.

**Lý do chọn phương án này:**
1. Gắn chặt với khe hở chính của bài gốc (không khai thác đầy đủ 13 bands).
2. Có baseline trực tiếp (RGB 98.57%) để so sánh.
3. Có hai metric độc lập (overall acc + macro-F1 lớp khó) — giảm rủi ro không cải thiện được.
4. Áp dụng được domain knowledge từ remote sensing (spectral indices).

**Mục tiêu:**
- Must-have: 13-band + indices ≥ RGB baseline về accuracy, AND macro-F1 lớp khó tăng ≥ 1.5%.
- Nice-to-have: ≥ 99.0% overall accuracy, AND macro-F1 lớp khó tăng ≥ 2.0%.
- Cả kết quả âm (multi-spectral không vượt RGB) cũng là phát hiện đáng báo cáo.

---

## 4. 4 Spectral indices được dùng

```
NDVI = (B08 - B04) / (B08 + B04 + 1e-8)   # Vegetation — nhắm Annual Crop, Pasture, Herbaceous
NDWI = (B03 - B08) / (B03 + B08 + 1e-8)   # Water — nhắm River, Sea/Lake
NDBI = (B11 - B08) / (B11 + B08 + 1e-8)   # Built-up — nhắm Industrial, Residential
NDMI = (B08 - B11) / (B08 + B11 + 1e-8)   # Moisture — nhắm Forest, Herbaceous Vegetation
```

**Cảnh báo:** kiểm tra mapping index của band trong loader cụ thể (torchgeo / kaggle / dfki) trước khi tính — sai index sẽ làm hỏng toàn bộ thí nghiệm mà không hiện ra rõ.

### Band order trong file TIFF (verified 2026-05-23)

File TIFF của `dataset/allbands/` lưu theo thứ tự numeric L1C với **B8A ở CUỐI** (idx 12) — **khác** thứ tự canonical của torchgeo (B8A nằm giữa, sau B08):

| idx  | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10      | 11  | 12      |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|---------|-----|---------|
| band | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 | B10 | **B11** | B12 | **B8A** |

Điểm dễ nhầm:

- **B11 ở idx 10** (không phải 11) — sai 1 đơn vị sẽ làm NDBI/NDMI tính trên B12 thay vì B11, không báo lỗi.
- **B8A ở idx 12** (cuối) — nếu cho rằng B8A đứng sau B08 (canonical) thì lấy nhầm B09.

Cách verify (đã thực hiện):

1. Per-band profile theo lớp: SeaLake có NDWI(B03,B08) trung bình +0.37, Forest có NDVI(B08,B04) +0.73, Residential có NDBI cao nhất; B10 (cirrus) ≈ 12 (gần 0, đúng đặc tính khí quyển 60m).
2. Notebook `notebooks/verify_allbands_loader.ipynb` — load 1 file, kiểm tra mapping + NDVI khớp với `compute_indices()` (diff = 0.0).
3. Tier-1 RGB baseline đạt 98.52% — khớp paper 98.57% → mapping của loader đúng.

**Hệ quả thực hành:** loader (`src/data/eurosat_dataset.py`) truy cập **theo TÊN band qua `BAND_TO_IDX`**, không index thô — code dùng loader này an toàn. Mọi code mới đọc raw TIFF phải tham chiếu bảng trên.

---

## 5. Setup chung

### Dataset
- Dùng **EuroSAT all-bands** (float32, 13 channels), KHÔNG dùng RGB-JPEG version.
- Split: **80/10/10 stratified** (21600 / 2700 / 2700), cố định bằng seed 42.
- Lưu split thành files cố định: `splits/train.txt`, `val.txt`, `test.txt`.
- **Protocol freeze 2026-05-26 (updated.md):** split này dùng XUYÊN SUỐT Refine → Research → Report. KHÔNG regenerate global splits. KHÔNG chuyển sang 10/90 như paper gốc. Research phase chỉ subsample TRAIN — val + test bất biến.

### Cấu trúc dataset trên disk
- `dataset/rgb/` — phiên bản RGB JPEG gốc từ EuroSAT.zip (3 channels, uint8).
  Dùng để: visualization, demo, không dùng cho training chính.
- `dataset/allbands/` — phiên bản 13 bands TIFF từ EuroSATallBands.zip (float32).
  Dùng để: TẤT CẢ các thí nghiệm training. RGB cho Tầng 1 trích từ B04/B03/B02 ở đây.
- `dataset/splits/` — file train.txt, val.txt, test.txt cố định seed 42.

### Preprocessing
1. Upsample 20m và 60m bands về 10m bằng bilinear → tensor (13, 64, 64).
2. Tính 4 indices → tensor (17, 64, 64).
3. Normalize per-channel với stats tính trên train set, lưu `stats/channel_stats.json`.

### Augmentation
- Train: random flip H/V, random 90° rotation. KHÔNG color jitter (phá spectral relations).
- Val/Test: chỉ normalize.

### Model
ResNet-50 pretrained ImageNet, điều chỉnh conv1 cho N-channel input. **API hiện tại (sau fix 2026-05-24):** `build_resnet50(in_channels, num_classes, pretrained, bands=...)` tự tính `rgb_positions` từ `bands` list và align ImageNet R/G/B prior vào ĐÚNG vị trí của B04/B03/B02 trong input.

```python
def _adapt_conv1(old_conv1, in_channels, rgb_positions=None):
    new_conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    with torch.no_grad():
        mean_w = old_conv1.weight.mean(dim=1, keepdim=True)
        for i in range(in_channels):
            new_conv1.weight[:, i:i+1] = mean_w * 0.5   # default "ấm" cho mọi channel
        if rgb_positions is not None and len(rgb_positions) == 3:
            for new_pos, old_pos in zip(rgb_positions, (0, 1, 2)):
                new_conv1.weight[:, new_pos:new_pos+1] = old_conv1.weight[:, old_pos:old_pos+1]
    return new_conv1

# build_resnet50 tự tính rgb_positions = [bands.index(b) for b in ("B04","B03","B02")]
# nếu pretrained=True và in_channels != 3.
```
Thay `model.fc = nn.Linear(2048, 10)`.

**⚠️ Lý do thiết kế:** Phiên bản cũ (`new_conv1.weight[:, :3] = old_conv1.weight[:, :3]`) copy RGB ImageNet weight VÔ ĐIỀU KIỆN vào 3 channel đầu của input. Với `bands: all` ([B01, B02, B03, B04, ...]) prior R/G/B bị áp lên Aerosol/Blue/Green (sai band) và B04 Red thực không nhận prior. Bug này gây **−0.55% acc, −1.13pp Hard F1** trên T1-R2 seed 42 (đo 2026-05-24, xem `paper/results_table.md` section "Option A"). Đã fix bằng API `rgb_positions`. Chi tiết §10 dòng 2026-05-24.

### Training recipe (cố định, mọi run dùng chung)
| Hyperparameter | Giá trị |
|---|---|
| Optimizer | AdamW |
| Initial LR | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR, T_max = epochs |
| Batch size | 32 (RTX 4050 6GB) hoặc 64 (8GB) |
| Epochs | 50 |
| Early stopping | patience=10 trên val accuracy |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Gradient clipping | max_norm=1.0 |
| Mixed precision | torch.cuda.amp |

Ước tính: ~3-5 phút/epoch trên RTX 4050 → ~3-4 giờ/run.

---

## 6. Cấu trúc thí nghiệm 3 tầng

### Tầng 1 — Baseline Establishment (`tier1-baseline`)
Mục đích: thiết lập điểm neo. **Sau khi xong, không sửa lại.**

| Run ID | Input | Channels | Seeds |
|---|---|---|---|
| T1-R1 | RGB | 3 | 42, 123, 2024 |
| T1-R2 | All 13 bands | 13 | 42, 123, 2024 |

**Tiêu chí pass:** T1-R1 đạt 97.5-98.7% (acceptable range so với 98.57% gốc). Nếu < 97% → debug training recipe.

### Tầng 2 — Hypothesis-driven (`tier2-hypothesis`)

| Run | Câu hỏi | Input | Channels | Seeds |
|---|---|---|---|---|
| T2-H2 | 13 bands + 4 indices có hơn 13 bands? | 13 bands + NDVI+NDWI+NDBI+NDMI | 17 | 42, 123, 2024 |
| T2-H3a | NDVI đóng góp bao nhiêu? | 13 bands + NDVI | 14 | 42 |
| T2-H3b | NDWI đóng góp bao nhiêu? | 13 bands + NDWI | 14 | 42 |
| T2-H3c | NDBI đóng góp bao nhiêu? | 13 bands + NDBI | 14 | 42 |
| T2-H3d | NDMI đóng góp bao nhiêu? | 13 bands + NDMI | 14 | 42 |
| T2-H4 | Indices có hữu ích cả khi không có 13 bands? | RGB + 4 indices | 7 | 42 |
| T2-H5a | 10m bands có đủ? | B02,B03,B04,B08 | 4 | 42 |
| T2-H5b | Thêm 20m bands? | 10m + B05,B06,B07,B8A,B11,B12 | 10 | 42 |
| T2-H5c | Atmospheric bands là noise? | 10m + B01,B09,B10 | 7 | 42 |

Tổng: 11 runs.

### Tầng 3 — Tuning (`tier3-tuning`)
Bắt đầu từ cấu hình winner của tầng 2. **Chỉ tune trên train+val, không nhìn test.**

| Group | Sweep | Số run |
|---|---|---|
| T3-A | LR ∈ {3e-5, 1e-4, 3e-4, 1e-3} | 4 |
| T3-B | Weight decay ∈ {1e-5, 1e-4, 5e-4} | 3 |
| T3-C | Augmentation: flip+rot / +RandomResizedCrop / +MixUp / +CutMix | 4 |
| T3-D | Optimizer: AdamW vs SGD+Nesterov | 2 |
| T3-E | Final config với 5 seeds: 42, 123, 2024, 7, 999 | 5 |

Tổng: ~16-18 runs (vài run trùng baseline).

### Tổng budget
- 33 runs × ~3.5h = ~115h GPU.
- Trên RTX 4050 chạy 8-12h/ngày → **10-14 ngày thực tế**.

---

## 7. W&B logging convention

### Project
`eurosat-refine`

### Run naming
`tier{N}-{exp_id}-{config_desc}-seed{S}`

Ví dụ: `tier2-H3a-13bands+NDVI-seed42`

### Tags
- Mỗi run phải có tag tier: `tier1-baseline` / `tier2-hypothesis` / `tier3-tuning`
- Tag thêm theo experiment ID: `H2`, `H3a`, etc.
- Tag model: `resnet50`

### Description (notes) — template
```
Question: [câu hỏi cụ thể]
Hypothesis: [giả thuyết kiểm chứng]
Baseline reference: [run ID hoặc paper reference để so sánh]
Expected outcome: [con số dự kiến]
```

### Config (wandb.config) — bắt buộc
```python
{
    "model": "resnet50",
    "input_channels": N,
    "input_type": "rgb" | "13bands" | "13bands+indices" | ...,
    "indices_used": ["NDVI", "NDWI", ...],
    "optimizer": "adamw",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "augmentation": "flip+rotate",
    "seed": seed,
    "train_split": "80/20-stratified-seed42",
}
```

### Metrics log mỗi epoch
- `train/loss`, `train/acc`
- `val/loss`, `val/acc`, `val/macro_f1`
- `lr`, `epoch`

### Metrics cuối training
- `test/acc`, `test/macro_f1`
- `test/per_class_f1` (dict)
- `test/confusion_matrix` (qua `wandb.plot.confusion_matrix`)
- Lưu best model checkpoint qua `wandb.Artifact`

### Nguyên tắc
- **KHÔNG xoá run thất bại.** Tag `tier-failed` nếu cần đánh dấu.
- **Set seed và log seed** cho mọi run.
- **Log đủ config qua wandb.config**, không hard-code trong script.

---

## 8. Checklist trước khi chạy mỗi run

- [ ] Set seed (torch, numpy, random, cudnn deterministic)
- [ ] Tên run đúng convention `tier{N}-{exp_id}-{desc}-seed{S}` (KHÔNG dùng ký tự `+` trong config_desc — wandb.Artifact reject; đã có sanitize ở `src/utils/wandb_utils.py:163`)
- [ ] Description đầy đủ (Question, Hypothesis, Baseline ref)
- [ ] Tag đúng tier
- [ ] `wandb.config` đầy đủ
- [ ] Kiểm tra mapping band index (B08 ở đâu? B11 ở đâu?)
- [ ] **Verify conv1 RGB-prior align đúng band** khi input ≠ 3ch + pretrained=True — `build_resnet50(bands=...)` phải nhận bands list để tự align (xem §5 ghi chú bug 2026-05-24)
- [ ] Verify train/val/test split không leak (check file hash)
- [ ] Early stopping patience đúng (10)
- [ ] Save best checkpoint theo val accuracy
- [ ] Normalization stats từ TRAIN set (không phải toàn bộ data)

---

## 9. Sản phẩm cuối phase Refine

1. **W&B project** với ~33 runs đầy đủ tag, config, metrics, artifacts.
2. **Bảng kết quả tổng hợp** so sánh tất cả cấu hình với baseline gốc và baseline reproduce.
3. **Mô hình winner** (checkpoint + config) — sẽ là input cho phase Research.
4. **Báo cáo Refine** 3-5 trang: motivation, method, ablation, discussion, limitations.

---

## 10. Decisions log (quan trọng — cập nhật khi có thay đổi)

| Ngày | Quyết định | Lý do |
|---|---|---|
| (init) | Chọn phương án multi-spectral fusion + 4 indices cho Refine | Gắn chặt với gap chính của bài gốc, có 2 metric độc lập |
| (init) | Dùng cấu trúc 3 tầng (baseline / hypothesis / tuning) | Tránh HARKing, log W&B sẽ rõ ràng cho reviewer |
| (init) | Dùng cả 4 indices (NDVI, NDWI, NDBI, NDMI) | Mỗi index nhắm một cặp lớp khó khác nhau |
| (init) | GPU: RTX 4050 → batch_size 32, mixed precision | Giới hạn VRAM 6-8GB |
| (init) | Đi đủ 3 tầng | Người dùng xác nhận |
| 2026-05-22 | Đổi tên folder dataset: `2750/` → `rgb/`, `tif/` → `allbands/`; thêm `dataset/splits/` | Tên rõ vai trò: rgb = visualization, allbands = training (13 bands), splits = file split cố định seed 42 |
| 2026-05-23 | Verify & cố định band order trong allbands TIFF: B8A ở idx 12 (cuối), B11 ở idx 10 — khác torchgeo canonical. Ghi vào §4. | Phòng ngừa lỗi NDBI/NDMI tính sai band mà không báo lỗi (cảnh báo §4). Verify bằng 3 cách độc lập: per-band profile theo lớp, notebook `verify_allbands_loader.ipynb`, Tier-1 RGB khớp paper (98.52% vs 98.57%). |
| 2026-05-23 | Commit `dataset/splits/` (đổi `.gitignore` từ `dataset/` → `dataset/*` + `!dataset/splits/`) | Split files (seed 42, 21600/2700/2700) cần version-controlled để tái lập; ảnh raw vẫn ignore vì nặng. |
| 2026-05-23 | **Chốt RQ1 (SSL pretrain → fine-tune varying label %) cho phase Research.** | Đánh thẳng vào số liệu cụ thể của paper §4.5 (~75% @ 10% train data); SSL có codebase open-source dồi dào (MAE / DINOv2 / SatMAE / Prithvi); Sentinel-2 unlabeled data miễn phí qua Copernicus/GEE/BigEarthNet. Winner Refine làm baseline ImageNet-pretrained để so sánh. |
| 2026-05-24 | **Phát hiện + fix bug conv1 RGB-prior misalignment** (`src/models/resnet.py`). API mới: `build_resnet50(bands=...)` + `_adapt_conv1(rgb_positions=...)` — RGB prior align theo vị trí B04/B03/B02 trong input thay vì copy mù vào 3 channel đầu. Test trên T1-R2 seed 42: Δ +0.55% acc, +1.13pp Hard F1 → bug MAJOR. Bare-minimum re-run T1-R2 (seed 123/2024) + T2-H3c (4 seeds). Skip re-run T2-H2/H3a/H3b/H3d (paired Δ valid). T1-R1 RGB và T2-H4 KHÔNG bị bug (verified). Chi tiết `paper/results_table.md` section "Option A". | Caller-responsibility contract trong docstring KHÔNG đủ — config `bands: all` vi phạm contract im lặng. Symptom "13b không vượt RGB" (paired Δ −0.04%) là smoking gun bị bỏ qua vì rationalize sai. Phát hiện do skeptical audit khi user hỏi "có khả năng tính sai?". Fix mới enforce ở boundary thay vì rely trên caller. |
| 2026-05-25 | Re-run T2-H2 (3 seeds) + H3a/b/d (1 seed each) fixed → **đảo winner Tier 2 thành T1-R2 alone** | Paired Δ multi-seed: H3c-T1-R2 = −0.17pp, H2-T1-R2 = −0.23pp (đều trong noise band SE ±0.27-0.32pp). Indices không cho thấy improvement có ý nghĩa thống kê sau fix. |
| 2026-05-25 | T2-H5 indices alone (4 channels) = 97.44%, Hard F1 95.44% | Direct test refute "indices = engineered value độc lập". Indices alone tệ HƠN RGB alone (−1.08pp acc, −1.82pp Hard F1) — confirm H_A: indices = lossy compression. Caveat: cũng không nhận ImageNet prior. |
| 2026-05-25 | Tier 3 T3-C aug sweep (RRC/MixUp/CutMix) + T3-A LR + T3-B WD + T3-D optim + T3-E SGD 5-seed final | Không config nào vượt baseline T1-R2 fixed có ý nghĩa thống kê. T3-E SGD 5 seeds = 98.73% (thua AdamW baseline −0.12pp acc, −0.16pp Hard F1). Single-seed sweep gives illusion of winners — T3-D s42 lucky outlier. |
| 2026-05-25 | **Winner Refine cuối: T1-R2 13b raw + AdamW default recipe** (mean 98.85% ± 0.24%, Hard F1 97.93%) | Simplest config + best Hard F1 + best stability. Sẽ là baseline ImageNet-pretrained (condition B) cho Research phase RQ1. |
| 2026-05-25 | **Must-have Hard F1 +1.5pp vs T1-R1 KHÔNG đạt** (best 97.93% vs target 98.76%, gap −0.83pp) | Toàn bộ input engineering (indices) + aug (MixUp/CutMix/RRC) + LR/WD/optim sweeps đều không break được gap Hard-Easy. Gap còn lại nhiều khả năng do label noise giữa vegetation classes + 64×64 resolution limit. Negative finding đáng publish. |
| 2026-05-26 | **Protocol freeze Research phase (updated.md):** giữ NGUYÊN split 80/10/10 seed 42; KHÔNG chuyển sang 10/90; chỉ subsample TRAIN cho low-label; val + test bất biến mọi experiment | Modern SSL evaluation không đổi global split (val proper, test fair). 10/90 paper gốc outdated — insufficient validation separation, overly large test, khó so sánh fair giữa methods. Câu hỏi Research là "cần bao nhiêu label?" KHÔNG phải "chia dataset thế nào?". |
| 2026-05-26 | **Chia RQ1 thành RQ1-A (core, bắt buộc) + RQ1-B (optional)** | RQ1-A giữ ResNet-50 backbone giống Refine để isolate "representation learning effect"; SSL methods: SimCLR / MoCo-v2 / BYOL (contrastive, dễ compare, low engineering complexity). Nếu nhảy thẳng MAE → đổi đồng thời backbone (ResNet→ViT) + objective + tokenization → không attribute được gain cho SSL hay architecture. MAE/SatMAE/ViT moved sang RQ1-B optional sau khi RQ1-A xong. |
| 2026-05-26 | **4 bước trước SSL nặng:** (1) Protocol freeze (2) Condition B low-label baseline curve (3) Verify stability (4) Tiny SSL pilot | Tránh launch full 100-200 epoch SSL trước khi verify pipeline; phải có "supervised degradation curve" của ImageNet baseline để biết ngưỡng SSL cần vượt; tiny SSL pilot verify loss giảm + embeddings non-trivial trước khi commit compute lớn. |
| 2026-05-26 | **Triết lý project chuyển dịch: "benchmark optimization" → "representation learning analysis"** | Mục tiêu mới = hiểu data efficiency + representation quality + transfer behavior; KHÔNG còn chạy theo +0.1pp ở 100%. Khoảng cách giữa 3 condition (A/B/C) ở 1-10% label là tâm điểm. |

---

## 11. Hướng Research (giai đoạn tiếp theo) — **RQ1 chia 2 sub-phase: RQ1-A (core) + RQ1-B (optional)**

> **Protocol freeze 2026-05-26** (xem `updated.md` cho bản đầy đủ). Research giữ NGUYÊN split 80/10/10 (21600/2700/2700) seed 42 từ Refine. KHÔNG chuyển sang 10/90 như paper gốc. Chỉ subsample TRAIN cho low-label regime — val + test BẤT BIẾN mọi experiment.

### RQ1 — Câu hỏi nghiên cứu

> SSL pretrain trên Sentinel-2 không nhãn có giảm nhu cầu nhãn labeled không?

**Tham chiếu paper gốc (Helber 2019, §4.5 + Figure 7):** acc giảm mạnh khi train set nhỏ — ở 10% train data chỉ đạt ~75%. Paper không thử SSL → khe hở "low-data regime" (CLAUDE.md §2, đánh dấu "Cao — dành cho Research").

> **Lưu ý quan trọng:** "low-data regime" reproduce **CONCEPT** của paper, KHÔNG reproduce literal 10/90 split. Biến quan trọng là *lượng nhãn training*, không phải tỷ lệ train/test toàn cục.

### RQ1-A — Controlled SSL Study (CORE, BẮT BUỘC)

**Mục tiêu:** isolate "representation learning effect", giữ continuity với Refine, giảm confounders.

**Backbone CỐ ĐỊNH = ResNet-50** (giống Refine). Lý do: nếu đổi đồng thời backbone (ResNet→ViT) + SSL objective + tokenization → không attribute được gain cho SSL hay architecture.

**SSL methods ưu tiên:** SimCLR / MoCo-v2 / BYOL (contrastive — dễ compare với supervised baseline, low engineering complexity, clean experimental design).

**Ma trận chính: 3 condition × 4 label fraction = 12 run**

| Condition | Backbone | Init |
|---|---|---|
| (A) Scratch | ResNet-50 | Random — sàn dưới |
| **(B) ImageNet** | ResNet-50 | **= winner Refine T1-R2 fixed** (frozen baseline) |
| (C) SSL | ResNet-50 | Contrastive SSL trên Sentinel-2 unlabeled |

**Label fractions (stratified seed 42, subsample từ TRAIN, nested):**

| Fraction | Samples |
|---|---|
| 1% | 216 |
| 5% | 1080 |
| 10% | 2160 |
| 100% | 21600 |

Nested: 1% ⊂ 5% ⊂ 10% ⊂ 100%. Val (2700) + Test (2700) GIỮ NGUYÊN cho mọi run.

**Metric:** acc, macro-F1, Hard-class F1, **data-efficiency curve** (acc vs label %).

### RQ1-B — MAE / Foundation Extension (OPTIONAL)

Chỉ chạy SAU KHI RQ1-A xong + baseline low-label curve ổn định + còn compute.

**Mở rộng sang:** ViT-Small / MAE / SatMAE / DINOv2 / Prithvi / SatDINO — đổi đồng thời architecture + SSL objective + tokenization.

**Câu hỏi mở rộng:** "Masked spectral-spatial reconstruction (MAE/SatMAE) có outperform contrastive SSL cho multispectral low-label?"

**Vì sao MAE/SatMAE promising cho remote sensing:** Sentinel-2 có strong spatial + spectral redundancy (SWIR↔NIR correlation, vegetation structure shared cross bands) → reconstruction objective phù hợp. Contrastive SSL ở remote sensing khó vì augmentation dễ phá spectral meaning (color jitter không physically valid). MAE ít phụ thuộc augmentation mạnh.

### 4 bước tiến hành (TRƯỚC khi launch SSL nặng)

1. **Protocol Freeze** — không tune supervised thêm; chốt splits, metrics, recipe, seed policy, evaluation.
2. **Condition B low-label baseline curve** — chạy winner Refine (T1-R2 fixed, ImageNet) trên 1% / 5% / 10% / 100% → "supervised degradation curve" làm baseline ngưỡng SSL cần vượt.
3. **Verify low-label stability** — subset generation đúng stratification, variance qua seeds, training stability ở fraction nhỏ.
4. **Tiny SSL pilot** — chạy nhỏ (epoch ít, batch nhỏ) verify: SSL loss giảm, embeddings non-trivial, fine-tune work, data pipeline correct, memory stable. **KHÔNG launch full 100-200 epoch SSL ngay.**

### Tái dùng từ Refine (DONE 2026-05-25)

- **Condition B (frozen baseline ImageNet)** = winner Refine T1-R2 13b raw + AdamW (mean 98.85% ± 0.24%, Hard F1 97.93%)
- Training recipe: AdamW lr=1e-4, wd=1e-4, batch=32, cosine T_max=50, patience=10, label_smoothing=0.1, flip+rot90
- Splits seed 42 (21600/2700/2700) + thêm stratified subsets 1/5/10/100% cho Research
- **KHÔNG dùng indices ở Research** — confirmed không có domain value sau fix bug conv1 (xem `paper/results_table.md` §7.1 F5)
- Conv1 adapter API `build_resnet50(bands=...)` extend cho SSL load 12→13 channels (BigEarthNet thiếu B10) → apply rule §13 A1/B1 enforce ở boundary
- Toàn bộ rule §13 (bộ quy tắc phòng ngừa lỗi) áp dụng cho mọi adapter SSL mới

### Triết lý chuyển dịch (Research phase)

Project chuyển từ **"benchmark optimization"** sang **"representation learning analysis"**:

- Mục tiêu cũ: maximize supervised accuracy
- Mục tiêu mới: hiểu **data efficiency** + **representation quality** + **transfer behavior** ở low-label
- +0.1pp ở 100% không còn quan trọng; **khoảng cách giữa (A)/(B)/(C) ở 1-10% label** là tâm điểm
- Main evaluation focus: low-label regime (1% / 5% / 10%), KHÔNG phải squeeze gain ở 100%

### Guard rails Research phase (theo updated.md §Final Guidance)

1. **NEVER** regenerate global dataset splits
2. **NEVER** switch to literal 10/90 train/test evaluation
3. **ALWAYS** keep val + test fixed
4. **ONLY** subsample from TRAIN, stratified seed 42
5. Treat **T1-R2 fixed = official supervised baseline** (condition B)
6. Refine **COMPLETE** — không reopen trừ khi phát hiện critical methodological bug
7. SSL experiments phải **isolate representation learning effect**, KHÔNG confound với optimizer tuning / architecture scaling
8. RQ1-A core trước, RQ1-B optional sau

### Các RQ còn lại (không chọn, lưu để tham khảo)

- RQ2: Foundation models cho remote sensing (SatMAE, Prithvi, SatDINO) — zero/few-shot performance
- RQ3: Cross-domain generalization — train EuroSAT (châu Âu) → test AID/NWPU (Trung Quốc/khác)
- RQ-A: Transformer (ViT/Swin) có lợi gì ngoài accuracy? Calibration, data efficiency, interpretability

---

## 12. Tài liệu tham khảo chính

- Helber et al. 2019 (paper gốc): arXiv:1709.00029
- Dataset: https://github.com/phelber/EuroSAT
- torchgeo: https://torchgeo.readthedocs.io/
- W&B docs: https://docs.wandb.ai/

---

## 13. Bộ quy tắc phòng ngừa lỗi do bất cẩn / thiếu kiểm chứng (BẮT BUỘC cho Claude Code)

> Trigger: rút từ post-mortem bug conv1 RGB-prior 2026-05-24 (§10). Bug đó sống 2 ngày, ảnh hưởng 7/9 run 13-band, đảo finding khoa học — vì 8 failure mode đồng thời (caller-responsibility không enforce, smoking gun bị rationalize, paired Δ che absolute error, v.v.). Mọi quy tắc dưới đây ÁP DỤNG BẮT BUỘC. Nếu Claude vi phạm, user có quyền chỉ ra để correction; nếu Claude phải bỏ một quy tắc trong tình huống cụ thể, phải nói rõ TẠI SAO trước khi bỏ.

### A. Khi viết / sửa code (defensive coding)

**A1. Cấm "caller responsibility" trong docstring đứng một mình.**
Nếu hàm có assumption về input (vd "channel 0 phải là Red", "weight phải đã set seed"), phải enforce ở code: `raise ValueError` nếu vi phạm, HOẶC compute từ context internal (như `_adapt_conv1(rgb_positions=...)` mới — caller truyền `bands`, hàm tự suy ra position). Docstring KHÔNG đủ — đã chứng minh bằng bug conv1.

**A2. Khi config có nhiều preset/variant** (vd `bands: rgb | all | 10m | atmospheric`): liệt kê từng assumption downstream cho TỪNG variant trước khi merge. Nếu một variant vi phạm assumption của một downstream module → fix module để khái quát, KHÔNG trust convention ngầm "ai cũng biết".

**A3. Đụng pretrained weights → phải có position-to-semantic mapping explicit.** Không bao giờ làm `weight[:, :K] = pretrained_weight[:, :K]` mà không kiểm tra `K` channel đó có cùng nghĩa với `K` channel pretrained không.

**A4. Verification scope phải explicit ở header của notebook/script.** Mỗi file verify phải có dòng đầu rõ ràng: `# Verify scope: DATA pipeline only` HOẶC `# Verify scope: MODEL adapter only` HOẶC `# Verify scope: END-TO-END (data → loss)`. Cell kết luận CHỈ được claim trong phạm vi đã test. Bug conv1 sống được phần lớn vì `notebooks/verify_allbands_loader.ipynb` claim "loader sẵn sàng cho T1-R2 và Tầng 2" — câu sau overreach ra ngoài scope loader, tạo false confidence.

### B. Khi code đụng boundary (data ↔ model, position ↔ semantic, layer ↔ layer)

**B1. Adapter/transform phải có weight-alignment test programmatic** trước khi train run dài. Test dạng:

```python
m = build_model(...)
assert torch.allclose(m.conv1.weight[:, pos_B04], imagenet_red_filter)
assert torch.allclose(m.conv1.weight[:, pos_B01], mean_w * 0.5)
```

Không có test = không launch run dài.

**B2. Swap backbone / change channel count / re-map index** = phải inspect `model.conv1.weight.shape`, `[:, i].mean()`, `n_params` programmatically và compare với reference TRƯỚC khi train. Print ra log để có audit trail.

### C. Khi đọc / phân tích kết quả (skeptical reading)

**C1. Trigger audit khi kết quả phản trực giác về physics/domain.** Các trigger cụ thể:

- "Thêm 10× thông tin input → Δ ≈ 0" → audit code (KHÔNG rationalize "domain noise")
- "Pretrained beats scratch by < 1% trên transfer task" → audit
- "Domain-specific feature (vd index nhắm Hard class) không cải thiện Hard class" → audit
- "Cải thiện nhỏ hơn paper baseline / literature expectation > 50%" → audit

Quy tắc cứng: rationalize bằng giả thuyết domain CHỈ SAU KHI đã audit code và data. Bug conv1 chính là smoking gun "13b không vượt RGB" bị rationalize sai 2 ngày.

**C2. Match một metric với paper ≠ end-to-end correctness.** Chỉ chứng minh code path duy nhất đã test. Mọi code path khác (vd `bands: all` vs `bands: rgb`, `pretrained=True` vs `False`, `in_channels==3` vs `!= 3`) phải verify riêng. Bug conv1 sống được vì T1-R1 RGB khớp paper 98.52% ≈ 98.57% → false confidence → không ai check 13-band path.

**C3. Internal consistency KHÔNG = correctness.** Paired Δ hợp lý + std nhỏ + ranking ổn định giữa các run CÙNG bug → vẫn có thể sai absolute. Phải có external sanity: paper baseline, physics expectation, ngưỡng từ domain literature. Bug conv1 có std cực nhỏ (0.07%) — tưởng là dấu hiệu training stable, thực chất là dấu hiệu "mọi seed đều bị cùng handicap giống nhau".

### D. Trước khi launch run dài (≥ 1h GPU)

**D1. Smoke test bắt buộc** với EXACT config sắp chạy:

```bash
python -m scripts.train --config <config> --seed 42 --epochs 1 --limit 64 \
  --device cpu --wandb-mode disabled --num-workers 0
```

Verify: model build OK, 1 batch forward + backward không crash, val/test loop chạy được, output shape đúng.

**D2. Inspect model state sau build, trước fit.** In log: `in_channels`, `n_params`, weight stats của conv1 (mean, std, shape) + fc, optimizer param groups. Đây là audit trail nếu sau này nghi ngờ.

**D3. So config với §6 checklist:** verify nothing skipped (seed, naming, tags, wandb.config, band index mapping, conv1 RGB-prior alignment, split leak check, etc.).

### E. Khi user yêu cầu skeptical audit ("có khả năng sai không?")

**E1. Audit FULL stages, không skip.** Stage list bắt buộc:

1. **Data:** loader đọc đúng band mapping? Splits không leak (file hash)? Class balance kỳ vọng?
2. **Preprocess:** Stats tính trên TRAIN only? Per-channel? Sau khi compute indices?
3. **Indices:** Công thức đúng band? Sanity vật lý (NDBI/NDMI anti-symmetric, NDVI Forest > 0.5)?
4. **Model build:** Weight alignment với assumption? `n_params` đúng?
5. **Forward:** 1 sample produce output shape đúng? Gradient flow tới mọi param?
6. **Loss/optimizer:** Loss reasonable ở epoch 0? LR schedule đúng?

Output: status per stage (✓ verified / ⚠ potential issue / ❌ confirmed bug), KHÔNG generic "everything looks good".

**E2. Sau audit, nếu tìm thấy bug ứng cử viên** (chưa proven), đề xuất TEST cụ thể (vd "chạy 1 seed với fix, compare paired Δ") TRƯỚC khi proven major/minor.

**E3. Kiểm tra coverage matrix khi audit.** Liệt kê tất cả các verify đã thực hiện (notebook, script, CLAUDE.md §4-like) và SCOPE của chúng. Nếu mọi verify cluster vào 1-2 stage trong 6 stage (E1), đó là **systematic blind spot** — phải tìm verify ở stage chưa cover. Bug conv1: cả 3 cách verify §4 CLAUDE.md cũ đều là data-layer (per-band profile, loader notebook, RGB baseline match paper) — KHÔNG có model-layer verify. Coverage matrix lẽ ra phải lộ điều này.

### F. Khi document

**F1. Decisions log §10 update khi có change correctness-affecting.** Reason field BẮT BUỘC kỹ thuật — không phải "vì user yêu cầu" mà phải nói TẠI SAO về mặt code/science. Format: `| ngày | quyết định | lý do (kỹ thuật) |`.

**F2. Mỗi bug discovered phải có post-mortem:**

- Memory file riêng `eurosat-<bug-name>.md` với: phát hiện thế nào, impact đo bằng số, fix scope, run nào bị/không bị
- Section "Bug X" trong `paper/results_table.md` với bảng số liệu before/after
- Update §10 Decisions log với lý do kỹ thuật

**F3. KHÔNG xoá artifact của bug** (W&B runs, configs, log files cũ). Đó là evidence cho paper Discussion section "Limitations & Bugs Found".

### G. Meta-rule

**G1. Nếu Claude cảm thấy mình đang "skip" một quy tắc vì lý do thời gian / convenience** → phải báo user và xin xác nhận. Vd: "Tôi sẽ skip smoke test D1 cho run này vì config y hệt run trước. OK không?"

**G2. Khi bug được phát hiện, Claude phải tự update §13** nếu failure mode mới chưa có quy tắc tương ứng. Quy tắc này là living document.

---

## Hướng dẫn cho Claude Code

Khi làm việc trong dự án này:

1. **Đọc file này đầu phiên** để hiểu context. Không cần hỏi lại các quyết định đã ghi ở mục 10.
2. **Tôn trọng cấu trúc 3 tầng** — không skip thẳng sang tuning trước khi xong baseline + hypothesis.
3. **Mỗi script training phải có W&B integration** theo convention ở mục 7.
4. **Khi tôi yêu cầu chạy thí nghiệm mới**, kiểm tra xem nó thuộc tầng nào và đã được thiết kế ở mục 6 chưa. Nếu chưa, hỏi tôi trước khi tự thêm.
5. **Khi viết code preprocessing**, lưu ý cảnh báo về band index mapping ở mục 4.
6. **Khi đụng model adapter cho N-channel input** (`build_resnet50`, `_adapt_conv1`, hoặc bất kỳ swap backbone nào): luôn verify weight alignment bằng unit test trước khi train, KHÔNG rely vào docstring "caller responsibility". Xem bug §10 dòng 2026-05-24 — caller-responsibility contract đã sống lâu ngày mà không bị enforce. Adapter mới phải enforce ở boundary (raise ValueError nếu input không đáp ứng assumption).
7. **Audit skeptical định kỳ:** khi kết quả phản trực giác về mặt physics/domain (vd "thêm 10 bands không cải thiện gì"), KHÔNG rationalize bằng giả thuyết domain; phải audit code trước. Smoking gun của bug 2026-05-24 chính là finding "13b không vượt RGB" (paired Δ −0.04%) bị bỏ qua 2 ngày.
8. **Khi có quyết định thiết kế mới**, cập nhật mục 10 (Decisions log).
9. **Không tự ý xoá hoặc disable W&B runs** dù chúng có vẻ thất bại — đó là evidence cho paper. Runs "Failed" do `wandb.Artifact()` reject `+` (đã fix sanitize 2026-05-24) là KHÔNG phải lỗi training, dữ liệu test đã log Summary trước khi crash.
