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
- Split: 80/20 train/test stratified, cố định bằng seed 42. Trong train tách 10% làm val.
- Lưu split thành files cố định: `splits/train.txt`, `val.txt`, `test.txt`.

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
ResNet-50 pretrained ImageNet, điều chỉnh conv1:
```python
new_conv1 = nn.Conv2d(N, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv1.weight[:, :3] = old_conv1.weight  # giữ RGB ImageNet prior
    mean_w = old_conv1.weight.mean(dim=1, keepdim=True)
    for i in range(3, N):
        new_conv1.weight[:, i:i+1] = mean_w * 0.5  # khởi tạo "ấm" cho bands khác
```
Thay `model.fc = nn.Linear(2048, 10)`.

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
- [ ] Tên run đúng convention `tier{N}-{exp_id}-{desc}-seed{S}`
- [ ] Description đầy đủ (Question, Hypothesis, Baseline ref)
- [ ] Tag đúng tier
- [ ] `wandb.config` đầy đủ
- [ ] Kiểm tra mapping band index (B08 ở đâu? B11 ở đâu?)
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

---

## 11. Hướng Research (giai đoạn tiếp theo) — **RQ1 đã chốt**

Sau Refine sẽ chuyển sang Research phase (tuần 6-7). **Đã chốt RQ1** (2026-05-23).

### RQ1 (chốt) — Self-Supervised Learning cho low-data regime

**Câu hỏi:** SSL pretrain trên Sentinel-2 không nhãn có giảm nhu cầu nhãn labeled không?

**Tham chiếu paper gốc (Helber et al. 2019, §4.5 + Figure 7):**
- Acc giảm mạnh khi training set nhỏ — ở **10% train data chỉ đạt ~75%**.
- Paper không thử bất kỳ phương pháp giảm phụ thuộc nhãn nào → khe hở "low-data regime" (xem CLAUDE.md §2, đánh dấu "Cao — dành cho Research").

**Phương pháp:**
1. **Pretrain SSL** (không nhãn) — MAE làm baseline (well-studied, đơn giản), tham khảo SatMAE/Prithvi nếu thời gian cho phép. Data: BigEarthNet hoặc tile Copernicus ngoài EuroSAT. Backbone: ViT-Small hoặc ResNet-50 để match Refine.
2. **Fine-tune** (supervised) trên **1% / 5% / 10% / 100%** train EuroSAT (subset stratified seed 42 — tái dùng splits từ Refine). Recipe y hệt Refine.
3. **So sánh 3 điều kiện × 4 label fractions:**
   - (A) Scratch (random init) — sàn dưới
   - (B) ImageNet pretrained — **baseline = winner Refine**
   - (C) SSL pretrained — đề xuất RQ1
4. **Metric:** acc, macro-F1 hard classes, **data-efficiency curve** (acc vs label %).

**Tái dùng từ Refine:** training recipe, splits cố định (seed 42), baseline số liệu, cấu hình input của winner Tầng 3.

### Các RQ còn lại (không chọn, lưu để tham khảo)

- RQ2: Foundation models cho remote sensing (SatMAE, Prithvi, SatDINO) — zero/few-shot performance.
- RQ3: Cross-domain generalization — train EuroSAT (châu Âu) → test AID/NWPU (Trung Quốc/khác).
- RQ-A: Transformer (ViT/Swin) có lợi gì ngoài accuracy? Calibration, data efficiency, interpretability.

Mô hình winner của Refine sẽ là baseline cho RQ1.

---

## 12. Tài liệu tham khảo chính

- Helber et al. 2019 (paper gốc): arXiv:1709.00029
- Dataset: https://github.com/phelber/EuroSAT
- torchgeo: https://torchgeo.readthedocs.io/
- W&B docs: https://docs.wandb.ai/

---

## Hướng dẫn cho Claude Code

Khi làm việc trong dự án này:

1. **Đọc file này đầu phiên** để hiểu context. Không cần hỏi lại các quyết định đã ghi ở mục 10.
2. **Tôn trọng cấu trúc 3 tầng** — không skip thẳng sang tuning trước khi xong baseline + hypothesis.
3. **Mỗi script training phải có W&B integration** theo convention ở mục 7.
4. **Khi tôi yêu cầu chạy thí nghiệm mới**, kiểm tra xem nó thuộc tầng nào và đã được thiết kế ở mục 6 chưa. Nếu chưa, hỏi tôi trước khi tự thêm.
5. **Khi viết code preprocessing**, lưu ý cảnh báo về band index mapping ở mục 4.
6. **Khi có quyết định thiết kế mới**, cập nhật mục 10 (Decisions log).
7. **Không tự ý xoá hoặc disable W&B runs** dù chúng có vẻ thất bại — đó là evidence cho paper.
