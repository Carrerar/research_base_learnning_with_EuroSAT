---
marp: true
theme: default
paginate: true
size: 16:9
header: "Multi-Spectral Fusion & SSL on EuroSAT — DPL302M"
footer: "2026-05-23 · Refine: Tier 1 ✅  Tier 2 ⏳  ·  Research: RQ1 (SSL)"
---

# Hợp nhất đa phổ và học tự giám sát hiệu quả nhãn

### cho phân loại sử dụng đất Sentinel-2 trên bộ dữ liệu EuroSAT

---

*Multi-Spectral Fusion and Label-Efficient Self-Supervised Learning*
*for Sentinel-2 Land Use Classification on EuroSAT*

---

**DPL302M — Research-Based Learning Project**
Phase **Refine** (3/5): Tầng 1 ✅ · Tầng 2 ⏳ · Tầng 3 ⏳   →   Phase **Research**: RQ1 (SSL)
2026-05-23

---

## 1 · Khe hở của paper gốc

Helber et al. 2019 đạt **98.57% (RGB)** — nhưng bỏ ngỏ:

- ❌ Không có model **"13-band native"** — chỉ test combinations 3-band (RGB / CI / SWIR)
- ❌ Không dùng **spectral indices** (NDVI/NDWI/NDBI/NDMI) — domain knowledge remote sensing bỏ phí
- ⚠️ 4 lớp nông nghiệp/đồng cỏ dễ nhầm: `AnnualCrop`, `PermanentCrop`, `Pasture`, `HerbaceousVegetation`
- ⚠️ Single number, **không std, không seed** → không kiểm tra significance

→ Refine khai thác 2 khe đầu, đo bằng **2 metric độc lập**

---

## 2 · Phương án Refine đã chốt

**Input:** 13 bands + 4 indices = **17 channels**
**Model:** ResNet-50 pretrained ImageNet (conv1 adapt N-channel)

| Metric | Vai trò |
|---|---|
| Overall accuracy | So với baseline paper (98.57%) |
| **Macro-F1 trên 4 lớp khó** | Đo trực tiếp đóng góp của bands/indices |

> Cả **kết quả âm** (không vượt RGB) cũng là phát hiện đáng báo cáo —
> tránh HARKing, log giả thuyết vào W&B *trước* khi chạy.

---

## 3 · Mục tiêu cốt lõi & tiêu chí thành công

**Must-have (sàn — phải đạt):**
- Cấu hình winner **≥ baseline RGB** về accuracy
- Macro-F1 trên 4 lớp khó **tăng ≥ 1.5%** (paired vs RGB)

**Nice-to-have (đích phấn đấu):**
- Overall accuracy **≥ 99.0%**
- Macro-F1 lớp khó **tăng ≥ 2.0%**

**Acceptable (cũng tính là kết quả):**
- Negative finding: multi-spectral KHÔNG vượt RGB → vẫn báo cáo nếu có giả thuyết rõ + W&B đầy đủ
- Hard-class gap không thu hẹp → kết luận về **giới hạn của domain-knowledge approach**

> 📌 **Triết lý:** *hiểu được vì sao* quan trọng hơn số đẹp.
> Mỗi run = 1 câu hỏi + 1 giả thuyết, log W&B **trước** khi chạy → tránh HARKing.

💡 Tầng 1 đã cho thấy `13bands raw ≈ RGB` → must-have chuyển trọng tâm sang **H2 (13bands + indices)** là phép thử quyết định.

---

## 4 · 4 Spectral Indices

```
NDVI = (B08 - B04) / (B08 + B04)   → vegetation   (AnnualCrop, Pasture, Herbaceous)
NDWI = (B03 - B08) / (B03 + B08)   → water        (River, SeaLake)
NDBI = (B11 - B08) / (B11 + B08)   → built-up     (Industrial, Residential)
NDMI = (B08 - B11) / (B08 + B11)   → moisture     (Forest, Herbaceous)
```

**Cảnh báo (đã verify 2026-05-23):**
File TIFF EuroSAT allbands lưu **B11 ở idx 10, B8A ở idx 12 (cuối)** — KHÁC thứ tự torchgeo canonical.
→ Loader truy cập bands **theo TÊN** qua `BAND_TO_IDX` (an toàn). Mọi code mới đọc raw TIFF phải check bảng band order ở `CLAUDE.md §4`.

---

## 5 · Cấu trúc 3 tầng

| Tầng | Mục đích | # runs | Time |
|---|---|---|---|
| **Tầng 1** Baseline | Reproduce paper, neo so sánh | 6 (2 config × 3 seeds) | ~21h |
| **Tầng 2** Hypothesis | 11 câu hỏi có giả thuyết tường minh | 11 | ~36h |
| **Tầng 3** Tuning | Tinh chỉnh winner Tầng 2 | ~16-18 | ~60h |
| | **Tổng** | **~33** | **~115h GPU** |

GPU: RTX 4050 (6GB) · ~3-5 phút/epoch · ~3.5h/run

---

## 6 · Setup chung (cố định mọi run)

| | |
|---|---|
| Dataset | EuroSAT **allbands TIFF**, float32, 27,000 ảnh, 10 lớp |
| Split | 80/10/10 stratified, **seed 42** → 21600 / 2700 / 2700 |
| Model | ResNet-50 (IMAGENET1K_V2), conv1 N-channel adapt, fc→10 |
| Optimizer | AdamW · lr 1e-4 · weight_decay 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Augment | Flip H/V + rot90 — **KHÔNG color jitter** (phá spectral relations) |
| Stop | Early stop val_acc, patience 10 |
| Other | label_smoothing 0.1 · grad clip 1.0 · AMP |

---

## 7 · Pipeline xử lý dữ liệu

**Setup (1 lần):**
1. `create_splits.py` → stratified seed 42 → `dataset/splits/{train,val,test}.txt`
2. `compute_stats.py` → mean/std **trên TRAIN-only** → `stats/channel_stats.json` (13 bands + 4 indices)

**Mỗi sample (lúc training):**
1. Read TIFF → tensor `(13, 64, 64)` float32
2. **Compute 4 indices từ RAW bands** (trước normalize, giữ ý nghĩa vật lý)
3. Select bands theo **TÊN** qua `BAND_TO_IDX`
4. Concat bands + indices
5. Normalize per-channel (stats train-only)
6. Train: flip H/V + rot90 ngẫu nhiên

---

## 8 · Tầng 1 — Thiết kế

| Run ID | Input | Channels | Seeds |
|---|---|---|---|
| T1-R1 | RGB (B04, B03, B02) | 3 | **42, 123, 2024** |
| T1-R2 | All 13 bands | 13 | **42, 123, 2024** |

**Vì sao 3 seeds, vì sao CÙNG bộ số?**
- Stratified split + random augmentation → 1 run dễ bị nhiễu
- **Paired comparison:** mỗi seed cho cả R1 + R2 → chênh do CẤU HÌNH, KHÔNG do RNG
- 3 seeds = đủ rẻ + đủ informative (mean ± std có ý nghĩa, std error giảm theo 1/√n)

---

## 9 · Tầng 1 — Kết quả

| Run | Acc (mean ± std) | Macro-F1 | Match paper? |
|---|---|---|---|
| **T1-R1 (RGB)** | **98.52% ± 0.27%** | 98.47% | ✅ khớp 98.57% → pipeline đáng tin cậy |
| **T1-R2 (13 bands)** | **98.48% ± 0.07%** | 98.43% | std nhỏ hơn RGB (consistent hơn) |
| **Paired Δ (R2−R1)** | **−0.04% ± 0.26%** | −0.05% | — |

> 🚨 **13 bands KHÔNG vượt RGB** — chênh nằm trong nhiễu seed.
> Phát hiện âm-ish: thêm raw bands không đủ; **indices mới là đòn bẩy** → kiểm chứng ở **Tầng 2 H2**.

---

## 10 · Hard-class pattern (persistent)

4 lớp F1 thấp nhất giữ nguyên qua **cả 6 runs Tầng 1**:

> `PermanentCrop` · `HerbaceousVegetation` · `AnnualCrop` · `Pasture`

**Hard-vs-Easy gap ~2.0–2.3%** — overall acc đã gần ceiling, nhưng macro-F1 lớp khó còn dư địa.

🎯 **Target Tầng 2:** macro-F1 hard tăng **≥ 1.5–2.0%** (per `CLAUDE.md §3`)
🎯 Indices nhắm đúng nhóm này: NDVI cho vegetation, NDMI cho moisture, NDWI cho water-vs-land.

---

## 11 · Tầng 2 — 11 thí nghiệm (đã thiết kế, chưa chạy)

| Run | Câu hỏi | Ch | Seeds |
|---|---|---|---|
| **H2** | 13 bands + 4 indices > 13 bands? *(câu hỏi trung tâm)* | 17 | 42, 123, 2024 |
| H3a-d | NDVI / NDWI / NDBI / NDMI **riêng lẻ** đóng góp? | 14 | 42 |
| H4 | Indices có ích cả khi **không có 13 bands**? | RGB+4 = 7 | 42 |
| H5a/b/c | 10m đủ chưa? Thêm 20m? Atmospheric có là noise? | 4 / 10 / 7 | 42 |

**Roll-out 3 phase (gate giữa các phase):**
- **Phase 1** = H2 × 3 seeds (~2.5h) — chỉ run này có 3 seeds vì là câu hỏi chính
- **Phase 2** = H3a-d + H4 (~3.5h) — ablation indices
- **Phase 3** = H5a/b/c (~2.5h) — ablation band-group

---

## 12 · Trạng thái & roadmap

✅ **Đã xong**
- Scaffold (`src/{data,models,training,utils}`, `configs/`, `scripts/`)
- Loader (band-by-name, on-the-fly indices, train-only normalize)
- Splits + stats (commit-able trong `dataset/splits/`)
- Training pipeline + W&B integration
- **Tầng 1: 6 runs** → `paper/results_table.md`
- **Tầng 2: 9 YAML configs** verified

⏳ **Đang chờ:** quyết định khởi động **Phase 1 Tầng 2** (T2-H2 × 3 seeds)

🔜 **Tiếp theo:** Tầng 2 (3 phase) → Tầng 3 tuning (LR/WD/Aug/Optim) → winner cuối làm baseline cho **Research phase: RQ1 — Self-Supervised Learning** (8 slide tiếp: chốt → khái niệm → 5 slide phương pháp → kết nối Refine)

📊 W&B project `eurosat-refine` — đủ config/metrics/per-class F1/confusion matrix/artifacts. **Không xoá run thất bại** (là evidence cho paper).

---

## 13 · Research phase — RQ1 đã chốt

**Câu hỏi nghiên cứu (RQ1):**

> SSL pretrain trên Sentinel-2 **không nhãn** có giảm nhu cầu nhãn labeled không?
> So sánh fine-tune trên **1% / 5% / 10% / 100%** nhãn EuroSAT vs baseline ImageNet-pretrained.

**Tham chiếu paper gốc (Helber et al. 2019, §4.5 + Figure 7):**
- Acc giảm mạnh khi training set nhỏ — ở **10% train data chỉ đạt ~75%**
- Paper KHÔNG thử bất kỳ phương pháp giảm phụ thuộc nhãn nào
- → Khe hở **"low-data regime"** (CLAUDE.md §2 — đánh dấu *"Cao — dành cho Research"*)

**Vì sao chọn RQ1?**
- Đánh thẳng vào số liệu cụ thể của paper (75% @ 10%) — dễ định lượng cải thiện
- Sentinel-2 unlabeled data dồi dào, miễn phí (Copernicus / GEE / BigEarthNet)
- SSL là direction nóng 2022-2025, có model open-source (MAE, DINOv2, SatMAE, Prithvi) ready

---

## 14 · SSL khái niệm — MAE & DINOv2

**Self-Supervised Learning (SSL):** học representation từ data **không nhãn** bằng cách tự tạo "task giả" (pretext task) từ chính ảnh.

| Phương pháp | Pretext task | Strength |
|---|---|---|
| **MAE** (He et al. CVPR 2022) | Mask 75% patches → reconstruct pixels | Spatial structure, scaling tốt, đơn giản |
| **DINOv2** (Oquab et al. 2023) | Self-distillation teacher-student, không reconstruction | Semantic features, transfer linear-probe tốt |

**Biến thể cho remote sensing:**
- **SatMAE** (NeurIPS 2022): MAE + temporal/spectral encoding cho Sentinel-2
- **Prithvi** (NASA-IBM 2023): MAE trên HLS (Sentinel-2 + Landsat)

→ Sẽ dùng **MAE** làm baseline SSL (well-studied, codebase Meta sạch).
→ Tham khảo **SatMAE** nếu thời gian cho phép — kiểm chứng "domain-specific SSL có hơn generic SSL không?"

---

## 15 · Phương pháp RQ1 — Logic tổng thể

**Vấn đề:** với ít nhãn (1-10%), model học từ đâu để classify tốt?

**Trực giác — chia làm 2 việc:**

| Việc | Cần nhãn? | Học từ đâu |
|---|---|---|
| **Hiểu ảnh Sentinel-2** (textures, edges, spectral patterns) | ❌ không | Rất nhiều ảnh **không nhãn** (SSL) |
| **Map ảnh → 10 lớp EuroSAT** | ✅ có | Ít nhãn cũng được, vì phần "hiểu ảnh" đã sẵn |

→ Đây là cấu trúc **pretrain → fine-tune**.

**Khác baseline ImageNet ở đâu?**
Pretrain trên **đúng domain** (Sentinel-2 vệ tinh) thay vì ảnh tự nhiên (chó, mèo, xe).

→ Giả thuyết: representation từ Sentinel-2 sẽ transfer **gần hơn** đến task LULC.

---

## 16 · Bước 1 — Pretrain SSL với MAE

**MAE (Masked Autoencoder, He et al. CVPR 2022) làm gì?**

1. Chia ảnh 64×64 thành patches (8×8 = 64 patches)
2. **Random mask 75% patches** (giữ 16, ẩn 48)
3. Encoder (ViT) nhìn 16 patches → embedding
4. Decoder lấy embedding + position của patches bị ẩn → **dự đoán raw pixel** của 48 patches
5. Loss = MSE(pixel dự đoán, pixel thật)

> 💡 Để dự đoán patch bị ẩn (vd giữa cánh đồng), encoder **phải hiểu context** xung quanh — texture, màu, spectral signature. Đó chính là "hiểu ảnh".

| | Lựa chọn | Lý do |
|---|---|---|
| Data | **BigEarthNet** (~590k Sentinel-2 tile) | Ready, lớn, cùng sensor với EuroSAT |
| Backbone | ViT-Small (~22M) hoặc ResNet-50 | MAE designed for ViT; ResNet để match Refine |
| Mask ratio | 75% | Theo paper gốc — task đủ khó để học representation tốt |
| Epochs | 100-200 | SSL cần lâu hơn supervised |
| **KHÔNG dùng** | nhãn EuroSAT | Đây là điểm cốt lõi — pretrain phải **completely unsupervised** |

---

## 17 · Bước 2 — Fine-tune & stratified subset

**Cấu trúc:**
```
Encoder (đã pretrain MAE)  →  Linear head (random init, 10 lớp)
```

**Stratified subset — chìa khoá quan trọng**

Từ train 21,600 ảnh EuroSAT, tạo 4 subset bằng stratified sampling seed 42:

| Fraction | Tổng ảnh | ~Ảnh/lớp |
|---|---|---|
| 1% | 216 | 22 |
| 5% | 1,080 | 108 |
| 10% | 2,160 | 216 |
| 100% | 21,600 | 2,160 |

- **Stratified** = giữ tỷ lệ 10 lớp đều → tránh subset 1% bị lệch (vd 0 ảnh `River`)
- **Seed 42** = 4 subset luôn giống nhau → 3 điều kiện (A)(B)(C) fine-tune trên **CÙNG** subset → so fair

**Recipe fine-tune:** y hệt Refine (AdamW lr 1e-4, cosine, augment geometric, early stop) → **cô lập biến "init strategy"**.

**Val/test giữ NGUYÊN** (2700 + 2700) cho mọi fraction → thước đo nhất quán.

---

## 18 · Bước 3 — Ma trận thí nghiệm 3 × 4 = 12 runs

| Init | 1% | 5% | 10% | 100% |
|---|---|---|---|---|
| **(A) Scratch** (random) | A-1 | A-5 | A-10 | A-100 |
| **(B) ImageNet** (winner Refine) | B-1 | B-5 | B-10 | B-100 |
| **(C) SSL** (MAE của ta) | C-1 | C-5 | C-10 | C-100 |

**Mỗi cặp so sánh cô lập biến gì?**

| Cặp | Đo |
|---|---|
| C vs A | Pretrain (bất kỳ) có ích hơn random init? — sanity check |
| **C vs B** | Pretrain **domain-specific** có hơn generic ImageNet? — **câu hỏi chính RQ1** |
| (C−B) tại 1% vs (C−B) tại 100% | Lợi ích SSL có **lớn hơn ở low-data** không? — kỳ vọng cốt lõi |

**Vì sao cần cả 3 điều kiện?**
Nếu chỉ B vs C → không biết cải thiện đến từ "pretrain nói chung" hay "pretrain đúng domain". A là **sàn** để định nghĩa thế nào là "có ích".

---

## 19 · Diễn giải kết quả — 4 kịch bản có thể có

**Data-efficiency curve** (acc vs label %):

```
 acc                                ● B (ImageNet)
98 |                          ●    ● C (SSL)
   |               ●         ●
90 |          ●         ●                ← Δ lớn nhất ở đây
   |     ●         ●                       (giá trị thực của RQ1)
70 | ●         ●
   | A (Scratch)
   +-+----+----+----+----+----+→ label %
     1    5   10              100
```

**Mỗi kịch bản đều là phát hiện hợp lệ — Research phase sẽ giải thích VÌ SAO:**

| Kết quả quan sát | Kết luận |
|---|---|
| C-1, C-5 ≫ B-1, B-5 và C-100 ≈ B-100 | ✅ SSL "bù nhãn" mạnh ở low-data → **RQ1 thành công** |
| C ≈ B mọi điểm | ❌ ImageNet đã đủ — domain-specific không thêm gì |
| C > B mọi điểm (kể cả 100%) | ⚠️ Bất ngờ tốt — SSL chuẩn hơn cả cho high-data |
| C < B ở 100% nhưng C > B ở 1% | ⚠️ Trade-off có ý nghĩa: chọn theo budget nhãn thực tế |

---

## 20 · Expected outcome & kết nối Refine

**Kỳ vọng (giả định):**

| Label % | (B) ImageNet baseline | (C) SSL pretrain | Δ kỳ vọng |
|---|---|---|---|
| 1% | ~55-65% (cực thấp) | ~70-80% | **+10-15%** |
| 5% | ~70-75% | ~85-90% | **+10-15%** |
| 10% | ~75% *(số paper)* | ~85-90% | **+10%** |
| 100% | ~98.5% *(số Refine)* | ~98.5-99% | match hoặc nhỉnh hơn |

→ Giá trị nghiên cứu: SSL có khả năng **"bù nhãn"** — quan trọng cho real-world deployment (labeling đắt, chậm, cần chuyên gia).

**Kết nối với Refine (3 yếu tố tái dùng):**
1. **Training recipe** + splits cố định seed 42 → so sánh fair giữa (B) và (C)
2. **Baseline số liệu** từ Tầng 1 + winner Tầng 3 → biết ngưỡng cần vượt
3. **Cấu hình input** của winner Refine (bands + indices) → dùng cho cả 3 điều kiện → cô lập biến **"init strategy"**
4. **Hard-class pattern** từ Refine → diễn giải SSL có cải thiện đúng nhóm khó không

---

## 21 · Q&A

**Tài liệu**
- [CLAUDE.md](../CLAUDE.md) — decisions log, band order, full training recipe, §11 chi tiết RQ1
- [paper/results_table.md](results_table.md) — bảng kết quả Tầng 1 đầy đủ
- [configs/](../configs/) — 11 YAML (Tầng 1: 2, Tầng 2: 9)
- W&B project: `eurosat-refine`

**Câu hỏi mở để thảo luận:**
1. Có nên thêm seed thứ 4 cho H2 nếu paired Δ vẫn trong nhiễu sau Phase 1?
2. Nếu H4 (RGB+indices) **vượt** H2 (13bands+indices) → giả thuyết về **lợi ích đến từ indices, không phải bands** có hợp lý?
3. Tầng 3 nên ưu tiên tune **augmentation** (MixUp/CutMix) hay **LR schedule**?
4. **RQ1:** pretrain SSL nên dùng **BigEarthNet** (ready, lớn, nhưng có overlap địa lý với EuroSAT) hay **tự gom tile Copernicus ngoài châu Âu** (sạch hơn nhưng tốn công)?
