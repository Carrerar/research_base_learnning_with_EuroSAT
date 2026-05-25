---
marp: true
theme: default
paginate: true
size: 16:9
header: "Multi-Spectral Fusion & SSL on EuroSAT — DPL302M"
footer: "2026-05-26 · Refine: Tier 1 ✅  Tier 2 ✅  Tier 3 ✅  ·  Winner: T1-R2 13b raw + AdamW (98.85%)  ·  Research: RQ1-A (ResNet-50 + contrastive SSL)"
---

# Hợp nhất đa phổ và học tự giám sát hiệu quả nhãn

### cho phân loại sử dụng đất Sentinel-2 trên bộ dữ liệu EuroSAT

---

*Multi-Spectral Fusion and Label-Efficient Self-Supervised Learning*
*for Sentinel-2 Land Use Classification on EuroSAT*

---

**DPL302M — Research-Based Learning Project**
Phase **Refine** (3/5): Tầng 1 ✅ · Tầng 2 ✅ · Tầng 3 ✅   →   Phase **Research**: RQ1-A (ResNet-50 + contrastive SSL) → RQ1-B optional (MAE)
**Winner Refine:** T1-R2 13 bands raw + AdamW recipe — mean 98.85% (n=3)
**Protocol freeze:** split 80/10/10 seed 42 — bất biến Refine → Research → Report
2026-05-26

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

💡 Sau bug fix conv1 RGB-prior (xem slide 9bis): **13 bands raw VƯỢT RGB +0.33pp** (đảo finding cũ), và **indices KHÔNG còn cải thiện thêm**. Winner Tier 2 thực sự là **T1-R2 13b alone**.

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

## 9 · Tầng 1 — Kết quả (SAU bug fix conv1)

| Run | Acc (mean ± std) | Macro-F1 | Hard F1 | Match paper? |
|---|---|---|---|---|
| **T1-R1 (RGB, 3ch)** | **98.52% ± 0.22%** | 98.47% | 97.26% | ✅ khớp 98.57% → pipeline tin cậy |
| **T1-R2 (13 bands, FIXED)** | **98.85% ± 0.24%** | 98.80% | **97.93%** | — |
| **Paired Δ (R2−R1)** | **+0.33pp** | +0.33pp | **+0.68pp** | — |

> ✅ **13 bands VƯỢT RGB +0.33pp** (paired) sau khi fix bug conv1 RGB-prior alignment (slide 9bis). Hard F1 cải thiện đáng kể **+0.68pp** — bands IR/SWIR/red-edge giúp 4 lớp Hard.
> Đảo finding cũ (trước fix: paired Δ −0.04pp, "13 bands không vượt RGB" — đó là **artifact của bug**).

---

## 9bis · 🐛 Bug critical: conv1 RGB-prior misalignment

**Phát hiện 2026-05-24** (skeptical audit khi user hỏi "có khả năng tính sai?"):

```python
# Code cũ — copy RGB ImageNet prior MÙ vào 3 channel đầu của input:
new_conv1.weight[:, :3] = old_conv1.weight[:, :3]
# Với bands: all = [B01, B02, B03, B04, ...] thì:
#   B01 (Aerosol) ← Red prior     ❌ SAI band
#   B02 (Blue)    ← Green prior   ❌
#   B03 (Green)   ← Blue prior    ❌
#   B04 (Red THỰC) ← KHÔNG có prior ❌ (mất bands quan trọng nhất)
```

**Fix:** API `build_resnet50(bands=...)` tự compute `rgb_positions` từ bands list — map ImageNet R/G/B vào đúng vị trí B04/B03/B02.

**Impact đo trên T1-R2 seed 42:** +0.55pp acc, **+1.13pp Hard F1**. Bug MAJOR → re-run toàn bộ 13-band experiments.

> 🎓 **Bài học methodological cho paper Discussion:** Mọi N-channel adapter kế thừa ImageNet conv1 phải verify weight alignment PROGRAMMATIC, không chỉ rely "caller responsibility" docstring. Đã thêm bộ quy tắc bắt buộc vào `CLAUDE.md §13`.

---

## 10 · Hard-class pattern (persistent)

4 lớp F1 thấp nhất giữ nguyên qua **cả 6 runs Tầng 1** + mọi run Tầng 2:

> `PermanentCrop` · `HerbaceousVegetation` · `AnnualCrop` · `Pasture`

**Gap Hard-vs-Easy:**

| Cấu hình | Hard F1 | Easy F1 | Gap |
|---|---|---|---|
| T1-R1 RGB | 97.26% | 99.28% | 2.03pp |
| **T1-R2 13b fixed** | **97.93%** | 99.38% | **1.44pp** ⬇ |

→ 13 bands thu hẹp gap **0.59pp** (chủ yếu cải thiện PermanentCrop +1.63pp, HerbaceousVegetation +0.88pp).

🎯 **Target Tầng 2:** macro-F1 hard tăng **≥ 1.5pp vs T1-R1** (must-have CLAUDE.md §3)
🎯 Indices nhắm đúng nhóm này: NDVI/NDMI cho vegetation, NDWI cho water — **liệu chúng có thêm value gì sau khi 13 bands raw đã giúp?**

---

## 11 · Tầng 2 — Kết quả (SAU bug fix)

| Run | Input | Ch | n | Acc mean | Hard F1 | Paired Δ vs T1-R2 |
|---|---|---|---|---|---|---|
| **T1-R2** | 13b raw | 13 | 3 | **98.85%** | **97.93%** | — (baseline) |
| H3c | 13b + NDBI | 14 | 4 | 98.80% | 97.77% | **−0.17pp** acc, **−0.32pp** Hard F1 |
| H2 | 13b + 4 indices | 17 | 3 | 98.62% | 97.52% | **−0.23pp** acc, **−0.41pp** Hard F1 |
| H3d | 13b + NDMI | 14 | 1 | 98.74% | 97.66% | (seed 42: −0.22pp) |
| H3b | 13b + NDWI | 14 | 1 | 98.37% | 96.98% | (seed 42: −0.59pp) |
| H3a | 13b + NDVI | 14 | 1 | 98.33% | 96.77% | (seed 42: −0.63pp) |
| H4 | RGB + 4 indices | 7 | 1 | 98.48% | 97.23% | — |
| **H5 (new)** | **4 indices ONLY** | **4** | 1 | **97.44%** | **95.44%** | — |

> 🚨 **Đảo finding cũ "NDBI là winner"** — sau fix, NO cấu hình indices nào vượt T1-R2 alone có ý nghĩa thống kê. Mọi paired Δ trong noise band (SE ±0.27-0.32pp).

---

## 11bis · Insights từ Tầng 2

**Finding 1 — Indices KHÔNG cung cấp thêm thông tin** khi đã có 13 raw bands + conv1 đúng prior:
- H3c (NDBI) − T1-R2 = −0.17pp (n=3); H2 (4 indices) − T1-R2 = −0.23pp (n=3)
- Mọi index riêng lẻ trên seed 42 đều ≤ T1-R2 fixed
- Mechanism (giả thuyết): ResNet-50 đủ capacity học implicit NDVI/NDBI từ B04/B08/B11 raw → explicit indices = redundant

**Finding 2 — Indices alone TỆ HƠN cả RGB alone** (T2-H5, direct test):
- 4 indices only: **97.44%** vs RGB only 98.52% → **−1.08pp acc**
- Hard F1: 95.44% vs RGB 97.26% → **−1.82pp** (drop mạnh ở chính 4 lớp vegetation NDVI được thiết kế cho!)
- Mechanism: indices = lossy compression — collapse 2D spectral về 1D ratio, mất fine spectral signature

**Finding 3 — Bug conv1 từng tạo finding khoa học sai:**
- Trước fix: ranking H3c > T1-R2 > T1-R1 (NDBI "thắng")
- Sau fix: ranking T1-R2 > H3c > T1-R1 (T1-R2 alone thắng) — **đảo**
- Internal consistency (paired Δ, std, ranking) đều ổn định ở cả 2 era → bug khó phát hiện

---

---

## 12 · Trạng thái & roadmap

✅ **Đã xong toàn bộ Refine (3/3 tầng)**

- **Tầng 1 (6 runs buggy + 3 runs fixed):** T1-R1 RGB 98.52%, T1-R2 fixed 98.85%
- **Tầng 2 (13 fixed runs):** T1-R2 alone winner; H3c/H2/H3a/b/d/H4/H5 indices đều không vượt
- **Tầng 3 (10 runs):** T3-C aug + T3-A/B/D sweeps + T3-E SGD 5-seed final = 98.73% — **vẫn dưới baseline**
- **Bug conv1 RGB-prior** discovered & fixed 2026-05-24 → re-run toàn bộ 13-band experiments
- Quy tắc phòng ngừa lỗi (CLAUDE.md §13) + post-mortem documented

🏆 **Winner Refine cuối cùng: T1-R2 13 bands raw + AdamW default recipe** (mean 98.85% ± 0.24%, Hard F1 97.93%)

🔜 **Tiếp theo:** Research phase RQ1 — SSL pretrain Sentinel-2 unlabeled, fine-tune 1/5/10/100% EuroSAT label fractions. T1-R2 fixed = baseline ImageNet condition (B).

📊 W&B project `eurosat-refine` — ~35 runs đủ config/metrics/per-class F1/confusion matrix/artifacts. **Không xoá run thất bại** (evidence cho paper Discussion).

---

## 12bis · Tầng 3 — Kết quả & verdict

**Setup:** Base = T1-R2 13b fixed, seed 42 cho sweep, 5 seeds cho final.

| Sweep | Variants | Winner single seed | Δ acc vs baseline 98.96% |
|---|---|---|---|
| T3-C aug | flip+rot (baseline) / RRC / MixUp α=0.2 / CutMix α=1.0 | **CutMix** 99.15% | +0.19pp marginal |
| T3-A LR | 3e-5 / 1e-4 (baseline) / 3e-4 / **1e-3** | **lr=1e-3** 99.19% | +0.23pp marginal |
| T3-B WD | 1e-5 / 1e-4 (baseline) / 5e-4 | baseline tie | 0.00pp |
| T3-D optim | AdamW (baseline) / **SGD+Nesterov** | SGD 99.07% | +0.11pp |
| **T3-E** final 5 seeds | SGD+Nesterov lr=0.01 | mean 98.73% ± 0.28% | **−0.12pp** ❌ |

🚨 **T3-D s42 lucky outlier:** single seed 99.07% → predict winner → 5-seed mean 98.73% (thua baseline AdamW 98.85%).

| Metric | T1-R2 baseline (AdamW, n=3) | T3-E SGD (n=5) | Δ |
|---|---|---|---|
| Acc mean | **98.85%** | 98.73% | −0.12pp |
| Hard F1 | **97.93%** | 97.77% | −0.16pp |

**Must-have Hard F1 ≥ T1-R1 + 1.5pp (98.76%)** → **KHÔNG đạt** với cả baseline (97.93%) và T3-E (97.77%).

> 📌 **3 bài học methodological từ Tier 3:**
> 1. **Single-seed ranking không đáng tin** — sweep dùng 1 seed CHỈ để thu hẹp search space; commit phải multi-seed
> 2. **Seed effect lớn cho lớp Hard** — std Hard F1 (0.53pp) gấp đôi std acc (0.28pp)
> 3. **Aug + optim đều không phải đòn bẩy** — gap Hard-Easy còn lại là fundamental (label noise / 64×64 resolution)

---

## 13 · Research phase — RQ1 (protocol freeze 2026-05-26)

**Câu hỏi nghiên cứu (RQ1):**

> SSL pretrain trên Sentinel-2 **không nhãn** có giảm nhu cầu nhãn labeled không?
> So sánh fine-tune trên **1% / 5% / 10% / 100%** nhãn EuroSAT vs baseline ImageNet-pretrained.

**Tham chiếu paper gốc (Helber et al. 2019, §4.5 + Figure 7):**
- Acc giảm mạnh khi training set nhỏ — ở **10% train data chỉ đạt ~75%**
- Paper KHÔNG thử bất kỳ phương pháp giảm phụ thuộc nhãn nào → khe hở **"low-data regime"**

🔒 **Protocol freeze (CLAUDE.md §11):**
- Giữ NGUYÊN split **80/10/10 seed 42** từ Refine — KHÔNG chuyển sang 10/90 paper gốc
- Chỉ subsample **TRAIN** cho low-label; **val + test BẤT BIẾN** mọi experiment
- Reproduce **CONCEPT** "low-data regime", KHÔNG reproduce literal split ratio

**RQ1 chia 2 sub-phase để giảm confounders:**

| | RQ1-A (CORE, bắt buộc) | RQ1-B (OPTIONAL, sau) |
|---|---|---|
| Backbone | **ResNet-50** (giống Refine) | ViT-Small / MAE |
| SSL method | Contrastive (SimCLR / MoCo-v2 / BYOL) | MAE / SatMAE / DINOv2 / Prithvi |
| Biến isolate | Representation learning effect | Architecture × SSL synergy |
| Khi chạy | Trước, bắt buộc | Sau RQ1-A xong + còn compute |

→ Nhảy thẳng MAE = đổi đồng thời 3 biến (backbone + objective + tokenization) → không attribute được gain cho SSL hay architecture.

---

## 14 · SSL khái niệm — Contrastive (RQ1-A) vs Reconstruction (RQ1-B)

**Self-Supervised Learning (SSL):** học representation từ data **không nhãn** bằng cách tự tạo "task giả" (pretext task) từ chính ảnh.

| Phương pháp | Pretext task | Architecture tự nhiên | Sub-phase |
|---|---|---|---|
| **SimCLR** (Chen 2020) | 2 augmented views cùng ảnh → kéo gần / đẩy xa | ResNet | **RQ1-A** ✅ |
| **MoCo-v2** (He 2020) | Contrastive + momentum encoder + queue | ResNet | **RQ1-A** ✅ |
| **BYOL** (Grill 2020) | Self-distillation, KHÔNG negatives | ResNet | **RQ1-A** ✅ |
| **MAE** (He CVPR 2022) | Mask 75% patches → reconstruct | ViT (patch tokens) | RQ1-B (sau) |
| **DINOv2** (Oquab 2023) | Self-distillation teacher-student | ViT | RQ1-B (sau) |
| **SatMAE / Prithvi** | MAE + spectral/temporal encoding | ViT | RQ1-B (sau) |

**Vì sao RQ1-A bắt đầu với contrastive (không phải MAE)?**
- ResNet-50 backbone **giống Refine** → so sánh fair với baseline ImageNet
- Đổi 1 biến tại 1 thời điểm (chỉ pretrain method), không đổi cả architecture
- Engineering complexity thấp, codebase trưởng thành (lightly, lightly-ssl, vissl)

**Vì sao MAE đáng đầu tư (nhưng sau)?**
- Sentinel-2 có strong spatial + spectral redundancy → reconstruction phù hợp
- Contrastive ở remote sensing khó vì color jitter không physically valid
- MAE ít phụ thuộc augmentation mạnh → robust hơn cho multispectral

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

## 16 · Bước 1 — Pretrain SSL (RQ1-A: contrastive)

**Contrastive SSL làm gì?** Từ 1 ảnh không nhãn → tạo 2 "view" augment khác nhau → encoder map cả 2 về embedding → **kéo gần** 2 view CÙNG ảnh, **đẩy xa** view của ảnh khác.

```
Ảnh I  ──aug₁──→ x₁ ──encoder──→ z₁ ┐
       └─aug₂──→ x₂ ──encoder──→ z₂ ┘ kéo gần (positive)
Ảnh J  ──aug──→ x_j ─encoder──→ z_j   đẩy xa (negative)

Loss (InfoNCE):  −log[ sim(z₁,z₂) / Σ sim(z₁, z_k) ]
```

> 💡 Để 2 view augment của cùng ảnh map về cùng embedding, encoder phải học **invariant features** (chống xoay, lật, crop) — đó chính là "hiểu ảnh".

| | Lựa chọn | Lý do |
|---|---|---|
| Data | **BigEarthNet** (~590k Sentinel-2 tile) | Ready, lớn, cùng sensor với EuroSAT |
| **Backbone** | **ResNet-50 (cố định)** | Giống Refine → isolate "SSL effect" độc lập |
| SSL method | **SimCLR** baseline (đơn giản) — fallback MoCo-v2 nếu OOM | Codebase mature, dễ debug |
| Augmentation | Flip + rot + Gaussian blur — **KHÔNG color jitter** | Color jitter phá spectral relations (CLAUDE.md §5) |
| Batch size | 256-512 (cần lớn cho contrastive) | InfoNCE cần đủ negatives |
| Epochs | 100-200 | SSL cần lâu hơn supervised |
| **KHÔNG dùng** | nhãn EuroSAT | Pretrain phải **completely unsupervised** |

**Adapter conv1:** BigEarthNet 12 bands (thiếu B10), EuroSAT 13 bands → init weight cho B10 bằng `mean_w * 0.5` (pattern giống bug fix conv1) + apply rule §13 A1/B1 (enforce ở boundary, weight-alignment test).

---

## 17 · Bước 2 — Fine-tune & stratified subset (protocol freeze)

**Cấu trúc:**
```text
Encoder (đã pretrain SSL — ResNet-50)  →  Linear head (random init, 10 lớp)
```

**Stratified subset — chìa khoá quan trọng (nested)**

Từ TRAIN 21,600 ảnh EuroSAT (val + test BẤT BIẾN), tạo 4 nested subset bằng stratified seed 42:

| Fraction | Tổng ảnh | ~Ảnh/lớp | Nested |
|---|---|---|---|
| 1% | 216 | 22 | ⊂ 5% |
| 5% | 1,080 | 108 | ⊂ 10% |
| 10% | 2,160 | 216 | ⊂ 100% |
| 100% | 21,600 | 2,160 | full |

- **Stratified** = giữ tỷ lệ 10 lớp đều → tránh subset 1% bị lệch (vd 0 ảnh `River`)
- **Nested** = mọi sample ở 1% phải có ở 5% / 10% / 100% → khử confounder "subset khác nhau"
- **Seed 42** = 4 subset luôn giống nhau → 3 điều kiện (A)(B)(C) fine-tune trên **CÙNG** subset → so fair

🔒 **Protocol freeze (CLAUDE.md §11 Guard rails):**
- NEVER regenerate global splits
- NEVER chuyển 10/90
- ONLY subsample TRAIN (val + test = 2700 + 2700 bất biến mọi fraction)
- Treat **T1-R2 fixed = official supervised baseline** (condition B đã frozen)

**Recipe fine-tune:** y hệt Refine (AdamW lr 1e-4, cosine, augment geometric, early stop) → **cô lập biến "init strategy"**.

---

## 18 · Bước 3 — Ma trận RQ1-A: 3 × 4 = 12 runs (backbone CỐ ĐỊNH = ResNet-50)

| Init | 1% (216) | 5% (1080) | 10% (2160) | 100% (21600) |
|---|---|---|---|---|
| **(A) Scratch** (random ResNet-50) | A-1 | A-5 | A-10 | A-100 |
| **(B) ImageNet** = winner Refine T1-R2 fixed | B-1 | B-5 | B-10 | B-100 |
| **(C) SSL** contrastive ResNet-50 (BigEarthNet) | C-1 | C-5 | C-10 | C-100 |

**Mỗi cặp so sánh cô lập biến gì?**

| Cặp | Đo |
|---|---|
| C vs A | Pretrain (bất kỳ) có ích hơn random init? — sanity check |
| **C vs B** | Pretrain **domain-specific Sentinel-2** có hơn generic ImageNet? — **câu hỏi chính RQ1-A** |
| (C−B) tại 1% vs (C−B) tại 100% | Lợi ích SSL có **lớn hơn ở low-data** không? — kỳ vọng cốt lõi |

**Vì sao backbone CỐ ĐỊNH ResNet-50?**
- Refine winner cũng ResNet-50 → so sánh fair, condition B = winner Refine (số liệu frozen)
- Đổi 1 biến tại 1 thời điểm: chỉ **init strategy** (Scratch / ImageNet / SSL), KHÔNG đổi architecture
- Nếu sau RQ1-A xong + còn compute → mở RQ1-B (ViT + MAE) để test architecture × SSL synergy

**Vì sao cần cả 3 điều kiện?**
Nếu chỉ B vs C → không biết cải thiện đến từ "pretrain nói chung" hay "pretrain đúng domain". A là **sàn** để định nghĩa thế nào là "có ích".

---

## 18bis · 4 bước trước khi launch SSL nặng (de-risk)

Thay vì nhảy thẳng vào 100-200 epoch SSL pretrain, đi qua 4 checkpoint kiểm soát rủi ro:

| Bước | Việc | Mục tiêu | Output |
|---|---|---|---|
| **1. Protocol freeze** | Chốt splits / metrics / recipe / seed policy | Không tune supervised thêm | CLAUDE.md §5+§11 (đã xong 2026-05-26) |
| **2. Condition B low-label curve** | Chạy winner Refine (T1-R2 fixed) trên 1/5/10/100% | Có **supervised degradation curve** ImageNet làm ngưỡng SSL phải vượt | 4 run, ~6h GPU |
| **3. Verify low-label stability** | Subset generation đúng stratification, variance qua seeds, train stability ở fraction nhỏ | Tránh false signal do subset bị skew | Notebook + 1-2 seed extra |
| **4. Tiny SSL pilot** | Chạy SSL nhỏ (epoch ít, batch nhỏ) | Verify: loss giảm, embeddings non-trivial, fine-tune work, pipeline correct, memory stable | 1 short SSL run + 1 fine-tune |

🚫 **KHÔNG launch full 100-200 epoch SSL pretrain TRƯỚC bước 4.**

> 💡 Bài học rút từ Refine: bug conv1 sống 2 ngày vì không có "smoke test" before launch run dài (CLAUDE.md §13 D1). RQ1 áp dụng nguyên tắc này ở scale lớn hơn — 4 checkpoint thay vì 1.

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

## 20 · Expected outcome & triết lý chuyển dịch

**Kỳ vọng (giả định) — focus low-label, KHÔNG focus 100%:**

| Label % | (B) ImageNet baseline | (C) SSL pretrain | Δ kỳ vọng | Tầm quan trọng |
|---|---|---|---|---|
| **1%** | ~55-65% | ~70-80% | **+10-15%** | 🎯 cốt lõi |
| **5%** | ~70-75% | ~85-90% | **+10-15%** | 🎯 cốt lõi |
| **10%** | ~75% *(số paper)* | ~85-90% | **+10%** | 🎯 cốt lõi |
| 100% | ~98.85% *(winner Refine)* | ~98.5-99% | match hoặc nhỉnh | thứ yếu |

→ Giá trị nghiên cứu: SSL **"bù nhãn"** — quan trọng cho real-world deployment (labeling đắt, chậm, cần chuyên gia).

🔄 **Triết lý chuyển dịch (Research vs Refine):**

| | Refine | Research |
|---|---|---|
| Mục tiêu | Maximize supervised accuracy | Understand data efficiency |
| Câu hỏi | "Multispectral + indices có giúp không?" | "Bao nhiêu label là đủ?" |
| Metric quan trọng | +0.1pp ở 100% data | Khoảng cách (A)/(B)/(C) ở 1-10% |
| Triết lý | Benchmark optimization | Representation learning analysis |

**Kết nối với Refine (4 yếu tố tái dùng):**
1. **Training recipe** + splits seed 42 (FROZEN) → so sánh fair giữa (B) và (C)
2. **Condition B = winner Refine T1-R2 fixed** → biết ngưỡng cần vượt (số liệu frozen, không re-tune)
3. **Cấu hình input 13 bands raw** (không indices, confirmed lossy) → dùng cho cả 3 điều kiện → cô lập biến **"init strategy"**
4. **Hard-class pattern** từ Refine → diễn giải SSL có cải thiện đúng nhóm khó không
5. **Rule §13 bộ quy tắc phòng ngừa lỗi** → mọi adapter SSL mới (12→13 bands) phải weight-alignment test programmatic

---

## 21 · Q&A

**Tài liệu**
- [CLAUDE.md](../CLAUDE.md) — decisions log, band order, training recipe, **§13 bộ quy tắc phòng ngừa lỗi**, §11 RQ1
- [paper/results_table.md](results_table.md) — bảng kết quả đầy đủ (Tầng 1, 2 + bug history)
- [configs/](../configs/) — 17 YAML (Tier 1, Tier 2 buggy+fixed, Tier 3 aug sweep)
- W&B project: `eurosat-refine` (~30 runs)

**Câu hỏi mở để thảo luận:**
1. **Bug discovery:** Bug conv1 RGB-prior sống 2 ngày dù mọi internal consistency metric ổn định. Quy trình nào (rule §13) lẽ ra catch sớm hơn? Cho remote sensing community, nên audit conv1 init thế nào?
2. **Indices null finding:** Với n=3 seeds, paired Δ trong noise band — đủ để claim "indices không help" hay cần n=5-10? Trade-off GPU cost vs statistical power.
3. **Protocol freeze:** Quyết định giữ 80/10/10 thay vì literal 10/90 paper gốc — có làm comparison với paper Helber 2019 yếu đi không? Hay reproduce "low-data CONCEPT" đủ giá trị?
4. **RQ1-A vs RQ1-B trade-off:** Bắt đầu contrastive (SimCLR/MoCo) trên ResNet-50 để isolate biến — nhưng có rủi ro "miss" upside của MAE/SatMAE cho multispectral. Cần chạy RQ1-B song song hay tuần tự?
5. **SSL data source:** Pretrain dùng **BigEarthNet** (ready, có overlap địa lý EuroSAT — risk train-test leakage qua representation?) hay **tự gom tile Copernicus ngoài châu Âu** (sạch hơn nhưng tốn công)?
6. **Methodological:** Finding "ResNet học implicit indices" hiện là *giả thuyết* — đáng đầu tư probe activations để verify mechanism cho paper Discussion?
