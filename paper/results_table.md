# Kết quả thực nghiệm — EuroSAT Refine

> Bảng tổng hợp kết quả mọi tầng. Số liệu trong file này phản ánh kết quả **SAU FIX bug conv1 RGB-prior** (2026-05-24). Các run BUGGY còn lưu trong W&B nhưng KHÔNG dùng để rút kết luận khoa học — chỉ giữ ở §7 cho transparency.

**Tóm tắt 1 câu:** Sau khi fix bug conv1 RGB-prior alignment, **13 spectral bands raw (T1-R2 fixed) cho kết quả tốt nhất** (mean 98.85%); **không có cấu hình spectral indices nào (H3a/b/c/d/H2) vượt T1-R2 alone có ý nghĩa thống kê** — finding cũ "NDBI là winner" là artifact của bug. T2-H5 "indices alone" (97.44%, Hard F1 95.44%) cho thấy indices = lossy compression. **Tier 3 tuning (T3-A/B/C/D/E SGD 5 seeds = 98.73%) KHÔNG vượt được baseline T1-R2 AdamW** — must-have Hard F1 ≥ +1.5pp KHÔNG đạt (gap −0.83pp). **Winner Refine cuối = T1-R2 13b fixed + AdamW default recipe.**

---

## 1. Tầng 1 — Baseline Establishment

### 1.1 T1-R1: RGB baseline (ResNet-50, 3 channels)

Run này **KHÔNG bị bug** (`in_channels==3` không gọi adapter conv1). Số liệu giữ nguyên từ lần chạy đầu.

| Seed | Test Acc | Macro-F1 | Best Val Acc | W&B Run id |
|------|----------|----------|--------------|------------|
| 42   | 98.30%   | 98.23%   | 98.56%       | `a2yhj02w` |
| 123  | 98.81%   | 98.77%   | 98.96%       | `dqnchwqu` |
| 2024 | 98.44%   | 98.42%   | 98.96%       | `ixzj3j7h` |
| **Mean ± Std** | **98.52% ± 0.22%** | **98.47% ± 0.23%** | — | — |

**Match paper gốc:** 98.52% so với Helber 2019 báo cáo 98.57% — sai số 0.05%. Pipeline tin cậy.

### 1.2 T1-R1 — Per-class F1 (3 seeds)

| Class | seed 42 | seed 123 | seed 2024 | Mean | Group |
|---|---|---|---|---|---|
| PermanentCrop        | 0.9571 | 0.9535 | 0.9676 | **0.9594** | Hard |
| HerbaceousVegetation | 0.9768 | 0.9785 | 0.9735 | 0.9763 | Hard |
| AnnualCrop           | 0.9734 | 0.9800 | 0.9799 | 0.9778 | Hard |
| Pasture              | 0.9679 | 0.9824 | 0.9800 | 0.9768 | Hard |
| Industrial           | 0.9861 | 0.9901 | 0.9859 | 0.9874 | Easy |
| Highway              | 0.9879 | 0.9980 | 0.9901 | 0.9920 | Easy |
| River                | 0.9900 | 0.9980 | 0.9900 | 0.9927 | Easy |
| SeaLake              | 0.9917 | 1.0000 | 0.9900 | 0.9939 | Easy |
| Residential          | 0.9950 | 0.9983 | 0.9884 | 0.9939 | Easy |
| Forest               | 0.9967 | 0.9983 | 0.9967 | **0.9972** | Easy |

| Nhóm | Mean F1 |
|---|---|
| Hard (4 lớp) | **0.9726** |
| Easy (6 lớp) | 0.9928 |
| **Gap** | **2.03pp** |

**Pattern lớp khó:** PermanentCrop / HerbaceousVegetation / AnnualCrop / Pasture luôn là 4 lớp F1 thấp nhất ở mọi seed; PermanentCrop thấp nhất ở mọi seed. Phù hợp paper gốc Helber 2019.

### 1.3 T1-R2: 13 bands baseline FIXED (ResNet-50, 13 channels)

Run với conv1 RGB-prior **đã fix** — B04 (Red) nhận ImageNet Red weight, B03 (Green) nhận Green, B02 (Blue) nhận Blue. Các band còn lại nhận `mean_w * 0.5` init.

| Seed | Test Acc | Macro-F1 | Best Val Acc | W&B Run id |
|------|----------|----------|--------------|------------|
| 42   | 98.96%   | 98.93%   | 99.30%       | `tgasat2z` |
| 123  | 98.52%   | 98.44%   | 99.15%       | `ol1knd07` |
| 2024 | 99.07%   | 99.02%   | 99.41%       | `ijl5mydn` |
| **Mean ± Std** | **98.85% ± 0.24%** | **98.80% ± 0.25%** | — | — |

> **Std 0.24%** — tương đương T1-R1 (0.22%). Training stable. Seed 123 thấp nhất 98.52% (cùng giá trị với T1-R1 seed 42 = 98.30 → +0.22pp) — không phải outlier mạnh.

### 1.4 T1-R2 — Per-class F1 (3 seeds)

| Class | seed 42 | seed 123 | seed 2024 | Mean | Group |
|---|---|---|---|---|---|
| PermanentCrop        | 0.9757 | 0.9673 | 0.9839 | **0.9757** | Hard |
| Pasture              | 0.9848 | 0.9694 | 0.9749 | 0.9764 | Hard |
| AnnualCrop           | 0.9818 | 0.9770 | 0.9818 | 0.9802 | Hard |
| HerbaceousVegetation | 0.9851 | 0.9802 | 0.9900 | 0.9851 | Hard |
| Industrial           | 0.9859 | 0.9860 | 0.9899 | 0.9873 | Easy |
| Highway              | 0.9940 | 0.9880 | 0.9940 | 0.9920 | Easy |
| River                | 0.9960 | 0.9880 | 0.9960 | 0.9933 | Easy |
| Residential          | 0.9917 | 0.9934 | 0.9950 | 0.9934 | Easy |
| Forest               | 0.9983 | 0.9983 | 0.9967 | 0.9978 | Easy |
| SeaLake              | 1.0000 | 0.9967 | 1.0000 | **0.9989** | Easy |

| Nhóm | Mean F1 |
|---|---|
| Hard (4 lớp) | **0.9793** |
| Easy (6 lớp) | 0.9938 |
| **Gap** | **1.44pp** |

**So sánh per-class với T1-R1 RGB:**
- PermanentCrop +1.63pp (0.9594 → 0.9757) — cải thiện lớn nhất
- HerbaceousVegetation +0.88pp
- AnnualCrop +0.24pp
- Pasture −0.04pp (đứng yên)
- Mọi lớp Easy đứng yên hoặc tăng nhẹ (mean +0.10pp)

Pattern: bands ngoài RGB (chủ yếu NIR B08, SWIR B11/B12, red-edge B05-07) chủ yếu giúp **PermanentCrop và HerbaceousVegetation** — hai lớp vegetation có spectral signature đặc biệt (cây lâu năm vs cỏ tự nhiên), khó phân biệt chỉ bằng RGB.

---

## 2. So sánh T1-R1 vs T1-R2 fixed (paired comparison)

Vì cả hai dùng cùng bộ seed (42, 123, 2024), so từng cặp — khác biệt còn lại chỉ do **input**.

### 2.1 Tổng quan

| Metric | T1-R1 (RGB, 3ch) | T1-R2 (13b, 13ch) | Paired Δ (R2 − R1) |
|---|---|---|---|
| Test acc | 98.52% ± 0.22% | 98.85% ± 0.24% | **+0.33pp** |
| Macro-F1 | 98.47% ± 0.23% | 98.80% ± 0.25% | **+0.33pp** |
| F1 Hard mean | 0.9726 | **0.9793** | **+0.67pp** |
| F1 Easy mean | 0.9928 | 0.9938 | +0.10pp |
| Gap Hard-Easy | 2.03pp | **1.44pp** | **−0.59pp** |

### 2.2 Paired Δ per seed

| Seed | Δ Test Acc | Δ Macro-F1 | Δ Hard F1 |
|---|---|---|---|
| 42   | **+0.67pp** | +0.70pp | **+1.31pp** |
| 123  | **−0.30pp** | −0.33pp | −0.01pp |
| 2024 | **+0.63pp** | +0.60pp | +0.74pp |
| **Mean** | **+0.33pp** | **+0.32pp** | **+0.68pp** |

**Std paired Δ ≈ 0.55pp** → SE = 0.55/√3 = **0.32pp** → mean Δ +0.33pp **vừa đúng ngưỡng** signal vs noise. Marginal significance (không strong).

### 2.3 Diễn giải

**13 bands raw VƯỢT RGB +0.33pp** (paired, n=3). Đây là **đảo finding cũ** (trước fix: −0.04pp). Lý do đảo: bug conv1 cũ làm B04 (Red) không nhận ImageNet Red prior → model 13b mất ưu thế chính của bands chứa thông tin tương tự RGB; sau fix, B04 đúng prior + thêm 10 band IR/SWIR/red-edge → đầy đủ thông tin spectral.

**Điều cần lưu ý về significance:**
- 2/3 seeds Δ rõ ràng dương (+0.67, +0.63), 1 seed âm (−0.30).
- SE ≈ 0.32pp, mean +0.33pp → **borderline significant** (~95% CI lower bound gần 0).
- **Honest claim:** 13b CÓ XU HƯỚNG hơn RGB, nhưng không phải kết quả rất chắc chắn với n=3.

**Hard F1 +0.68pp** là số ý nghĩa hơn — direction nhất quán (2/3 dương rõ, seed 123 ≈ 0). Đây cho thấy IR/SWIR bands chủ yếu giúp 4 lớp Hard, đặc biệt PermanentCrop.

**Pattern lớp khó vẫn giữ:** 4 lớp Hard luôn là F1 thấp nhất ở cả 6 run.

---

## 3. Tổng kết Tầng 1

| Hạng mục | Trạng thái |
|---|---|
| T1-R1 RGB khớp paper gốc (98.52% vs 98.57%) | ✅ Pipeline tin cậy |
| T1-R2 13bands fixed (98.85%) vượt RGB | ✅ Đảo finding cũ, paired Δ +0.33pp |
| Std training stable (0.22-0.24%) | ✅ Cả hai cấu hình ổn định |
| Hard:Easy gap thu hẹp khi dùng 13b (2.03 → 1.44pp) | ✅ Tín hiệu domain knowledge hoạt động |
| Must-have Hard F1 +1.5pp (vs T1-R1) | ⚠️ Chưa đạt mean (+0.68pp), nhưng seed 42 alone đạt +1.31pp |

**Mục tiêu Tầng 2:** Spectral indices có thêm value gì so với 13 bands raw không? Baseline neo: **T1-R2 fixed = 98.85%, Hard F1 0.9793**.

---

## 4. Tầng 2 — Hypothesis-driven

### 4.1 T2-H2: 13 bands + 4 indices FIXED (ResNet-50, 17 channels)

**Câu hỏi:** 13 bands + 4 indices (NDVI/NDWI/NDBI/NDMI) có hơn 13 bands trần?

| Seed | Test Acc | Macro-F1 | Best Val Acc | W&B Run id |
|------|----------|----------|--------------|------------|
| 42   | 98.48%   | 98.44%   | 99.22%       | `8wpd9e74` |
| 123  | 98.93%   | 98.88%   | 99.26%       | `4avj5mvp` |
| 2024 | 98.44%   | 98.37%   | 99.07%       | `zuph7frg` |
| **Mean ± Std** | **98.62% ± 0.22%** | **98.56% ± 0.22%** | — | — |

> **Std 0.22%** — bằng T1-R1, thấp hơn buggy T2-H2 (0.37%). Sau fix, thêm indices không còn gây instability.

### 4.2 T2-H2 — Per-class F1 (3 seeds)

| Class | seed 42 | seed 123 | seed 2024 | Mean | Group |
|---|---|---|---|---|---|
| Pasture              | 0.9747 | 0.9773 | 0.9626 | 0.9716 | Hard |
| PermanentCrop        | 0.9703 | 0.9817 | 0.9658 | 0.9726 | Hard |
| HerbaceousVegetation | 0.9799 | 0.9851 | 0.9682 | 0.9777 | Hard |
| AnnualCrop           | 0.9668 | 0.9885 | 0.9818 | 0.9790 | Hard |
| Industrial           | 0.9900 | 0.9859 | 0.9900 | 0.9886 | Easy |
| Highway              | 0.9820 | 0.9920 | 0.9901 | 0.9881 | Easy |
| Residential          | 0.9933 | 0.9917 | 0.9900 | 0.9917 | Easy |
| River                | 0.9940 | 0.9920 | 0.9940 | 0.9933 | Easy |
| SeaLake              | 0.9967 | 0.9950 | 0.9983 | 0.9967 | Easy |
| Forest               | 0.9967 | 0.9983 | 0.9967 | 0.9972 | Easy |

| Nhóm | Mean F1 |
|---|---|
| Hard (4 lớp) | **0.9752** |
| Easy (6 lớp) | 0.9926 |
| **Gap** | **1.74pp** |

### 4.3 Paired Δ T2-H2 vs T1-R2 fixed (đóng góp ròng của 4 indices)

| Seed | Δ Test Acc | Δ Macro-F1 | Δ Hard F1 |
|---|---|---|---|
| 42   | **−0.48pp** | −0.49pp | **−0.90pp** |
| 123  | +0.41pp | +0.44pp | +0.97pp |
| 2024 | **−0.63pp** | −0.65pp | **−1.31pp** |
| **Mean** | **−0.23pp** | **−0.23pp** | **−0.41pp** |

> Per-seed std rất lớn (0.57pp), mean Δ âm nhưng SE ≈ 0.32pp → **không significant**. Honest: 4 indices **không cho thấy benefit**, có thể có drag nhẹ. 1/3 seeds dương rõ, 2/3 âm rõ.

### 4.4 Mục tiêu must-have có đạt không?

| Tiêu chí (CLAUDE.md §3) | Yêu cầu | Đo được | Status |
|---|---|---|---|
| Acc ≥ baseline RGB (98.52%) | ≥ 98.52% | 98.62% | ✅ Đạt (+0.10pp) |
| Acc ≥ baseline 13bands fixed (98.85%) | ≥ 98.85% | 98.62% | ❌ KHÔNG đạt (−0.23pp) |
| Macro-F1 Hard tăng ≥ 1.5pp vs T1-R1 RGB | ≥ 98.76% | 97.52% | ❌ KHÔNG đạt |
| Macro-F1 Hard tăng ≥ 1.5pp vs T1-R2 fixed | ≥ 99.43% | 97.52% | ❌ KHÔNG đạt |
| Acc ≥ 99.0% (nice-to-have) | ≥ 99.0% | 98.62% | ❌ |

**Kết luận:** Thêm 4 indices vào 13 bands không cải thiện mean acc so với 13 bands alone (paired Δ −0.23pp, SE ±0.32pp — trong noise band). Direction âm consistently trên 2/3 seeds (seed 42: −0.48pp, seed 2024: −0.63pp), 1/3 seeds dương (seed 123: +0.41pp). Đây là **negative result đáng note**, nhưng với n=3 chưa đủ statistical power để claim chắc chắn "indices hại"; chỉ đủ để claim "indices KHÔNG có benefit hiển nhiên".

---

## 5. Tầng 2 Phase 2 — Index ablation (T2-H3a/b/c/d + T2-H4)

### 5.1 Setup

Để rõ index nào đóng góp gì (nếu có), chạy 4 ablation đơn lẻ + 1 ablation "indices không có 13 bands":

| Run | Input | Channels | Seeds chạy |
|---|---|---|---|
| T2-H3a | 13b + NDVI | 14 | seed 42 |
| T2-H3b | 13b + NDWI | 14 | seed 42 |
| T2-H3c | 13b + NDBI | 14 | seed 42, 123, 2024, 7 (multi-seed) |
| T2-H3d | 13b + NDMI | 14 | seed 42 |
| T2-H4 | RGB + 4 indices | 7 | seed 42 (KHÔNG bị bug) |

> **Cảnh báo:** H3a/H3b/H3d chỉ 1 seed (42) sau fix — paired Δ chỉ valid trên 1 data point. Nhiều khả năng underestimate (seed 42 hay early-stop sớm với cấu hình indices, xem §6.3). H3c là chuẩn so sánh (n=4).

### 5.2 Bảng tổng hợp ablation (seed 42 fixed)

| Run | Input | Test Acc | Macro-F1 | Hard F1 mean | Easy F1 mean | Gap |
|---|---|---|---|---|---|---|
| **T1-R2 seed 42 fixed** (ref) | 13 bands | 98.96% | 98.93% | **98.69%** | 99.16% | 0.47pp |
| T2-H2 seed 42 | 13b + 4 indices | 98.48% | 98.44% | 97.79% | 99.21% | 1.41pp |
| T2-H3a seed 42 | 13b + NDVI | 98.33% | 98.26% | 96.77% | 99.26% | 2.49pp |
| T2-H3b seed 42 | 13b + NDWI | 98.37% | 98.35% | 96.98% | 99.27% | 2.29pp |
| **T2-H3c seed 42** | 13b + NDBI | 98.48% | 98.44% | 97.34% | 99.34% | 2.00pp |
| T2-H3d seed 42 | 13b + NDMI | 98.74% | 98.69% | 97.66% | 99.38% | 1.72pp |
| T2-H4 seed 42 | RGB + 4 indices | 98.48% | 98.42% | 97.23% | 99.22% | 1.99pp |
| **T2-H5 seed 42** | **4 indices ONLY (no bands)** | **97.44%** | 97.36% | **95.44%** | 98.64% | **3.20pp** |

**Quan sát chính (seed 42):**
- **Mọi cấu hình thêm indices ≤ T1-R2 fixed** trên seed 42. Δ từ −0.22pp (H3d) tới −0.63pp (H3a).
- H3a (NDVI) yếu nhất — bất ngờ về mặt vật lí (NDVI nhắm vegetation, đáng lẽ giúp 4 lớp Hard nhất).
- H3d (NDMI) tốt thứ hai — gần với T1-R2 fixed (−0.22pp acc).
- **H3c seed 42 fixed early-stopped epoch 13** — nguyên nhân chính của số thấp (xem §6.3).
- H4 (RGB+indices) **không hơn T1-R1 RGB** (98.48 vs 98.52) → indices KHÔNG bù được mất mát 10 band IR/SWIR/red-edge.
- **T2-H5 (indices alone, 4 ch): 97.44%, Hard F1 95.44%** — thấp hơn T1-R1 RGB rõ rệt (−1.08pp acc, **−1.82pp Hard F1**). Gap Hard-Easy giãn lên 3.20pp (vs RGB 2.03pp). Per-class: PermanentCrop 0.9407, HerbaceousVegetation 0.9493 — drop mạnh nhất. **Nghịch lý mạnh:** NDVI/NDMI được thiết kế để discriminate vegetation, nhưng "indices alone" lại **tệ HƠN RGB ở 4 lớp Hard vegetation**. Lý do vật lí: indices collapse 2 chiều spectral (B08+B04) về 1 chiều ratio → mất fine spectral signature cần để phân biệt Annual/Permanent/Pasture/Herbaceous. Đây là **direct evidence** rằng indices = lossy compression (H_A), KHÔNG phải engineered value độc lập (H_B).

### 5.3 T2-H3c — Multi-seed (n=4 sau fix)

Để discriminate "H3c thật sự kém hay seed 42 outlier", chạy 4 seeds.

| Seed | Test Acc | Macro-F1 | Best Val Acc | W&B Run id |
|------|----------|----------|--------------|------------|
| 42   | 98.48%   | 98.44%   | 98.85%       | `nzh4fzap` |
| 123  | 98.93%   | 98.89%   | 99.11%       | `ge4qs8br` |
| 2024 | 98.63%   | 98.61%   | 99.15%       | `grur79gt` |
| 7    | 99.15%   | 99.12%   | 99.41%       | `czocz87v` |
| **Mean ± Std (n=4)** | **98.80% ± 0.26%** | **98.76% ± 0.26%** | — | — |

> Std 0.26% — ngang T1-R2 fixed (0.24%). Seed 7 best 99.15% (cao nhất Tier 2 sau fix), seed 42 lowest 98.48% (early-stop epoch 13).

### 5.4 T2-H3c — Per-class F1 (4 seeds)

| Class | s7 | s42 | s123 | s2024 | Mean | Group |
|---|---|---|---|---|---|---|
| PermanentCrop        | 0.9820 | 0.9662 | 0.9704 | 0.9702 | **0.9722** | Hard |
| AnnualCrop           | 0.9868 | 0.9766 | 0.9766 | 0.9750 | 0.9787 | Hard |
| Pasture              | 0.9799 | 0.9776 | 0.9824 | 0.9774 | 0.9793 | Hard |
| HerbaceousVegetation | 0.9816 | 0.9732 | 0.9867 | 0.9815 | 0.9808 | Hard |
| Highway              | 0.9940 | 0.9843 | 0.9940 | 0.9980 | 0.9926 | Easy |
| River                | 0.9980 | 0.9899 | 0.9900 | 0.9901 | 0.9920 | Easy |
| Industrial           | 0.9960 | 0.9899 | 0.9940 | 0.9920 | 0.9930 | Easy |
| Residential          | 0.9950 | 0.9950 | 0.9967 | 0.9900 | 0.9942 | Easy |
| Forest               | 0.9983 | 0.9934 | 1.0000 | 0.9934 | 0.9963 | Easy |
| SeaLake              | 1.0000 | 0.9983 | 0.9983 | 0.9933 | **0.9975** | Easy |

| Nhóm | Mean F1 |
|---|---|
| Hard (4 lớp) | **0.9777** |
| Easy (6 lớp) | 0.9942 |
| **Gap** | **1.65pp** |

### 5.5 Paired Δ T2-H3c vs T1-R2 fixed (đóng góp ròng của NDBI), n=3 common seeds

| Seed | Δ Test Acc | Δ Macro-F1 | Δ Hard F1 |
|---|---|---|---|
| 42   | **−0.48pp** | −0.49pp | **−0.85pp** |
| 123  | +0.41pp | +0.44pp | +0.55pp |
| 2024 | **−0.44pp** | −0.41pp | **−0.66pp** |
| **Mean (n=3)** | **−0.17pp** | **−0.15pp** | **−0.32pp** |

> SE ≈ 0.27pp, mean Δ −0.17pp → **không significant**. 2/3 seeds âm rõ, 1/3 seeds dương rõ.

**Honest claim:** NDBI **KHÔNG cho thấy benefit** so với 13b alone. Tương tự T2-H2.

### 5.6 Mục tiêu must-have cho H3c (n=4)

| Tiêu chí | Yêu cầu | Đo được (mean n=4) | Status |
|---|---|---|---|
| Acc ≥ baseline RGB (98.52%) | ≥ 98.52% | 98.80% | ✅ +0.28pp |
| Acc ≥ baseline 13bands fixed (98.85%) | ≥ 98.85% | 98.80% | ⚠️ Sát (−0.05pp) |
| Macro-F1 Hard tăng ≥ 1.5pp vs T1-R1 | ≥ 98.76% | 97.77% | ❌ |
| Macro-F1 Hard tăng ≥ 1.5pp vs T1-R2 fixed | ≥ 99.43% | 97.77% | ❌ |
| Acc ≥ 99.0% (nice-to-have) | ≥ 99.0% | 98.80% | ❌ (sát) |

---

## 6. Cross-comparison toàn Tầng 2 fixed

### 6.1 Bảng tổng hợp tất cả cấu hình fixed (acc + Hard F1)

| Config | Channels | n seeds | Test Acc mean ± std | Hard F1 mean | Easy F1 mean | Gap |
|---|---|---|---|---|---|---|
| T1-R1 RGB (no-bug) | 3 | 3 | 98.52% ± 0.22% | 0.9726 | 0.9928 | 2.03pp |
| **T1-R2 13b fixed** | 13 | 3 | **98.85% ± 0.24%** | **0.9793** | 0.9938 | **1.44pp** |
| T2-H2 13b+4idx fixed | 17 | 3 | 98.62% ± 0.22% | 0.9752 | 0.9926 | 1.74pp |
| T2-H3a 13b+NDVI fixed | 14 | 1 | 98.33% | 0.9677 | 0.9926 | 2.49pp |
| T2-H3b 13b+NDWI fixed | 14 | 1 | 98.37% | 0.9698 | 0.9927 | 2.29pp |
| T2-H3c 13b+NDBI fixed | 14 | 4 | 98.80% ± 0.26% | 0.9777 | 0.9942 | 1.65pp |
| T2-H3d 13b+NDMI fixed | 14 | 1 | 98.74% | 0.9766 | 0.9938 | 1.72pp |
| T2-H4 RGB+4idx (no-bug) | 7 | 1 | 98.48% | 0.9723 | 0.9922 | 1.99pp |
| T2-H5 indices ONLY (no bands) | 4 | 1 | **97.44%** | **0.9544** | 0.9864 | **3.20pp** |

**Ranking theo Test Acc mean:**
1. **T1-R2 13b fixed (98.85%)** — winner
2. T2-H3c (98.80%, n=4) — sát, không signif
3. T2-H3d (98.74%, n=1) — sát, n thấp
4. T2-H2 (98.62%, n=3) — kém T1-R2 −0.23pp
5. T1-R1 RGB (98.52%, n=3)
6. T2-H4 (98.48%, n=1)
7. T2-H3b (98.37%, n=1)
8. T2-H3a (98.33%, n=1)
9. **T2-H5 indices-only (97.44%, n=1)** — thấp nhất, gap Hard-Easy giãn nhất

### 6.2 Paired Δ cross-config (n=3 seeds common)

| So sánh | Δ acc | Δ Hard F1 | Verdict |
|---|---|---|---|
| H3c − T1-R2 fixed | −0.17pp | −0.32pp | NDBI không help |
| H2 − T1-R2 fixed | −0.23pp | −0.41pp | 4 indices combo không help |
| H2 − H3c | −0.06pp | −0.09pp | 4 indices ≈ NDBI alone (cả hai cùng "không help") |
| T1-R2 fixed − T1-R1 RGB | **+0.33pp** | **+0.68pp** | **13 bands HƠN RGB** (marginal) |
| H3c − T1-R1 RGB | +0.16pp | +0.36pp | Có dấu hiệu nhưng < T1-R2 alone |
| H5 (indices alone) − T1-R1 RGB | **−1.08pp** | **−1.82pp** | Indices alone TỆ HƠN RGB alone (n=1) |
| H5 (indices alone) − T1-R2 fixed | **−1.41pp** | **−2.49pp** | Indices alone tệ hơn 13b raw rất nhiều |

**Verdict tổng:**
1. Mọi cấu hình thêm indices đều ≤ T1-R2 fixed (13b alone) → indices không thêm value khi đã có raw bands
2. **Indices alone (H5)** TỆ HƠN cả RGB alone → indices = lossy compression của raw bands, không phải engineered value độc lập
3. Kết hợp 2 finding: **Trên cấu hình ResNet-50 + EuroSAT, indices KHÔNG cung cấp discriminative information mà raw bands không có; ngược lại chúng throw away một phần info quan trọng cho fine-grained vegetation classification**

### 6.3 Tại sao seed 42 hay "kẹt" với cấu hình indices?

| Config | seed 42 stop epoch | seed 42 acc |
|---|---|---|
| T1-R2 fixed | 35 | 98.96% |
| T2-H2 fixed | (chưa parse, val cao) | 98.48% |
| T2-H3a fixed | 13 | 98.33% |
| T2-H3b fixed | 20 | 98.37% |
| T2-H3c fixed | 13 | 98.48% |
| T2-H3d fixed | 36 | 98.74% |

**Pattern:** seed 42 + cấu hình có indices → early-stop sớm (epoch 13-20) → kẹt ở local optimum nông. T1-R2 alone không bị (epoch 35).

**Vật lí giải thích:** Thêm channel index = thêm 3,136 params/channel vào conv1 với prior yếu (`mean_w * 0.5`). Loss landscape có thêm flat region; cosine LR decay (T_max=50) làm step size nhỏ dần → không đủ thoát local minimum. Seed 42 vô tình init vào vùng này. Seed khác (123, 2024, 7) không bị.

**Hệ quả:** Số H3a/H3b/H3d (chỉ seed 42) bị underestimate mạnh. Nếu có multi-seed, mean có khả năng cao tương đương H3c — vẫn không vượt T1-R2 fixed.

---

## 7. Tầng 2 — Tổng kết & decisions

### 7.1 Findings khoa học chính

**F1 (positive):** 13 spectral bands raw (T1-R2 fixed) cho mean 98.85% — vượt RGB +0.33pp paired, vượt H3c +0.05pp (không signif). **Winner Tier 2 = T1-R2 13b alone.**

**F2 (negative):** Spectral indices **không cho thấy improvement có ý nghĩa thống kê** so với 13 bands raw + conv1 đúng prior (paired Δ mean −0.17pp đến −0.23pp, n=3 seeds, SE ±0.27-0.32pp; mọi Δ nằm trong noise band). Đây là negative result đáng note nhưng **chưa đủ "bác bỏ" giả thuyết indices có domain value nói chung** — sample size n=3 không đủ để claim equivalence với confidence cao. Honest claim: **trên cấu hình ResNet-50 pretrained ImageNet + EuroSAT 80/20 split này, indices KHÔNG phải đòn bẩy hiển nhiên** như tradition remote sensing có thể gợi ý. Cần n ≥ 5-10 seeds để claim equivalence chắc chắn.

**F3 (mechanism — giả thuyết):** ResNet-50 (25M params, 50 layers) có capacity về mặt lý thuyết để học implicit transformations tương đương indices từ B04/B03/B08/B11/B12 raw (subtraction trivial; ratio approximable). Nếu giả thuyết này đúng, indices precomputed trở thành **redundant features** → chỉ thêm noise vào optimization, không thêm information. **Lưu ý:** đây là *mechanism hypothesis* để giải thích F2, KHÔNG phải đã được prove. Để verify cần ablation thêm: probe intermediate activations xem có học implicit indices không, hoặc test trên model nhỏ hơn (capacity hạn chế) xem indices có giúp.

**F5 (direct evidence — indices = lossy compression, từ T2-H5):** Chạy "indices alone" (4 channels NDVI/NDWI/NDBI/NDMI, không raw bands, seed 42) → acc **97.44%**, **THẤP HƠN RGB 98.52%** (−1.08pp acc, **−1.82pp Hard F1**). Per-class: PermanentCrop, HerbaceousVegetation, Pasture đều drop −1.87 đến −2.70pp so với RGB alone. Đây là direct evidence rằng:
- Indices throw away discriminative information có trong raw bands (rõ nhất ở fine-grained vegetation classes)
- 4 indices < 3 RGB bands về discriminative power cho EuroSAT classification
- Giả thuyết "indices = engineered domain knowledge bonus" bị **trực tiếp refute** trên cấu hình này: nếu indices có engineered value độc lập, "indices alone" nên gần RGB (~98%); thực tế thấp hơn rõ
- **Caveat:** "indices alone" cũng không nhận ImageNet RGB prior (không có B04/B03/B02 trong input) → một phần drop có thể do prior loss, không hoàn toàn do information loss. Để discriminate cleanly cần test "3 random spectral bands without ImageNet prior" làm control.

**F4 (gap):** Hard:Easy gap thu hẹp khi dùng 13b (2.03 → 1.44pp) nhưng KHÔNG thu hẹp thêm khi indices added. Suggests gap còn lại (~1.5pp) là **label noise** giữa các vegetation classes (Annual/Permanent/Pasture/Herbaceous), không phải input-side bottleneck.

### 7.2 Must-have target (CLAUDE.md §3)

| Tiêu chí | Mean tốt nhất | Best single-seed | Đạt? |
|---|---|---|---|
| Acc ≥ T1-R1 RGB baseline (98.52%) | 98.85% (T1-R2) | 99.15% (H3c seed 7) | ✅ |
| Macro-F1 Hard tăng ≥ 1.5pp vs T1-R1 (cần ≥ 98.76%) | 97.93% (T1-R2) | 98.69% (T1-R2 seed 42) | ⚠️ Mean chưa đạt (gap −0.83pp); single seed 42 gần đạt |
| Acc ≥ 99.0% (nice-to-have) | 98.85% (T1-R2) | 99.15% (H3c seed 7) | ⚠️ Mean chưa đạt; single seed có |
| Hard F1 tăng ≥ 2.0pp (nice-to-have) | +0.68pp | +1.31pp (T1-R2 seed 42) | ❌ |

### 7.3 Quyết định Tầng 3

**Input chốt: T1-R2 (13 bands raw, no indices)** — đảo so với plan cũ ("H3c+NDBI là winner"). Lý do: paired Δ multi-seed sau fix cho thấy indices không cung cấp thêm value.

**Ưu tiên Tier 3:**

1. **Augmentation sweep (ưu tiên #1)** — vì input engineering (indices) không cho thấy benefit trong Tier 2; hướng tiếp theo khả thi nhất là regularization/data augmentation. Aug candidates: MixUp / CutMix / RandomResizedCrop. Target: thu hẹp gap +0.83pp Hard F1 còn thiếu để đạt must-have.
2. **LR sweep (T3-A)** — test 3e-5, 3e-4 ngoài 1e-4 hiện tại.
3. **Optimizer sweep (T3-D)** — SGD+Nesterov vs AdamW.
4. **T3-E final 5 seeds (42/123/2024/7/999)** — chốt số liệu cuối Refine.

**Có thể skip:** Tier 2 Phase 3 (T2-H5a/b/c band-group ablation) — câu hỏi "bands nào quan trọng" trở nên ít cấp bách vì 13b alone đã winner. Cân nhắc giữ nếu cần Discussion section trong paper bàn về spectral importance.

---

## 7bis. Tier 3 — Tuning & Final (2026-05-25)

Mọi Tier 3 sweep dùng base = **T1-R2 13b raw, fixed-conv1, seed 42** (single-seed cho sweep, multi-seed cho T3-E final). Recipe baseline: AdamW lr=1e-4, wd=1e-4, batch_size=32, cosine T_max=50.

### 7bis.1 T3-C — Augmentation sweep (seed 42)

| Run | Aug | Test Acc | Hard F1 | Easy F1 | Gap | Δ acc vs baseline |
|---|---|---|---|---|---|---|
| **Baseline T1-R2 s42** | flip+rot90 | 98.96% | 98.69% | 99.16% | 0.47pp | — |
| T3-C RRC | flip+rot90 + RandomResizedCrop scale[0.7,1] | 98.22% | 97.47% | 98.70% | 1.23pp | **−0.74pp** ❌ |
| T3-C MixUp | flip+rot90 + MixUp α=0.2 | 99.04% | 97.94% | 99.72% | 1.78pp | +0.08pp ~tie |
| **T3-C CutMix** | flip+rot90 + CutMix α=1.0 | **99.15%** | 98.36% | 99.62% | 1.26pp | **+0.19pp** ✅ |

**Findings T3-C:**
- **CutMix** best acc single seed (+0.19pp) nhưng Hard F1 thấp hơn baseline (−0.33pp)
- **MixUp** acc tie baseline nhưng **HẠI Hard F1 −0.75pp** — blending 2 ảnh satellite tạo input vật lí không sensical, nhạy với fine-grained vegetation classes
- **RRC HẠI cả acc lẫn Hard F1** — confirm "EuroSAT class = whole tile", crop 70% mất context cần cho discrimination
- Đa phần aug **giãn gap Hard-Easy** (từ baseline 0.47pp → 1.23-1.78pp) → aug giúp Easy classes nhiều hơn Hard

### 7bis.2 T3-A — Learning rate sweep (seed 42, AdamW)

| LR | Test Acc | Hard F1 | Easy F1 | Gap | Δ acc vs baseline |
|---|---|---|---|---|---|
| 3e-5 | 98.44% | 97.35% | 99.11% | 1.76pp | −0.52pp ❌ |
| **1e-4 (baseline)** | **98.96%** | 98.69% | 99.16% | 0.47pp | — |
| 3e-4 | 98.63% | 97.66% | 99.19% | 1.54pp | −0.33pp ❌ |
| **1e-3** | **99.19%** 🏆 | 98.47% | 99.61% | 1.14pp | **+0.23pp** ✅ |

**Findings T3-A:**
- **Bất ngờ:** lr=1e-3 (10× baseline) thắng single seed, KHÔNG diverge như dự đoán
- Cosine annealing nhanh decay lr cao → ổn định
- 3e-5 và 3e-4 đều dưới baseline → optimal lr không nằm gần 1e-4
- Hard F1 giảm nhẹ ở lr cao — model fit tốt hơn Easy hơn Hard

### 7bis.3 T3-B — Weight decay sweep (seed 42, AdamW lr=1e-4)

| WD | Test Acc | Hard F1 | Easy F1 | Gap | Δ acc vs baseline |
|---|---|---|---|---|---|
| 1e-5 | 98.96% | 98.19% | 99.43% | 1.25pp | 0.00pp (tie) |
| **1e-4 (baseline)** | **98.96%** | 98.69% | 99.16% | 0.47pp | — |
| 5e-4 | 98.26% | 97.04% | 99.01% | 1.97pp | **−0.70pp** ❌ |

**Findings T3-B:**
- WD 1e-5 tie baseline về acc nhưng Hard F1 thấp hơn 0.50pp
- WD 5e-4 HẠI mạnh (−0.70pp acc, **−1.65pp Hard F1**) — early-stop epoch 12, quá mạnh regularize làm underfit
- **Baseline wd=1e-4 là optimal** — không có gain từ sweep

### 7bis.4 T3-D — Optimizer sweep (seed 42)

| Optimizer | Hyperparams | Test Acc | Hard F1 | Easy F1 | Gap | Δ acc vs baseline |
|---|---|---|---|---|---|---|
| **AdamW (baseline)** | lr=1e-4, wd=1e-4 | 98.96% | **98.69%** | 99.16% | **0.47pp** | — |
| **SGD+Nesterov** | lr=0.01, mom=0.9, wd=5e-4 | **99.07%** | 98.63% | 99.33% | 0.70pp | **+0.11pp** ✅ |

**Findings T3-D:**
- SGD+Nesterov single seed gần tie baseline (acc +0.11pp, Hard F1 −0.06pp ≈ tie)
- Cả 2 trong noise band → cần multi-seed (T3-E) để discriminate

### 7bis.5 Ranking Tier 3 single-seed (winner seed 42)

| Rank | Config | Acc | Hard F1 | Gap |
|---|---|---|---|---|
| 1 | T3-A lr 1e-3 | **99.19%** | 98.47% | 1.14pp |
| 2 | T3-C CutMix | 99.15% | 98.36% | 1.26pp |
| 3 | **T3-D SGD+Nesterov** | 99.07% | **98.63%** | **0.70pp** |
| 4 | T3-C MixUp | 99.04% | 97.94% | 1.78pp |
| 5 | T1-R2 baseline (=T3-B wd1e-5) | 98.96% | **98.69%** | 0.47pp |
| 6 | T3-A lr 3e-4 | 98.63% | 97.66% | 1.54pp |
| 7 | T3-A lr 3e-5 | 98.44% | 97.35% | 1.76pp |
| 8 | T3-B wd 5e-4 | 98.26% | 97.04% | 1.97pp |
| 9 | T3-C RRC | 98.22% | 97.47% | 1.23pp |

**Quyết định:** T3-D SGD+Nesterov chọn cho T3-E final multi-seed vì balance tốt nhất (top-3 acc, Hard F1 sát baseline, gap nhỏ thứ 2).

### 7bis.6 T3-E — Final multi-seed (5 seeds, SGD+Nesterov)

| Seed | Test Acc | Macro-F1 | Hard F1 | Easy F1 | Gap |
|---|---|---|---|---|---|
| 42   | 99.07% | 99.05% | 98.63% | 99.33% | 0.70pp |
| 123  | 98.85% | 98.82% | 97.81% | 99.49% | 1.69pp |
| 2024 | 98.70% | 98.67% | 97.62% | 99.37% | 1.75pp |
| 7    | 98.30% | 98.26% | 97.20% | 98.97% | 1.78pp |
| 999  | 98.74% | 98.72% | 97.57% | 99.48% | 1.91pp |
| **Mean ± Std** | **98.73% ± 0.28%** | **98.70% ± 0.29%** | **97.77% ± 0.53%** | **99.33% ± 0.21%** | **1.56pp** |

**Per-class F1 (5 seeds):**

| Class | Mean F1 | Std | Group |
|---|---|---|---|
| PermanentCrop | 0.9719 | 0.0063 | Hard |
| Pasture | 0.9809 | 0.0055 | Hard |
| AnnualCrop | 0.9774 | 0.0047 | Hard |
| HerbaceousVegetation | 0.9804 | 0.0058 | Hard |
| Highway | 0.9891 | 0.0049 | Easy |
| Industrial | 0.9895 | 0.0034 | Easy |
| Residential | 0.9924 | 0.0050 | Easy |
| River | 0.9936 | 0.0022 | Easy |
| Forest | 0.9963 | 0.0018 | Easy |
| SeaLake | **0.9987** | 0.0007 | Easy |

### 7bis.7 So sánh T3-E với baseline T1-R2 fixed

| Metric | T1-R2 fixed (AdamW, n=3) | T3-E SGD (n=5) | Δ |
|---|---|---|---|
| Test Acc | **98.85% ± 0.24%** | 98.73% ± 0.28% | −0.12pp |
| Macro-F1 | 98.80% ± 0.25% | 98.70% ± 0.29% | −0.10pp |
| Hard F1 | **97.93%** | 97.77% ± 0.53% | −0.16pp |
| Easy F1 | 99.38% | 99.33% ± 0.21% | −0.05pp |
| Gap | 1.44pp | 1.56pp | +0.12pp (rộng hơn) |

**🚨 SGD KHÔNG vượt AdamW baseline** trên mean 5 seeds. T3-D single seed (s42 = 99.07%) là **lucky outlier dương**; mean qua 5 seeds giảm xuống 98.73%.

### 7bis.8 Must-have target — verdict cuối

| Tiêu chí (CLAUDE.md §3) | Yêu cầu | T3-E mean | Baseline T1-R2 mean | Status |
|---|---|---|---|---|
| Acc ≥ T1-R1 RGB (98.52%) | ≥ 98.52% | 98.73% | 98.85% | ✅ |
| Acc ≥ baseline T1-R2 fixed | ≥ 98.85% | 98.73% | — | ❌ |
| **Hard F1 ≥ T1-R1 + 1.5pp (must-have)** | ≥ 98.76% | 97.77% | 97.93% | ❌ (gap −0.99pp / −0.83pp) |
| Acc ≥ 99.0% (nice-to-have) | ≥ 99.0% | 98.73% | 98.85% | ❌ |
| Hard F1 ≥ T1-R1 + 2.0pp (nice-to-have) | ≥ 99.26% | 97.77% | — | ❌ |

**Verdict:** Must-have Hard F1 +1.5pp **KHÔNG đạt** sau toàn bộ Refine. Best Hard F1 mean = **97.93% (T1-R2 baseline)**, thiếu 0.83pp.

### 7bis.9 Winner Refine cuối cùng

| Rank | Config | n | Acc mean | Hard F1 mean | Best single-seed acc |
|---|---|---|---|---|---|
| 🏆 1 | **T1-R2 13b fixed (AdamW lr 1e-4)** | 3 | **98.85% ± 0.24%** | **97.93%** | 99.07% (s2024) |
| 2 | T2-H3c 13b+NDBI fixed | 4 | 98.80% ± 0.26% | 97.77% | 99.15% (s7) |
| 3 | T3-E SGD+Nesterov | 5 | 98.73% ± 0.28% | 97.77% | 99.07% (s42) |
| 4 | T2-H2 13b+4idx fixed | 3 | 98.62% ± 0.22% | 97.52% | 98.93% (s123) |
| 5 | T1-R1 RGB | 3 | 98.52% ± 0.22% | 97.26% | 98.81% (s123) |

**Winner = T1-R2 13b fixed + AdamW default recipe** (simplest, best on Hard F1 mean, best stability).

### 7bis.10 Bài học methodological từ Tier 3

**Single-seed ranking KHÔNG đáng tin cho sweep:**

- T3-D SGD s42 = 99.07% → predict winner → multi-seed 5 seeds = 98.73% (thua baseline AdamW)
- T3-A lr 1e-3 s42 = 99.19% (best single) — chưa multi-seed verify, có thể cũng lucky outlier
- Lesson: sweep dùng single seed CHỈ để thu hẹp search space; commit phải multi-seed

**Seed effect lớn cho lớp Hard:**

- T3-E Hard F1 std = 0.53pp (gấp đôi acc std 0.28pp)
- Lớp Hard nhạy với random init hơn lớp Easy
- Single-seed Hard F1 ranking có thể đảo qua các seed

**Aug không phải đòn bẩy:**

- MixUp HẠI Hard F1 trên satellite imagery (trái với literature ImageNet)
- RRC HẠI cả acc lẫn Hard F1 (EuroSAT class = whole tile)
- CutMix marginal — chỉ gain Easy classes
- Confirm Tier 3 prediction: gap Hard-Easy còn lại là **fundamental** (label noise, 64×64 resolution), không phải aug/optim issue

---

## 8. Phụ lục — Bug history & old (buggy) results

Mọi run trước 2026-05-24 (trừ T1-R1 RGB và T2-H4) chạy với `_adapt_conv1` cũ — copy ImageNet RGB weights vô điều kiện vào 3 channel đầu của input. Với `bands: all` order [B01, B02, B03, B04, ...], R/G/B prior bị apply lên Aerosol/Blue/Green và B04 (Red) không nhận prior chính. Hệ quả: mọi run 13+channel underestimate acc ~0.4pp.

### 8.1 Số liệu trước/sau fix (mean acc)

| Config | Buggy mean | Fixed mean | Δ |
|---|---|---|---|
| T1-R2 13bands | 98.48% | **98.85%** | **+0.37pp** |
| T2-H2 13b+4idx | 98.48% | **98.62%** | +0.14pp |
| T2-H3c 13b+NDBI (3 common seeds) | 98.81% | 98.68% | −0.13pp |
| T2-H3a 13b+NDVI (seed 42) | 98.89% | 98.33% | −0.56pp |
| T2-H3b 13b+NDWI (seed 42) | 98.70% | 98.37% | −0.33pp |
| T2-H3d 13b+NDMI (seed 42) | 98.89% | 98.74% | −0.15pp |
| T1-R1 RGB (no bug) | 98.52% | 98.52% | 0 |
| T2-H4 (no bug) | 98.48% | 98.48% | 0 |

**Quan sát quan trọng:** Bug fix giúp T1-R2 alone (+0.37pp) MẠNH HƠN giúp các cấu hình có indices. **Giả thuyết giải thích (chưa prove):** với T1-R2, fix giải phóng full potential của B04 Red prior; với cấu hình indices, B04 + Red prior CÓ THỂ làm model học implicit NDVI/NDBI → indices explicit thành redundant. Cần probe activations hoặc test ablation thêm để confirm mechanism này.

### 8.2 Findings cũ ĐẢO sau fix

| Finding cũ (buggy) | Finding mới (fixed) |
|---|---|
| "13 bands không vượt RGB" (paired Δ −0.04pp) | **13 bands vượt RGB +0.33pp** |
| "NDBI là winner Tier 2" (H3c 98.86% > T1-R2 98.48%) | **T1-R2 alone là winner** (98.85% ≥ H3c 98.80%) |
| "NDBI cải thiện +0.38pp vs T1-R2" | NDBI giảm −0.17pp vs T1-R2 (paired, n=3) |
| "Single index > combo 4 indices" | Vẫn đúng định tính (H3c ≥ H2) nhưng cả hai cùng KHÔNG vượt T1-R2 |
| "Tier 3 input = 13b+NDBI" | **Tier 3 input = 13b alone** |
| Indices có domain value (giả thuyết Refine) | Indices KHÔNG cho thấy improvement có ý nghĩa thống kê (paired Δ trong noise band, n=3) — chưa đủ kết luận "redundant"; cần n ≥ 5-10 seeds |

### 8.3 Methodological lesson (đáng đưa vào paper Discussion)

Bug conv1 RGB-prior misalignment **tạo ra finding khoa học sai** mà mọi internal consistency metric (paired Δ, std, ranking) đều ổn định và consistent. Chỉ phát hiện bằng *external sanity check* + skeptical audit khi user hỏi "có khả năng tính sai không?". 

**Bài học cho remote sensing community:** Mọi paper N-channel adapter kế thừa ImageNet conv1 cần verify **weight alignment** programmatic, không chỉ rely vào "caller responsibility" trong docstring. Symptom đáng nghi: "thêm spectral bands không cải thiện accuracy" — nếu thấy, audit conv1 init.

Quy tắc đã thêm vào CLAUDE.md §13 (Bộ quy tắc phòng ngừa lỗi do bất cẩn / thiếu kiểm chứng) — bắt buộc cho mọi adapter swap channels.

---

## 9. Decisions log

| Ngày | Quyết định | Lý do (kỹ thuật) |
|---|---|---|
| 2026-05-22 | Chạy T1-R1 RGB ×3 seeds | Reproduce paper baseline 98.57% — pipeline sanity check. |
| 2026-05-23 | Chạy T1-R2 13b ×3 seeds (buggy) | Compare 13b vs RGB, paired same seeds. |
| 2026-05-23 | Chạy T2-H2 + H3a-d + H4 (buggy) | Tier 2 hypothesis testing. |
| 2026-05-24 | Phát hiện bug conv1 RGB-prior misalignment | Skeptical audit khi user hỏi "có khả năng tính sai?" — finding "13b không vượt RGB" smoking gun bị rationalize sai 2 ngày. |
| 2026-05-24 | Fix `_adapt_conv1(rgb_positions=...)` + `build_resnet50(bands=...)` enforce ở boundary | Caller-responsibility contract không đủ — config `bands: all` vi phạm im lặng. Fix mới compute rgb_positions từ bands list. |
| 2026-05-24 | Re-run T1-R2 (3 seeds) + H3c (4 seeds) fixed | Confirm bug MAJOR (T1-R2 seed 42 +0.55pp acc, +1.13pp Hard F1). |
| 2026-05-25 | Re-run T2-H2 (3 seeds) + H3a/b/d (1 seed each) fixed | Hoàn thiện picture sau fix; phát hiện indices không cung cấp benefit. |
| 2026-05-25 | **Đảo winner Tier 2: T1-R2 alone thay vì H3c** | Paired Δ multi-seed (H3c − T1-R2 fixed = −0.17pp, không signif) + H2 − T1-R2 = −0.23pp. Indices không cho thấy improvement có ý nghĩa thống kê sau fix; T1-R2 nhỉnh hơn về mean. |
| 2026-05-25 | **Tier 3 input = T1-R2 (13b raw)**, ưu tiên aug sweep | Input engineering (indices) không cho thấy benefit; hướng khả thi tiếp theo là regularization/augmentation để thu hẹp gap còn lại đến must-have +1.5pp Hard F1. |
| 2026-05-25 | T2-H5 indices alone (4 channels, no raw bands) — direct test "indices = engineered value" | Confirm H_A: indices alone 97.44% < RGB 98.52%; Hard F1 95.44% drop −1.82pp. Indices = lossy compression của raw bands. |
| 2026-05-25 | Tier 3 T3-C aug sweep (RRC/MixUp/CutMix, seed 42) | CutMix marginal +0.19pp acc nhưng −0.33pp Hard F1; MixUp tie acc nhưng HẠI Hard F1 −0.75pp; RRC HẠI cả 2 metric. Aug không phải đòn bẩy cho EuroSAT 64×64. |
| 2026-05-25 | Tier 3 T3-A LR sweep (3e-5, 3e-4, 1e-3) | Bất ngờ: lr=1e-3 thắng single seed (99.19%) — không diverge nhờ cosine. Nhưng Hard F1 thấp hơn baseline. |
| 2026-05-25 | Tier 3 T3-B WD sweep (1e-5, 5e-4) | WD 1e-5 tie acc baseline; WD 5e-4 HẠI mạnh (early-stop ep 12, underfit). Baseline wd=1e-4 optimal. |
| 2026-05-25 | Tier 3 T3-D SGD+Nesterov (seed 42) | 99.07% gần tie baseline → chọn cho T3-E final multi-seed. |
| 2026-05-25 | **T3-E SGD final 5 seeds = 98.73% ± 0.28% (Hard F1 97.77%)** | T3-D single-seed s42 lucky outlier; mean 5 seeds dưới baseline T1-R2 AdamW (−0.12pp acc, −0.16pp Hard F1). Bài học: single-seed ranking không đáng tin cho sweep. |
| 2026-05-25 | **Winner Refine cuối = T1-R2 13b fixed + AdamW default recipe** | Mean acc 98.85% ± 0.24%, Hard F1 97.93%. Không config Tier 3 nào vượt baseline có ý nghĩa thống kê. Must-have Hard F1 +1.5pp KHÔNG đạt (gap −0.83pp). |
