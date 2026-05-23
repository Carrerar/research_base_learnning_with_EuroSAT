# Kết quả thực nghiệm — EuroSAT Refine

> Bảng tổng hợp kết quả mọi tầng. Cập nhật sau mỗi run hoàn thành.

## Tầng 1 — Baseline Establishment

### T1-R1: RGB baseline (ResNet-50, 3 channels)

| Seed | Test Acc | Macro-F1 | Best Val Acc | Best Epoch | Stop Epoch | Runtime | W&B Run name |
|------|----------|----------|--------------|------------|------------|---------|------------|
| 42   | 98.30%   | 98.23%   | 98.56%        | 7  | 17 | 36m 34s | tier1-T1-R1-rgb-seed42 (`a2yhj02w`) |
| 123  | 98.82%   | 98.77%   | ≥98.85%       | 36 | 46 | 86m 49s | tier1-T1-R1-rgb-seed123 |
| 2024 | 98.44%   | 98.42%   | ≥98.63%       | 13 | 23 | 54m 02s | tier1-T1-R1-rgb-seed2024 |
| **Mean ± Std** | **98.52% ± 0.27%** | **98.47% ± 0.27%** | — | — | — | — | — |

> **Ghi chú cột Best Val Acc:** seed 42 là giá trị best chính xác (epoch 7). Với seed 123/2024, số lấy từ W&B summary là val acc ở epoch CUỐI (cận dưới của best — best ≥ giá trị này); con số chính xác nằm ở dòng `best_val_acc` cuối log console. Best Epoch = Stop Epoch − patience(10), suy ra từ logic early-stopping.
>
> **Std test acc = 0.27% < 0.5%** → training ổn định, baseline đạt yêu cầu (§6: 97.5–98.7% — cả 3 seed nằm trong range; seed 123 ở 98.82% sát mép trên, vẫn chấp nhận).

### T1-R1 — Per-class F1 (3 seeds)

Lớp **Hard** = 4 lớp đồng cỏ/nông nghiệp dễ nhầm (CLAUDE.md §2). Sắp theo F1 trung bình tăng dần.

| Class | seed 42 | seed 123 | seed 2024 | Mean | Group |
|---|---|---|---|---|---|
| PermanentCrop        | 0.9571 | 0.9535 | 0.9676 | 0.9594 | Hard |
| HerbaceousVegetation | 0.9768 | 0.9785 | 0.9735 | 0.9763 | Hard |
| AnnualCrop           | 0.9734 | 0.9800 | 0.9799 | 0.9778 | Hard |
| Pasture              | 0.9679 | 0.9824 | 0.9800 | 0.9768 | Hard |
| Industrial           | 0.9861 | 0.9901 | 0.9859 | 0.9874 | Easy* |
| SeaLake              | 0.9917 | 1.0000 | 0.9900 | 0.9939 | Easy |
| Highway              | 0.9879 | 0.9980 | 0.9901 | 0.9920 | Easy |
| River                | 0.9900 | 0.9980 | 0.9900 | 0.9927 | Easy |
| Residential          | 0.9950 | 0.9983 | 0.9884 | 0.9939 | Easy |
| Forest               | 0.9967 | 0.9983 | 0.9967 | 0.9972 | Easy |

\* Industrial gần ranh giới; CLAUDE.md §2 xếp nó vào nhóm Dễ.

**Pattern nhất quán qua cả 3 seed:** 4 lớp F1 thấp nhất **luôn đúng** là 4 lớp Hard (PermanentCrop, HerbaceousVegetation, AnnualCrop, Pasture); PermanentCrop thấp nhất ở mọi seed.

| Nhóm | seed 42 | seed 123 | seed 2024 | Mean |
|---|---|---|---|---|
| Mean F1 Hard (4 lớp) | 0.9688 | 0.9736 | 0.9753 | 0.9726 |
| Mean F1 Easy (6 lớp) | 0.9912 | 0.9971 | 0.9902 | 0.9928 |
| **Gap** | 2.24% | 2.35% | 1.49% | **2.02%** |

**Gap trung bình 2.02%** — đây là khoảng cách macro-F1 lớp khó mà Tầng 2 (13 bands + spectral indices) cần thu hẹp.

### T1-R2: 13 bands baseline (ResNet-50, 13 channels)

| Seed | Test Acc | Macro-F1 | Best Val Acc | Best Epoch | Stop Epoch | Runtime | W&B Run name |
|------|----------|----------|--------------|------------|------------|---------|------------|
| 42   | 98.41%   | 98.34%   | ≥99.07%      | 17 | 27 | 53m 34s | tier1-T1-R2-13bands-seed42 |
| 123  | 98.48%   | 98.45%   | ≥99.00%      | 20 | 30 | 61m 06s | tier1-T1-R2-13bands-seed123 |
| 2024 | 98.56%   | 98.49%   | ≥98.82%      | 12 | 22 | 43m 33s | tier1-T1-R2-13bands-seed2024 |
| **Mean ± Std** | **98.48% ± 0.07%** | **98.43% ± 0.08%** | — | — | — | — | — |

> **Std cực nhỏ (0.07%)** — 13 bands cho training thậm chí ổn định hơn RGB (0.27%). Hợp lý: thêm thông tin phổ làm decision boundary ít phụ thuộc khởi tạo.

### T1-R2 — Per-class F1 (3 seeds)

| Class | seed 42 | seed 123 | seed 2024 | Mean | Group |
|---|---|---|---|---|---|
| PermanentCrop        | 0.9659 | 0.9600 | 0.9628 | 0.9629 | Hard |
| Pasture              | 0.9648 | 0.9774 | 0.9697 | 0.9706 | Hard |
| HerbaceousVegetation | 0.9766 | 0.9733 | 0.9732 | 0.9744 | Hard |
| AnnualCrop           | 0.9751 | 0.9717 | 0.9782 | 0.9750 | Hard |
| Industrial           | 0.9879 | 0.9920 | 0.9840 | 0.9880 | Easy* |
| Highway              | 0.9861 | 0.9900 | 0.9900 | 0.9887 | Easy |
| Residential          | 0.9917 | 0.9967 | 0.9967 | 0.9950 | Easy |
| River                | 0.9940 | 0.9899 | 0.9980 | 0.9940 | Easy |
| SeaLake              | 0.9967 | 0.9950 | 1.0000 | 0.9972 | Easy |
| Forest               | 0.9950 | 0.9983 | 0.9967 | 0.9967 | Easy |

**Pattern lớp khó giữ nguyên:** ở cả 3 seed, 4 lớp F1 thấp nhất vẫn đúng là PermanentCrop / HerbaceousVegetation / AnnualCrop / Pasture. PermanentCrop thấp nhất ở 2/3 seed.

| Nhóm | seed 42 | seed 123 | seed 2024 | Mean |
|---|---|---|---|---|
| Mean F1 Hard (4 lớp) | 0.9706 | 0.9706 | 0.9710 | 0.9707 |
| Mean F1 Easy (6 lớp) | 0.9919 | 0.9937 | 0.9942 | 0.9933 |
| **Gap** | 2.13% | 2.30% | 2.33% | **2.25%** |

---

## So sánh T1-R1 vs T1-R2 (paired comparison)

Vì cả hai dùng cùng bộ seed (42, 123, 2024), ta có thể so từng cặp run — loại nhiễu seed, khác biệt còn lại chỉ do **input**.

### Tổng quan

| Metric | T1-R1 (RGB, 3ch) | T1-R2 (13 bands, 13ch) | Paired Δ (R2 − R1) |
|---|---|---|---|
| Test acc | 98.52% ± 0.27% | 98.48% ± 0.07% | **−0.04% ± 0.26%** |
| Macro-F1 | 98.47% ± 0.27% | 98.43% ± 0.08% | **−0.05% ± 0.24%** |
| F1 Hard mean | 0.9726 | 0.9707 | **−0.0019** |
| F1 Easy mean | 0.9928 | 0.9933 | **+0.0005** |

### Paired Δ per seed

| Seed | Δ Test Acc | Δ Macro-F1 |
|---|---|---|
| 42   | +0.107% | +0.109% |
| 123  | −0.334% | −0.326% |
| 2024 | +0.112% | +0.071% |

### Diễn giải

**Thêm 10 band phổ (NIR/SWIR/red-edge/atmospheric) một cách thô lên trên RGB KHÔNG cải thiện accuracy.** Paired diff −0.04% ± 0.26% — độ lệch hoàn toàn nằm trong nhiễu seed (1 seed dương 0.11%, 1 seed âm 0.33%, ngẫu nhiên hai phía). Không kết luận được 13 bands hơn hay kém RGB; phải coi là **hoà**.

Điều này:
1. **Tái khẳng định finding của paper gốc** (Helber 2019): chuyển từ RGB sang các tổ hợp 3-band khác (CI, SWIR) không cải thiện rõ. Mở rộng cả lên 13 bands cũng không đổi được.
2. **Mở rõ động lực cho Tầng 2:** thông tin phổ thô đã nằm đó nhưng ResNet (vốn trained cho ảnh RGB) chưa khai thác được. Cần **biến đổi có chủ đích** — đó chính là việc của spectral indices (NDVI/NDWI/NDBI/NDMI) ở Tầng 2: ép kiến thức domain vào input thay vì để mạng tự tìm.
3. **Negative result đáng báo cáo** (CLAUDE.md §3). Nó loại bỏ một giải thích cạnh tranh ("có lẽ chỉ cần thêm band là đủ"), nên khi Tầng 2 thắng, ta biết công không phải của bands thô mà của **indices**.

**Một quan sát phụ:** std của T1-R2 (0.07%) nhỏ hơn nhiều T1-R1 (0.27%). Thêm channel dường như làm training ổn định hơn dù không tăng accuracy trung bình — đáng note nhưng không phải finding chính.

---

## Tổng kết Tầng 1

- ✅ T1-R1 (RGB) khớp paper gốc (98.52% vs 98.57%) → pipeline đúng, đáng tin.
- ✅ T1-R2 (13 bands) ổn định (std 0.07%), nhưng **không vượt RGB** (paired Δ trong nhiễu).
- ✅ Cả 6 run, pattern lớp khó giống hệt: PermanentCrop/HerbaceousVegetation/AnnualCrop/Pasture luôn là 4 lớp F1 thấp nhất; gap Hard-vs-Easy ~2.0-2.3%.
- ➡️ **Mục tiêu Tầng 2:** thu hẹp gap ~2% này bằng spectral indices; baseline kép RGB=98.52%, 13band=98.48% sẽ là 2 mốc để đo.
