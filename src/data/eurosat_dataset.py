"""EuroSAT all-bands dataset loader.

Đọc các patch 13-band (uint16) từ ``dataset/allbands/<Class>/<file>.tif`` và trả về
tensor ``(N_channels, 64, 64)`` float32. Hỗ trợ:

- Chọn subset bands theo TÊN band (không phải index thô).
- Tính spectral indices (NDVI, NDWI, NDBI, NDMI) on-the-fly từ band gốc.
- Normalize per-channel bằng stats train (xem ``compute_stats.py``).

⚠️ BAND ORDER (đã verify thực nghiệm trên dữ liệu thật, KHÔNG theo thứ tự torchgeo):
File TIFF lưu theo thứ tự numeric L1C với **B8A ở CUỐI** (index 12):

    idx:  0    1    2    3    4    5    6    7    8    9    10   11   12
    band: B01  B02  B03  B04  B05  B06  B07  B08  B09  B10  B11  B12  B8A

Hệ quả quan trọng: B11 (SWIR1) ở index 10 (không phải 11). Dùng sai index sẽ
làm hỏng NDBI/NDMI mà không báo lỗi. Xem CLAUDE.md mục 4.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

# --- Lớp ---------------------------------------------------------------------
# Sắp xếp alphabet (khớp thứ tự os.listdir trên các folder class) và cố định.
CLASSES: list[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# --- Thứ tự band trong file TIFF (đã verify, xem docstring) -------------------
ALL_BANDS: list[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
]
BAND_TO_IDX: dict[str, int] = {b: i for i, b in enumerate(ALL_BANDS)}

# Presets tiện dùng cho cấu trúc thí nghiệm 3 tầng (CLAUDE.md mục 6).
RGB_BANDS = ["B04", "B03", "B02"]                                   # Red, Green, Blue
BANDS_10M = ["B02", "B03", "B04", "B08"]                            # T2-H5a
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]              # phần 20m
BANDS_ATMOSPHERIC = ["B01", "B09", "B10"]                           # phần 60m

# --- Spectral indices --------------------------------------------------------
# Mỗi index = (band_pos, band_neg) -> (pos - neg) / (pos + neg + eps).
INDEX_DEFS: dict[str, tuple[str, str]] = {
    "NDVI": ("B08", "B04"),  # Vegetation
    "NDWI": ("B03", "B08"),  # Water
    "NDBI": ("B11", "B08"),  # Built-up
    "NDMI": ("B08", "B11"),  # Moisture
}
ALL_INDICES: list[str] = list(INDEX_DEFS)


def read_patch(path: str) -> np.ndarray:
    """Đọc 1 file .tif -> ndarray float32 shape (13, H, W) theo thứ tự ALL_BANDS."""
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim != 3:
        raise ValueError(f"{path}: kỳ vọng 3 chiều, nhận {arr.shape}")
    # tifffile trả (H, W, 13); chuyển về (13, H, W).
    if arr.shape[-1] == len(ALL_BANDS):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] != len(ALL_BANDS):
        raise ValueError(f"{path}: không tìm thấy trục 13 bands trong {arr.shape}")
    return arr.astype(np.float32)


def compute_indices(
    full_bands: np.ndarray, indices: Sequence[str], eps: float = 1e-8
) -> np.ndarray:
    """Tính spectral indices từ tensor đầy đủ 13 band (shape (13, H, W)).

    Trả về ndarray (len(indices), H, W) float32. Index luôn tính từ band GỐC
    (trước normalize) để giữ đúng tỉ lệ phản xạ.
    """
    out = []
    for name in indices:
        if name not in INDEX_DEFS:
            raise KeyError(f"Index không hỗ trợ: {name}. Có: {ALL_INDICES}")
        pos_name, neg_name = INDEX_DEFS[name]
        pos = full_bands[BAND_TO_IDX[pos_name]]
        neg = full_bands[BAND_TO_IDX[neg_name]]
        out.append((pos - neg) / (pos + neg + eps))
    return np.stack(out, axis=0).astype(np.float32) if out else np.empty((0, *full_bands.shape[1:]), np.float32)


def load_split(split_path: str) -> list[tuple[str, int]]:
    """Đọc file split (mỗi dòng: ``relpath<TAB>label_idx``) -> list (relpath, label)."""
    samples = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath, label = line.split("\t")
            samples.append((relpath, int(label)))
    return samples


class EuroSATDataset(Dataset):
    """Dataset EuroSAT 13-band cấu hình được theo band + indices.

    Pipeline mỗi mẫu:
        1. Đọc full (13, 64, 64) raw.
        2. Tính indices (nếu có) từ full raw bands.
        3. Chọn các band yêu cầu theo tên -> giữ đúng thứ tự ``bands``.
        4. Ghép [bands đã chọn] + [indices] -> (C, 64, 64).
        5. Normalize per-channel (nếu có ``norm_mean``/``norm_std``).
        6. Áp ``transform`` (augmentation) nếu có.

    Channel cuối cùng có thứ tự = ``bands`` rồi tới ``indices`` (xem ``channel_names``).
    """

    def __init__(
        self,
        root: str,
        samples: list[tuple[str, int]],
        bands: Sequence[str] | None = None,
        indices: Sequence[str] | None = None,
        norm_mean: Sequence[float] | None = None,
        norm_std: Sequence[float] | None = None,
        transform=None,
        eps: float = 1e-8,
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.bands = list(bands) if bands is not None else list(ALL_BANDS)
        self.indices = list(indices) if indices is not None else []
        self.transform = transform
        self.eps = eps

        unknown = set(self.bands) - set(ALL_BANDS)
        if unknown:
            raise ValueError(f"Band không tồn tại: {sorted(unknown)}. Có: {ALL_BANDS}")

        self._band_pos = [BAND_TO_IDX[b] for b in self.bands]

        # Vector normalize theo từng output channel (shape (C,1,1)).
        self.norm_mean = self._as_chw(norm_mean) if norm_mean is not None else None
        self.norm_std = self._as_chw(norm_std) if norm_std is not None else None

    # -- API --------------------------------------------------------------
    @property
    def channel_names(self) -> list[str]:
        """Tên từng channel đầu ra, theo thứ tự: bands rồi indices."""
        return list(self.bands) + list(self.indices)

    @property
    def n_channels(self) -> int:
        return len(self.bands) + len(self.indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        relpath, label = self.samples[i]
        full = read_patch(os.path.join(self.root, relpath))          # (13, H, W)

        chans = [full[self._band_pos]]                                # bands đã chọn
        if self.indices:
            chans.append(compute_indices(full, self.indices, self.eps))
        x = np.concatenate(chans, axis=0)                             # (C, H, W)

        x = torch.from_numpy(x)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / self.norm_std
        if self.transform is not None:
            x = self.transform(x)
        return x, label

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _as_chw(values: Sequence[float]) -> torch.Tensor:
        t = torch.tensor(list(values), dtype=torch.float32)
        return t.view(-1, 1, 1)

    @classmethod
    def from_split_file(
        cls, root: str, split_path: str, **kwargs
    ) -> "EuroSATDataset":
        """Khởi tạo dataset từ file split (xem ``create_splits.py``)."""
        return cls(root=root, samples=load_split(split_path), **kwargs)


def stats_to_channel_vectors(
    channel_stats: dict, bands: Sequence[str], indices: Sequence[str]
) -> tuple[list[float], list[float]]:
    """Lấy (mean, std) theo đúng thứ tự output channel từ channel_stats.json.

    ``channel_stats`` có dạng ``{"bands": {name: {mean, std}}, "indices": {...}}``.
    """
    mean, std = [], []
    for b in bands:
        mean.append(channel_stats["bands"][b]["mean"])
        std.append(channel_stats["bands"][b]["std"])
    for idx in indices:
        mean.append(channel_stats["indices"][idx]["mean"])
        std.append(channel_stats["indices"][idx]["std"])
    return mean, std
