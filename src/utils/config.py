"""Đọc YAML config cho mỗi run và resolve band preset.

Config quyết định TẤT CẢ (không hard-code trong script) — đúng nguyên tắc mục 7.
"""

from __future__ import annotations

import yaml

from src.data.eurosat_dataset import (
    ALL_BANDS,
    BANDS_10M,
    BANDS_20M,
    BANDS_ATMOSPHERIC,
    RGB_BANDS,
)

BAND_PRESETS: dict[str, list[str]] = {
    "rgb": RGB_BANDS,
    "all": ALL_BANDS,
    "10m": BANDS_10M,
    "20m": BANDS_20M,
    "atmospheric": BANDS_ATMOSPHERIC,
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_bands(spec) -> list[str]:
    """``spec`` là tên preset ('rgb'/'all'/...), 1 list band, hoặc list lẫn preset."""
    if isinstance(spec, str):
        if spec not in BAND_PRESETS:
            raise KeyError(f"Band preset không có: {spec}. Có: {list(BAND_PRESETS)}")
        return list(BAND_PRESETS[spec])
    bands: list[str] = []
    for item in spec:
        if item in BAND_PRESETS:
            bands.extend(BAND_PRESETS[item])
        else:
            bands.append(item)
    return bands
