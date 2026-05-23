"""Tạo train/val/test split stratified 80/10/10, seed 42.

Quét toàn bộ ``dataset/allbands/<Class>/*.tif``, chia stratified theo lớp và ghi
ra ``dataset/splits/{train,val,test}.txt``. Mỗi dòng: ``relpath<TAB>label_idx``
với ``relpath`` dạng ``<Class>/<file>.tif``.

Split cố định bằng seed 42 (CLAUDE.md mục 5). In ra SHA-256 của mỗi file split
để verify không leak / không đổi giữa các lần chạy (checklist mục 8).

Cách chạy:
    python -m src.data.create_splits \
        --data-root dataset/allbands --out-dir dataset/splits --seed 42
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

from sklearn.model_selection import train_test_split

# Windows console (cp1252) không in được tiếng Việt -> ép UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.eurosat_dataset import CLASS_TO_IDX, CLASSES

# 80 / 10 / 10
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def gather_samples(data_root: str) -> tuple[list[str], list[int]]:
    """Trả về (relpaths, labels) cho mọi .tif, sắp xếp ổn định."""
    relpaths, labels = [], []
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(data_root, cls, "*.tif")))
        if not files:
            raise FileNotFoundError(f"Không thấy .tif nào trong {os.path.join(data_root, cls)}")
        for f in files:
            relpaths.append(f"{cls}/{os.path.basename(f)}")
            labels.append(CLASS_TO_IDX[cls])
    return relpaths, labels


def make_splits(relpaths: list[str], labels: list[int], seed: int):
    """Stratified split 80/10/10. Trả về 3 list (relpath, label)."""
    idx = list(range(len(relpaths)))
    # Bước 1: tách test (10%).
    trainval_idx, test_idx = train_test_split(
        idx, test_size=TEST_FRACTION, stratify=labels, random_state=seed
    )
    # Bước 2: tách val từ phần còn lại sao cho val = 10% TỔNG.
    val_rel = VAL_FRACTION / (1.0 - TEST_FRACTION)
    tv_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_rel, stratify=tv_labels, random_state=seed
    )

    def pick(ids):
        return sorted((relpaths[i], labels[i]) for i in ids)

    return pick(train_idx), pick(val_idx), pick(test_idx)


def write_split(path: str, samples: list[tuple[str, int]]) -> str:
    lines = [f"{rel}\t{lab}\n" for rel, lab in samples]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    h = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="dataset/allbands")
    p.add_argument("--out-dir", default="dataset/splits")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    relpaths, labels = gather_samples(args.data_root)
    print(f"Tổng {len(relpaths)} ảnh trên {len(CLASSES)} lớp.")

    train, val, test = make_splits(relpaths, labels, args.seed)

    # Kiểm tra không leak: 3 tập rời nhau, hợp lại = toàn bộ.
    s_train = {r for r, _ in train}
    s_val = {r for r, _ in val}
    s_test = {r for r, _ in test}
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test), "LEAK!"
    assert len(s_train | s_val | s_test) == len(relpaths), "Mất mẫu khi split!"

    for name, samples in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.out_dir, f"{name}.txt")
        h = write_split(path, samples)
        print(f"  {name:5s}: {len(samples):6d} mẫu  sha256={h[:16]}…  -> {path}")

    print(f"Seed={args.seed}. Tỉ lệ ~ {len(train)/len(relpaths):.2f}/"
          f"{len(val)/len(relpaths):.2f}/{len(test)/len(relpaths):.2f}")


if __name__ == "__main__":
    main()
