from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

def write(path: str, content: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    print(f"updated: {path}")

def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    if new in content:
        print(f"ok: {path} already patched")
        return
    if old not in content:
        print(f"skip: pattern not found in {path}")
        return
    write(path, content.replace(old, new, 1))

def require_repo_root() -> None:
    required = ["README.md", "CLAUDE.md", "requirements.txt", "scripts/train.py", "src/models/resnet.py"]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("Run this script from the repository root. Missing: " + ", ".join(missing))

require_repo_root()

# 1) requirements.txt
req = read("requirements.txt")
if "tifffile" not in {line.strip() for line in req.splitlines()}:
    req = req.replace("# EuroSAT Refine — Python dependencies", "# EuroSAT Refine/Research — Python dependencies")
    req = req.replace("rasterio\n", "rasterio\ntifffile\n")
    write("requirements.txt", req)
else:
    print("ok: requirements.txt already has tifffile")

# 2) README.md
readme = '''# EuroSAT Refine → Research

Dự án Research-Based Learning trên dataset EuroSAT (Sentinel-2), đi theo chu trình:

```text
Read → Reproduce → Refine → Research → Report
```

## Trạng thái hiện tại

- **Reproduce:** DONE — RGB baseline tái lập gần paper gốc.
- **Refine:** DONE — kiểm chứng 13 spectral bands, spectral indices, augmentation/optimizer tuning.
- **Research:** chuẩn bị RQ1-A — low-label self-supervised learning.

## Kết luận Refine

Refine ban đầu kiểm chứng giả thuyết:

> 13 bands + spectral indices có thể vượt RGB baseline và cải thiện nhóm lớp khó.

Sau bug fix conv1 RGB-prior alignment, kết luận chính là:

- Winner cuối: **T1-R2 fixed = 13-band raw + AdamW default recipe**.
- Spectral indices **không cải thiện thêm** khi đã có 13 raw bands + conv1 alignment đúng.
- EuroSAT supervised 80/10/10 gần saturation; research value còn lại nằm ở:
  - low-label regime,
  - representation learning,
  - SSL.

## Research protocol

Research phase giữ nguyên global split:

```text
Train / Val / Test = 21600 / 2700 / 2700
```

Không chuyển sang literal 10/90 split. Chỉ subsample từ TRAIN để tạo label fractions 1% / 5% / 10% / 100%.

Toàn bộ context, decisions log, protocol freeze và guard rails nằm trong [CLAUDE.md](CLAUDE.md) và [updated.md](updated.md).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```text
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

## Research low-label splits

Tạo nested train subsets:

```bash
python -m scripts.create_label_fraction_splits --train-split dataset/splits/train.txt --out-dir dataset/splits --seed 42
```

Sau đó chạy Condition B baseline curve, ví dụ:

```bash
python -m scripts.train --config configs/research/rq1_B_1pct.yaml --seed 42
```
'''
write("README.md", readme)

# 3) CLAUDE.md targeted updates
old_pre = """### Preprocessing
1. Upsample 20m và 60m bands về 10m bằng bilinear → tensor (13, 64, 64).
2. Tính 4 indices → tensor (17, 64, 64).
3. Normalize per-channel với stats tính trên train set, lưu `stats/channel_stats.json`.
"""
new_pre = """### Preprocessing
1. Upsample 20m và 60m bands về 10m bằng bilinear → tensor (13, 64, 64).
2. **Refine-only:** nếu config yêu cầu indices, tính thêm NDVI/NDWI/NDBI/NDMI → tối đa tensor (17, 64, 64).
3. **Research default:** dùng **13 bands raw**, KHÔNG concat indices.
4. Normalize per-channel với stats tính trên TRAIN set tương ứng với input config, lưu `stats/channel_stats.json`.
"""
replace_once("CLAUDE.md", old_pre, new_pre)

old_cfg = '    "seed": seed,\n    "train_split": "80/20-stratified-seed42",\n'
new_cfg = '    "seed": seed,\n    "split_protocol": "80/10/10-stratified-seed42",\n    "train_split_file": "train.txt",\n    "val_split_file": "val.txt",\n    "test_split_file": "test.txt",\n    "label_fraction": null,  # Research only: 0.01 / 0.05 / 0.10 / 1.00\n'
replace_once("CLAUDE.md", old_cfg, new_cfg)

# 4) src/models/resnet.py guard
guard = '''    if pretrained and in_channels == 3 and bands is not None:
        band_list = list(bands)
        if band_list != list(RGB_BAND_NAMES):
            raise ValueError(
                "pretrained=True + in_channels==3 assumes RGB band order "
                f"{list(RGB_BAND_NAMES)}. Received bands={band_list}. "
                "For non-RGB 3-channel inputs, use in_channels != 3 with explicit "
                "adapter logic or set pretrained=False."
            )

'''
resnet = read("src/models/resnet.py")
if "pretrained=True + in_channels==3 assumes RGB band order" not in resnet:
    target = "    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None\n    model = models.resnet50(weights=weights)\n\n    if in_channels != 3:\n"
    replacement = "    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None\n    model = models.resnet50(weights=weights)\n\n" + guard + "    if in_channels != 3:\n"
    if target not in resnet:
        print("skip: target not found in src/models/resnet.py")
    else:
        write("src/models/resnet.py", resnet.replace(target, replacement, 1))
else:
    print("ok: src/models/resnet.py already has 3-channel guard")

# 5) wandb tier 4
wandb = read("src/utils/wandb_utils.py")
if '4: "research-rq1"' not in wandb:
    if '    3: "tier3-tuning",\n}' in wandb:
        wandb = wandb.replace('    3: "tier3-tuning",\n}', '    3: "tier3-tuning",\n    4: "research-rq1",\n}')
        write("src/utils/wandb_utils.py", wandb)
    else:
        print("skip: tier tag block not found in src/utils/wandb_utils.py")
else:
    print("ok: src/utils/wandb_utils.py already has tier 4")

# 6) scripts/train.py support custom split files
train = read("scripts/train.py")
if "--train-split" not in train:
    old_block = '''    if batch_aug: aug_desc += f"+{batch_aug}(a={batch_aug_alpha},p={batch_aug_p})"

    def split_samples(name):
        s = load_split(f"{args.splits_dir}/{name}.txt")
        return s[: args.limit] if args.limit else s

    common = dict(root=args.data_root, bands=bands, indices=indices,
                  norm_mean=norm_mean, norm_std=norm_std,
                  batch_size=batch_size, num_workers=num_workers, seed=seed)
    train_loader, train_ds = build_loader(split_samples("train"), train=True,
                                          spatial_aug=spatial_aug, **common)
    val_loader, _ = build_loader(split_samples("val"), train=False, **common)
    test_loader, _ = build_loader(split_samples("test"), train=False, **common)
'''
    new_block = '''    if batch_aug: aug_desc += f"+{batch_aug}(a={batch_aug_alpha},p={batch_aug_p})"

    split_cfg = cfg.get("splits", {}) or {}
    train_split_file = args.train_split or split_cfg.get("train") or "train.txt"
    val_split_file = args.val_split or split_cfg.get("val") or "val.txt"
    test_split_file = args.test_split or split_cfg.get("test") or "test.txt"
    label_fraction = split_cfg.get("label_fraction")
    subset_seed = split_cfg.get("subset_seed")
    subset_strategy = split_cfg.get("subset_strategy")

    def split_samples(filename):
        s = load_split(f"{args.splits_dir}/{filename}")
        return s[: args.limit] if args.limit else s

    common = dict(root=args.data_root, bands=bands, indices=indices,
                  norm_mean=norm_mean, norm_std=norm_std,
                  batch_size=batch_size, num_workers=num_workers, seed=seed)
    train_loader, train_ds = build_loader(split_samples(train_split_file), train=True,
                                          spatial_aug=spatial_aug, **common)
    val_loader, _ = build_loader(split_samples(val_split_file), train=False, **common)
    test_loader, _ = build_loader(split_samples(test_split_file), train=False, **common)
'''
    train = train.replace(old_block, new_block)
    old_cfg_runtime = '''            "epochs": epochs, "augmentation": aug_desc,
            "optimizer": tr.get("optimizer", "adamw"),
            "train_split": "80/10/10-stratified-seed42", "pretrained": pretrained,
'''
    new_cfg_runtime = '''            "epochs": epochs, "augmentation": aug_desc,
            "optimizer": tr.get("optimizer", "adamw"),
            "split_protocol": "80/10/10-stratified-seed42",
            "train_split_file": train_split_file,
            "val_split_file": val_split_file,
            "test_split_file": test_split_file,
            "label_fraction": label_fraction,
            "subset_seed": subset_seed,
            "subset_strategy": subset_strategy,
            "pretrained": pretrained,
'''
    train = train.replace(old_cfg_runtime, new_cfg_runtime)
    old_args = '''    p.add_argument("--splits-dir", default="dataset/splits")
    p.add_argument("--stats", default="stats/channel_stats.json")
'''
    new_args = '''    p.add_argument("--splits-dir", default="dataset/splits")
    p.add_argument("--train-split", default=None, help="override train split filename, e.g. train_1.txt")
    p.add_argument("--val-split", default=None, help="override val split filename")
    p.add_argument("--test-split", default=None, help="override test split filename")
    p.add_argument("--stats", default="stats/channel_stats.json")
'''
    train = train.replace(old_args, new_args)
    write("scripts/train.py", train)
else:
    print("ok: scripts/train.py already has custom split arguments")

# 7) create_label_fraction_splits.py
create_splits = '''"""Create nested low-label train subsets for EuroSAT Research RQ1.

This script implements the official protocol freeze:

- DO NOT regenerate global train/val/test splits.
- Read the existing TRAIN split only.
- Create stratified, nested subsets from TRAIN.
- Keep val/test unchanged for every Research experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_FRACTIONS: dict[str, float] = {
    "1": 0.01,
    "5": 0.05,
    "10": 0.10,
    "100": 1.00,
}

def read_split(path: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                relpath, label = line.split("\\t")
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_no}: expected relpath<TAB>label, got {line!r}"
                ) from exc
            samples.append((relpath, int(label)))
    if not samples:
        raise ValueError(f"{path} is empty")
    return samples

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def group_by_label(samples: list[tuple[str, int]]) -> dict[int, list[tuple[str, int]]]:
    groups: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        groups[sample[1]].append(sample)
    return dict(groups)

def nested_subsets(
    samples: list[tuple[str, int]],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, list[tuple[str, int]]]:
    rng = random.Random(seed)
    groups = group_by_label(samples)
    shuffled: dict[int, list[tuple[str, int]]] = {}
    for label, items in groups.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        shuffled[label] = items_copy

    out: dict[str, list[tuple[str, int]]] = {}
    for name, frac in fractions.items():
        if not (0 < frac <= 1.0):
            raise ValueError(f"Invalid fraction {name}={frac}; expected (0,1]")
        subset: list[tuple[str, int]] = []
        for label in sorted(shuffled):
            items = shuffled[label]
            k = len(items) if frac == 1.0 else max(1, round(len(items) * frac))
            subset.extend(items[:k])
        subset.sort(key=lambda x: (x[1], x[0]))
        out[name] = subset
    return out

def assert_nested(subsets: dict[str, list[tuple[str, int]]]) -> None:
    order = ["1", "5", "10", "100"]
    present = [x for x in order if x in subsets]
    for small, large in zip(present, present[1:]):
        missing = set(subsets[small]) - set(subsets[large])
        if missing:
            preview = sorted(missing)[:5]
            raise AssertionError(
                f"Subset train_{small}.txt is not nested in train_{large}.txt. "
                f"Missing examples: {preview}"
            )

def write_split(path: Path, samples: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\\n") as f:
        for relpath, label in samples:
            f.write(f"{relpath}\\t{label}\\n")

def class_counts(samples: list[tuple[str, int]]) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    for _, label in samples:
        counts[label] += 1
    return {str(label): counts[label] for label in sorted(counts)}

def parse_fractions(values: list[str] | None) -> dict[str, float]:
    if not values:
        return DEFAULT_FRACTIONS
    parsed: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --fraction {item!r}. Use NAME=VALUE, e.g. 1=0.01")
        name, value = item.split("=", 1)
        parsed[name] = float(value)
    return parsed

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", default="dataset/splits/train.txt", type=Path)
    parser.add_argument("--out-dir", default="dataset/splits", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fraction", action="append")
    args = parser.parse_args()

    fractions = parse_fractions(args.fraction)
    samples = read_split(args.train_split)
    subsets = nested_subsets(samples, fractions, args.seed)
    assert_nested(subsets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_train_split": str(args.train_split),
        "source_train_sha256": sha256_file(args.train_split),
        "seed": args.seed,
        "strategy": "stratified_nested_prefix_per_class",
        "fractions": fractions,
        "total_train_samples": len(samples),
        "source_class_counts": class_counts(samples),
        "outputs": {},
    }

    for name, subset in subsets.items():
        out_path = args.out_dir / f"train_{name}.txt"
        write_split(out_path, subset)
        metadata["outputs"][f"train_{name}.txt"] = {
            "n_samples": len(subset),
            "class_counts": class_counts(subset),
            "sha256": sha256_file(out_path),
        }
        print(f"Wrote {out_path} | n={len(subset)} | counts={class_counts(subset)}")

    meta_path = args.out_dir / "label_fraction_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {meta_path}")

if __name__ == "__main__":
    main()
'''
write("scripts/create_label_fraction_splits.py", create_splits)

# 8) Research configs
configs = {
    "1pct": ("RQ1-B-1", "train_1.txt", "0.01", "[42, 123, 2024, 7, 999]",
             "ImageNet-pretrained 13-band ResNet-50 giữ được performance thế nào khi chỉ có 1% labels?",
             "Performance giảm mạnh ở 1% labels; curve này là baseline supervised để SSL phải vượt.",
             "low-label degradation; report mean±std over 5 seeds"),
    "5pct": ("RQ1-B-5", "train_5.txt", "0.05", "[42, 123, 2024]",
             "ImageNet-pretrained 13-band ResNet-50 giữ được performance thế nào khi có 5% labels?",
             "ImageNet pretraining giúp hơn scratch nhưng vẫn còn khoảng trống cho domain-specific SSL.",
             "low-label degradation; report mean±std over 3 seeds"),
    "10pct": ("RQ1-B-10", "train_10.txt", "0.10", "[42, 123, 2024]",
              "ImageNet-pretrained 13-band ResNet-50 đạt gì ở 10% labels trong protocol freeze hiện đại?",
              "10% labels là điểm nối với low-data finding của paper gốc nhưng dùng fixed val/test.",
              "compare with paper concept (~75% at 10%) without reproducing literal 10/90 split"),
    "100pct": ("RQ1-B-100", "train_100.txt", "1.00", "[42, 123, 2024]",
               "Condition B full-label baseline có tái tạo winner Refine trong Research config không?",
               "100% labels phải gần T1-R2 fixed 98.85%; đây là sanity check cho Research pipeline.",
               "test/acc ~98.5-99.1%"),
}

for desc, (exp_id, split_file, frac, seeds, question, hypothesis, expected) in configs.items():
    yaml = f'''# Research RQ1-A — Condition B: ImageNet baseline, {desc}.
tier: 4
exp_id: {exp_id}
config_desc: imagenet-13bands-{desc}
input_type: 13bands

bands: all
indices: []

seeds: {seeds}

splits:
  train: {split_file}
  val: val.txt
  test: test.txt
  label_fraction: {frac}
  subset_seed: 42
  subset_strategy: stratified_nested

model:
  name: resnet50
  pretrained: true

training:
  lr: 1.0e-4
  weight_decay: 1.0e-4
  batch_size: 32
  epochs: 50
  patience: 10
  label_smoothing: 0.1
  max_grad_norm: 1.0
  num_workers: 4

wandb:
  question: "{question}"
  hypothesis: "{hypothesis}"
  baseline_ref: "Winner Refine T1-R2 fixed; Research protocol freeze 80/10/10"
  expected: "{expected}"
'''
    write(f"configs/research/rq1_B_{desc}.yaml", yaml)

print()
print("DONE.")
print("Next commands:")
print("python -m scripts.create_label_fraction_splits --train-split dataset/splits/train.txt --out-dir dataset/splits --seed 42")
print("python -m scripts.train --config configs/research/rq1_B_1pct.yaml --seed 42 --epochs 1 --limit 64 --device cpu --wandb-mode disabled --no-pretrained")
