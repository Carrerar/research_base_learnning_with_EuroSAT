"""Create nested low-label train subsets for EuroSAT Research RQ1.

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
                relpath, label = line.split("\t")
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
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for relpath, label in samples:
            f.write(f"{relpath}\t{label}\n")

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
