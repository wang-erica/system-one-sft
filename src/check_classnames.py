"""Save a grid with one image per class, captioned with its label index and class name.

Used to confirm the label index -> class name mapping by eye before any model run:
    python -m src.check_classnames flowers102
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import DATASETS, DEFAULT_ROOT, REPO_ROOT, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    loaded = load_split(args.dataset, args.split, root=args.root)
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index in range(len(loaded)):
        indices_by_label[_label(loaded, index)].append(index)

    rng = random.Random(args.seed)
    labels = sorted(indices_by_label)
    columns = 12
    rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2.3))
    for axis in axes.flat:
        axis.axis("off")
    for axis, label in zip(axes.flat, labels):
        image, image_label = loaded[rng.choice(indices_by_label[label])]
        assert image_label == label
        axis.imshow(image)
        axis.set_title(f"{label}: {loaded.class_names[label]}", fontsize=7)
    figure.tight_layout()

    out = args.out or REPO_ROOT / "outputs" / f"{args.dataset}_{args.split}_classnames.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=80)
    print(f"wrote {out}")


def _label(loaded, index: int) -> int:
    """Read a label without decoding the image, where the dataset allows it."""
    dataset = loaded.data
    if hasattr(dataset, "_labels"):
        return dataset._labels[index]
    if hasattr(dataset, "indices") and hasattr(dataset.dataset, "_labels"):
        return dataset.dataset._labels[dataset.indices[index]]
    return loaded[index][1]


if __name__ == "__main__":
    main()
