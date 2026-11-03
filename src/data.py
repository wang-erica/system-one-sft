"""Dataset loaders with official splits and class-name lists.

Images are returned as PIL images (no transforms), because the Clef processor does its own
resizing and normalization. Labels are integer class indices into ``class_names``.

Splits:
    flowers102: official train / val / test from torchvision.
    food101:    official test; official train divided into train and a stratified held-out val.
                The val image IDs are committed in ``configs/splits/food101_val.json`` (see
                ``make-food101-split`` below), so every run uses the same split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset
from torchvision.datasets import Flowers102, Food101

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSNAMES_DIR = REPO_ROOT / "configs" / "classnames"
FOOD101_VAL_SPLIT_FILE = REPO_ROOT / "configs" / "splits" / "food101_val.json"
DEFAULT_ROOT = REPO_ROOT / "data"

DATASETS = ("flowers102", "food101")
SPLITS = ("train", "val", "test")
NUM_CLASSES = {"flowers102": 102, "food101": 101}

FOOD101_VAL_PER_CLASS = 38  # ceil(5% of 750 training images per class)
FOOD101_VAL_SEED = 0


@dataclass(frozen=True)
class Split:
    """One split of one dataset. Indexing yields ``(PIL image, label)``."""

    dataset_name: str
    split: str
    class_names: tuple[str, ...]
    data: Dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        return self.data[index]


def load_class_names(dataset_name: str) -> list[str]:
    """Class names in label order, one per line in ``configs/classnames/<dataset>.txt``."""
    _check_name(dataset_name)
    names = (CLASSNAMES_DIR / f"{dataset_name}.txt").read_text().splitlines()
    if len(names) != NUM_CLASSES[dataset_name] or len(set(names)) != len(names):
        raise ValueError(f"{dataset_name}: expected {NUM_CLASSES[dataset_name]} unique class names")
    return names


def load_split(
    dataset_name: str,
    split: str,
    root: str | Path = DEFAULT_ROOT,
    limit: int | None = None,
    seed: int = 0,
    download: bool = False,
) -> Split:
    """Load an official split. ``limit`` keeps a fixed-seed random subset of that size."""
    _check_name(dataset_name)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    class_names = load_class_names(dataset_name)

    if dataset_name == "flowers102":
        data: Dataset = Flowers102(root, split=split, download=download)
    else:
        data = _load_food101(root, split, class_names, download)

    if limit is not None:
        data = Subset(data, limit_indices(len(data), limit, seed))
    return Split(dataset_name, split, tuple(class_names), data)


def _load_food101(root: str | Path, split: str, class_names: list[str], download: bool) -> Dataset:
    official = Food101(root, split="test" if split == "test" else "train", download=download)
    expected = [name.replace("_", " ") for name in official.classes]
    if expected != class_names:
        raise ValueError("configs/classnames/food101.txt does not match torchvision Food101.classes")
    if split == "test":
        return official

    val_ids = set(read_food101_val_split()["ids"])
    ids = food101_image_ids(official)
    keep_val = split == "val"
    indices = [index for index, image_id in enumerate(ids) if (image_id in val_ids) == keep_val]
    if keep_val and len(indices) != len(val_ids):
        raise ValueError("Food-101 val split file lists IDs that are not in the official train split")
    return Subset(official, indices)


def food101_image_ids(dataset: Food101) -> list[str]:
    """Image IDs (``"<class>/<image number>"``, as in ``meta/train.json``) in dataset order."""
    # torchvision keeps the file list private; the IDs are the last two path components.
    return [f"{path.parent.name}/{path.stem}" for path in dataset._image_files]


def make_food101_val_split(
    image_ids: list[str],
    labels: list[int],
    per_class: int = FOOD101_VAL_PER_CLASS,
    seed: int = FOOD101_VAL_SEED,
) -> list[str]:
    """Pick ``per_class`` image IDs from every class with a fixed seed. Returns sorted IDs."""
    by_class: dict[int, list[str]] = defaultdict(list)
    for image_id, label in zip(image_ids, labels):
        by_class[label].append(image_id)
    rng = random.Random(seed)
    chosen: list[str] = []
    for label in sorted(by_class):
        members = sorted(by_class[label])
        if len(members) < per_class:
            raise ValueError(f"class {label} has only {len(members)} images")
        chosen.extend(rng.sample(members, per_class))
    return sorted(chosen)


def read_food101_val_split() -> dict[str, Any]:
    if not FOOD101_VAL_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"{FOOD101_VAL_SPLIT_FILE} is missing. Generate it once with "
            "`python -m src.data make-food101-split` and commit it."
        )
    return json.loads(FOOD101_VAL_SPLIT_FILE.read_text())


def limit_indices(length: int, limit: int, seed: int = 0) -> list[int]:
    """Fixed-seed random subset of ``range(length)``, sorted. Used for ``--limit N`` checks."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= length:
        return list(range(length))
    return sorted(random.Random(seed).sample(range(length), limit))


def _check_name(dataset_name: str) -> None:
    if dataset_name not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset_name!r}")


def _make_food101_split_command(args: argparse.Namespace) -> None:
    official = Food101(args.root, split="train", download=args.download)
    ids = make_food101_val_split(food101_image_ids(official), list(official._labels))
    FOOD101_VAL_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOOD101_VAL_SPLIT_FILE.write_text(
        json.dumps(
            {"seed": FOOD101_VAL_SEED, "per_class": FOOD101_VAL_PER_CLASS, "ids": ids},
            indent=0,
        )
        + "\n"
    )
    print(f"wrote {len(ids)} val IDs to {FOOD101_VAL_SPLIT_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("make-food101-split", help="write the Food-101 val split file")
    split_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    split_parser.add_argument("--download", action="store_true")
    split_parser.set_defaults(func=_make_food101_split_command)

    download_parser = commands.add_parser("download", help="download a dataset into --root")
    download_parser.add_argument("dataset", choices=DATASETS)
    download_parser.add_argument("--root", default=str(DEFAULT_ROOT))
    download_parser.set_defaults(
        func=lambda args: [
            load_split(args.dataset, split, args.root, download=True)
            for split in (("train", "val", "test") if args.dataset == "flowers102" else ("test",))
        ]
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
