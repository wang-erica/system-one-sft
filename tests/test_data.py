from collections import Counter

import pytest

from src import data

FLOWERS_DIR = data.DEFAULT_ROOT / "flowers-102"
needs_flowers = pytest.mark.skipif(
    not (FLOWERS_DIR / "jpg").is_dir() or not (FLOWERS_DIR / "setid.mat").exists(),
    reason="Flowers-102 not downloaded",
)


@pytest.mark.parametrize("dataset_name", data.DATASETS)
def test_class_names(dataset_name):
    names = data.load_class_names(dataset_name)
    assert len(names) == data.NUM_CLASSES[dataset_name]
    assert len(set(names)) == len(names)
    assert all(name == name.strip() and name and "_" not in name for name in names)


def test_class_names_known_positions():
    flowers = data.load_class_names("flowers102")
    assert flowers[0] == "pink primrose"
    assert flowers[53] == "sunflower"
    assert flowers[-1] == "blackberry lily"
    food = data.load_class_names("food101")
    assert food == sorted(food)
    assert food[0] == "apple pie" and food[-1] == "waffles"


def _synthetic_food_train():
    ids, labels = [], []
    for label in range(101):
        for number in range(750):
            ids.append(f"class{label:03d}/{100000 + number}")
            labels.append(label)
    return ids, labels


def test_food101_val_split_is_stratified_and_deterministic():
    ids, labels = _synthetic_food_train()
    split = data.make_food101_val_split(ids, labels)
    assert len(split) == 101 * data.FOOD101_VAL_PER_CLASS
    assert len(set(split)) == len(split)
    assert set(split) <= set(ids)
    per_class = Counter(image_id.split("/")[0] for image_id in split)
    assert set(per_class.values()) == {data.FOOD101_VAL_PER_CLASS}
    assert split == data.make_food101_val_split(ids, labels)
    assert split != data.make_food101_val_split(ids, labels, seed=1)


def test_food101_val_split_ignores_input_order():
    ids, labels = _synthetic_food_train()
    reversed_split = data.make_food101_val_split(ids[::-1], labels[::-1])
    assert reversed_split == data.make_food101_val_split(ids, labels)


def test_limit_indices():
    indices = data.limit_indices(1000, 100, seed=0)
    assert len(indices) == 100 and indices == sorted(set(indices))
    assert indices == data.limit_indices(1000, 100, seed=0)
    assert data.limit_indices(10, 100) == list(range(10))


def test_rejects_unknown_names():
    with pytest.raises(ValueError):
        data.load_class_names("cifar10")
    with pytest.raises(ValueError):
        data.load_split("flowers102", "validation")


@needs_flowers
@pytest.mark.parametrize("split, size", [("train", 1020), ("val", 1020), ("test", 6149)])
def test_flowers102_official_splits(split, size):
    loaded = data.load_split("flowers102", split)
    assert len(loaded) == size
    counts = Counter(loaded.data._labels)
    assert sorted(counts) == list(range(102))
    if split != "test":
        assert set(counts.values()) == {10}


@needs_flowers
def test_flowers102_limit_returns_pil_images():
    loaded = data.load_split("flowers102", "test", limit=5)
    assert len(loaded) == 5
    image, label = loaded[0]
    assert image.mode == "RGB" and 0 <= label < 102
