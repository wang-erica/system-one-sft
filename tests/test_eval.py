import json

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import Dataset

from src import eval as evaluation
from src.data import Split
from src.metrics import compute_metrics

NUM_CLASSES = 3


class ColorDataset(Dataset):
    """Image i has label i % 3, encoded in its red channel so the fake scorer can read it."""

    def __init__(self, size: int = 30) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index):
        label = index % NUM_CLASSES
        return Image.new("RGB", (4, 4), (label * 50, index, 0)), label


def fake_scorer(images):
    """Right on every image except those whose index (green channel) is a multiple of 5."""
    logits = np.zeros((len(images), NUM_CLASSES), dtype=np.float32)
    for row, image in enumerate(images):
        red, index, _ = image.getpixel((0, 0))
        label = red // 50
        guess = label if index % 5 else (label + 1) % NUM_CLASSES
        logits[row, guess] = 3.0
    return logits


@pytest.fixture
def config():
    return {
        "dataset": "flowers102",
        "seed": 0,
        "model": {"repo": "Cloudflare/clef-flash", "revision": "test"},
        "prompts": {"p0": {"state": "s", "instructions": "q", "option_description": None},
                    "p1": {"state": "s", "instructions": "q2", "option_description": None}},
        "primary_prompt": None,
    }


def split_of(size=30, split="val"):
    return Split("flowers102", split, ("a", "b", "c"), ColorDataset(size))


def test_run_writes_logits_and_run_log(tmp_path, config):
    record = evaluation.run(config, "val", "p0", fake_scorer, split_data=split_of(), batch_size=4,
                            results_dir=tmp_path, progress=False)

    saved = np.load(tmp_path / "raw" / f"{record['run_id']}.npz")
    assert saved["logits"].shape == (30, NUM_CLASSES) and saved["logits"].dtype == np.float32
    assert saved["labels"].tolist() == [i % NUM_CLASSES for i in range(30)]
    assert saved["indices"].tolist() == list(range(30))

    logged = json.loads((tmp_path / "runs" / f"{record['run_id']}.json").read_text())
    assert logged["metrics"]["top1_accuracy"] == pytest.approx(24 / 30)  # 6 multiples of 5 are wrong
    assert logged["metrics"] == compute_metrics(saved["labels"], saved["logits"])
    assert logged["primary_metric"] == "mean_per_class_accuracy"
    assert logged["prompt_id"] == "p0" and logged["prompt"] == config["prompts"]["p0"]
    assert logged["run_id"].startswith("flowers102_val_zeroshot_p0_")
    assert logged["environment"]["git_commit"]
    assert isinstance(logged["environment"]["git_dirty"], bool)
    assert len(logged["class_names_sha256"]) == 64


def test_limit_uses_fixed_seed_subset_and_records_indices(tmp_path, config):
    first = evaluation.run(config, "val", "p0", fake_scorer, split_data=split_of(), limit=7,
                           results_dir=tmp_path, progress=False)
    indices = np.load(tmp_path / "raw" / f"{first['run_id']}.npz")["indices"].tolist()
    assert len(indices) == 7 and indices == sorted(indices)
    assert "_limit7_" in first["run_id"]
    labels = np.load(tmp_path / "raw" / f"{first['run_id']}.npz")["labels"].tolist()
    assert labels == [index % NUM_CLASSES for index in indices]


def test_test_split_requires_primary_prompt(tmp_path, config):
    with pytest.raises(ValueError, match="choose primary_prompt"):
        evaluation.run(config, "test", "p0", fake_scorer, split_data=split_of(split="test"),
                       results_dir=tmp_path, progress=False)
    config["primary_prompt"] = "p1"
    with pytest.raises(ValueError, match="only allowed with primary_prompt"):
        evaluation.run(config, "test", "p0", fake_scorer, split_data=split_of(split="test"),
                       results_dir=tmp_path, progress=False)
    record = evaluation.run(config, "test", "p1", fake_scorer, split_data=split_of(split="test"),
                            results_dir=tmp_path, progress=False)
    assert record["split"] == "test"
    # The refused runs wrote nothing; only the allowed p1 run exists.
    assert [path.name for path in (tmp_path / "runs").iterdir()] == [f"{record['run_id']}.json"]
    assert len(list((tmp_path / "raw").iterdir())) == 1


def test_scorer_shape_is_checked(tmp_path, config):
    with pytest.raises(ValueError, match="shape"):
        evaluation.run(config, "val", "p0", lambda images: np.zeros((len(images), 2)),
                       split_data=split_of(), results_dir=tmp_path, progress=False)


def test_adapter_provenance_and_subset_seed(tmp_path, config):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    training = {"dataset": config["dataset"], "config": {"model": config["model"]},
                "prompt_id": "p0", "prompt": config["prompts"]["p0"], "seed": 42, "limit": 100}
    (tmp_path / "training.json").write_text(json.dumps(training))
    metadata = evaluation.adapter_metadata(adapter, config, "p0")
    assert config["seed"] == 42
    assert metadata["adapter_training_limit"] == 100
    with pytest.raises(ValueError, match="same prompt"):
        evaluation.adapter_metadata(adapter, config, "p1")
    training["dataset"] = "food101"
    (tmp_path / "training.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="different dataset"):
        evaluation.adapter_metadata(adapter, config, "p0")


def test_eval_cli_loads_adapter_and_marks_sft(tmp_path, monkeypatch, config):
    import sys
    import torch
    from types import SimpleNamespace
    from src import model
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    config["primary_prompt"] = "p0"
    monkeypatch.setattr(model, "load_config", lambda _: config)
    from src import data
    monkeypatch.setattr(data, "load_class_names", lambda _: ["a", "b", "c"])
    holder = torch.nn.Module()
    holder.language_model = torch.nn.Linear(2, 2)
    original = holder.language_model
    monkeypatch.setattr(model, "load_model", lambda *a, **kw: SimpleNamespace(model=holder))
    calls = []
    def load(base, path, is_trainable):
        assert base is original and path == str(adapter) and not is_trainable
        calls.append(path)
        return torch.nn.Linear(2, 2)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(model, "class_logits", lambda *args: torch.ones(1, 3))
    def run(config, split, prompt_id, scorer, **kwargs):
        assert split == "val" and prompt_id == "p0"
        assert kwargs["method"] == "sft"
        assert kwargs["extra"]["adapter"] == str(adapter.resolve())
        assert not holder.training
        assert scorer([None]).shape == (1, 3)
        return {"run_id": "test", "metrics": {}}
    monkeypatch.setattr(evaluation, "run", run)
    monkeypatch.setattr(sys, "argv", ["eval", "flowers102", "--split", "val", "--adapter", str(adapter)])
    evaluation.main()
    assert len(calls) == 1
