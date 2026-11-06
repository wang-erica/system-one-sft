"""Exercise real optimizer updates and validation checkpoint restoration with a tiny backbone."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src import model, train
from src.data import Split


def test_training_updates_weights_and_restores_best_validation(tmp_path, monkeypatch):
    backbone = torch.nn.Linear(2, 2, bias=False)
    saved = {}

    def save(path):
        saved["weights"] = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
    backbone.save_pretrained = save

    def wrap(base, config):
        for parameter in base.parameters():
            parameter.requires_grad_(True)
        return base

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=lambda **kw: kw, get_peft_model=wrap,
        set_peft_model_state_dict=lambda model, state: model.load_state_dict(state)))
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", SimpleNamespace(
        load_peft_weights=lambda path: saved["weights"]))
    holder = torch.nn.Module()
    holder.language_model = backbone
    clef = SimpleNamespace(model=holder)
    monkeypatch.setattr(model, "class_logits", lambda c, images, *args:
                        c.model.language_model(torch.stack(images)))
    data = [(torch.tensor([1., 0.]), 0), (torch.tensor([0., 1.]), 1)]
    original = backbone.weight.detach().clone()
    metrics = iter([0.9, 0.2])
    result = train.fit(clef, data, data, ["a", "b"], None,
                       {"epochs": 2, "batch_size": 2, "learning_rate": 0.1},
                       tmp_path / "adapter", lambda _: {"top1_accuracy": next(metrics)},
                       "top1_accuracy")
    assert not torch.equal(original, backbone.weight)
    assert torch.equal(saved["weights"]["weight"], backbone.weight)
    assert result["best_validation_score"] == 0.9
    assert len(result["history"]) == 2
    assert not holder.training


@pytest.mark.parametrize("dataset", ["flowers102", "food101"])
def test_training_cli_official_splits_and_artifacts(tmp_path, monkeypatch, dataset):
    from src import data, eval as evaluation
    config = {"dataset": dataset, "seed": 7, "primary_prompt": "p0", "model": {},
              "prompts": {"p0": {"state": "s", "instructions": "classify"}}}
    monkeypatch.setattr(model, "load_config", lambda _: config)
    monkeypatch.setattr(data, "load_class_names", lambda _: ["a", "b"])
    calls = []
    samples = [(Image.new("RGB", (2, 2), (i, 0, 0)), i % 2) for i in range(12)]

    def load_split(name, split, root):
        calls.append((name, split, root))
        return Split(name, split, ("a", "b"), samples)
    monkeypatch.setattr(data, "load_split", load_split)
    monkeypatch.setattr(evaluation, "environment", lambda: {"git_commit": "test"})
    holder = torch.nn.Module()
    monkeypatch.setattr(model, "load_model", lambda *a, **kw: SimpleNamespace(model=holder))
    monkeypatch.setattr(model, "class_logits", lambda c, images, *a: torch.tensor([
        [3., 0.] if im.getpixel((0, 0))[0] % 2 == 0 else [0., 3.] for im in images]))

    def fit(clef, train_data, val_data, names, prompt, settings, output, score, primary_metric):
        assert len(train_data) == len(val_data) == 5
        assert settings["epochs"] == 1
        assert score(val_data)["top1_accuracy"] == 1
        assert primary_metric == ("mean_per_class_accuracy" if dataset == "flowers102" else "top1_accuracy")
        output.mkdir()
        return {"best_validation_score": 1, "history": []}
    monkeypatch.setattr(train, "fit", fit)
    output = tmp_path / "run"
    train.main([dataset, "--output", str(output), "--root", str(tmp_path / "data"),
                "--device", "cpu", "--epochs", "1", "--limit", "5"])
    assert [c[1] for c in calls] == ["train", "val"]
    record = json.loads((output / "training.json").read_text())
    assert record["status"] == "complete"
    assert record["prompt_id"] == "p0"
    assert record["seed"] == 7
    assert record["environment"]["git_commit"] == "test"
    saved = np.load(output / "validation.npz")
    assert saved["indices"].tolist() == data.limit_indices(12, 5, 7)
    assert saved["logits"].shape == (5, 2)
    # A repeated run cannot overwrite the adapter or log.
    with pytest.raises(SystemExit):
        train.main([dataset, "--output", str(output)])


@pytest.mark.parametrize("option,value", [("--epochs", "0"), ("--limit", "0"),
    ("--learning-rate", "nan"), ("--dropout", "1"), ("--eval-batch-size", "-1")])
def test_training_cli_rejects_invalid_settings(tmp_path, option, value):
    with pytest.raises(SystemExit):
        train.main(["flowers102", "--output", str(tmp_path / "run"), option, value])
    assert not (tmp_path / "run").exists()
