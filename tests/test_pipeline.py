import json
import subprocess

import pytest
import yaml

from src.kaggle import discover, parse_source
from src.pipeline import plan


@pytest.mark.parametrize("path", ["competitions/flowers", "c/flowers/data"])
def test_competition_url(path):
    assert parse_source(f"https://www.kaggle.com/{path}?x=1")["ref"] == "flowers"


@pytest.mark.parametrize("url", ["https://evil.test/c/flowers", "http://kaggle.com/c/x",
                                  "https://kaggle.com/c/--help", "https://kaggle.com/code/a/b"])
def test_invalid_source(url):
    with pytest.raises(ValueError):
        parse_source(url)


def test_discovery_preserves_server_order_and_evidence():
    def runner(command, **kwargs):
        assert command[-2:] == ["--show", "--csv"]
        return subprocess.CompletedProcess(command, 0, "teamName,score\nwinner,0.1\nsecond,0.2\n")
    result = discover("https://kaggle.com/c/example", runner)
    assert result["leader"]["teamName"] == "winner"
    assert result["solution_status"].startswith("requires")
    assert "second" in result["leaderboard_csv"]


def test_dataset_has_no_winner():
    result = discover("https://kaggle.com/datasets/owner/images",
                      lambda *a, **kw: pytest.fail("datasets have no leaderboard"))
    assert result["leader"] is None


def test_plan_offline_and_no_overwrite(tmp_path):
    output = tmp_path / "plan"
    plan("https://kaggle.com/c/flowers", output, online=False)
    config = yaml.safe_load((output / "experiment.yaml").read_text())
    assert config["baseline"]["factory"] is None
    assert json.loads((output / "discovery.json").read_text())["status"] == "not_queried"
    with pytest.raises(FileExistsError):
        plan("https://kaggle.com/c/flowers", output, online=False)


def prepared_config(tmp_path):
    from PIL import Image
    root = tmp_path / "images"
    for split_index, split in enumerate(("train", "val", "test")):
        for label, name in enumerate(("zebra", "ant")):
            folder = root / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8), (label * 100, split_index * 50, 10)).save(folder / "image.png")
    return {"data": {"root": "images", "class_names": ["zebra", "ant"]}}


def test_data_mapping_and_leakage(tmp_path):
    import shutil
    from src.pipeline import load_data
    config = prepared_config(tmp_path)
    splits, manifest = load_data(config, tmp_path)
    # ImageFolder sorts 'ant' first; declared class order is the reverse.
    assert splits["train"][0][1] == 1
    assert len(manifest) == 6
    shutil.copy(tmp_path / "images/train/ant/image.png", tmp_path / "images/test/ant/image.png")
    with pytest.raises(ValueError, match="overlaps"):
        load_data(config, tmp_path)


def test_full_pipeline_with_fake_clef_and_training(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from src import model, train, pipeline
    config = prepared_config(tmp_path)
    config.update({"source": {"url": "https://kaggle.com/c/example"}, "seed": 0,
                   "primary_metric": "top1_accuracy", "model": {},
                   "prompt": {"state": "s", "instructions": "classify"},
                   "baseline": {"factory": "fake_winner:create", "checkpoint": "test",
                                "solution_url": "https://example.org/solution"},
                   "training": {}, "batch_size": 2})
    def predictions(images):
        return np.asarray([[3, 0] if im.getpixel((0, 0))[0] == 0 else [0, 3] for im in images])
    import sys
    monkeypatch.setitem(sys.modules, "fake_winner", SimpleNamespace(create=lambda *a: predictions))
    fake = SimpleNamespace(model=SimpleNamespace(eval=lambda: None), trained=False)
    monkeypatch.setattr(model, "load_model", lambda *a, **kw: fake)
    monkeypatch.setattr(model, "class_logits", lambda c, images, *a: torch.tensor(
        predictions(images) if c.trained else predictions(images)[:, ::-1].copy()))
    def fit(clef, train_data, val_data, names, prompt, settings, output, score, metric):
        assert len(train_data) == 1  # smoke limit affects training, not test
        clef.trained = True
        assert score(val_data)[metric] == 1
        return {"best_validation_score": 1}
    monkeypatch.setattr(train, "fit", fit)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "run"
    result = pipeline.run(config_path, output, device="cpu", limit=1)
    assert result["sft_minus_zeroshot"] == 1
    assert result["sft_minus_winning_solution"] == 0
    for method in ("winning_solution", "zeroshot", "sft"):
        saved = np.load(output / f"{method}.npz")
        assert saved["labels"].tolist() == [1, 0]
        assert saved["logits"].shape == (2, 2)
    assert json.loads((output / "status.json").read_text())["stage"] == "complete"
