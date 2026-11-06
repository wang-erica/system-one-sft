"""Exercise real optimizer updates and validation checkpoint restoration with a tiny backbone."""
import sys
from types import SimpleNamespace

import torch

from src import model, train


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
