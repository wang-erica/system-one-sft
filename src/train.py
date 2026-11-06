"""Supervised LoRA training for Clef's choice logits; checkpoint selection uses validation only."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(clef, train_data, val_data, class_names, prompt, settings: dict,
        output: Path, score: Callable, primary_metric: str) -> dict:
    from peft import LoraConfig, get_peft_model
    from src.model import class_logits

    epochs = int(settings.get("epochs", 3))
    batch_size = int(settings.get("batch_size", 1))
    learning_rate = float(settings.get("learning_rate", 1e-4))
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    # Adapt the backbone only. Clef's release forward supports a PEFT language_model.
    for parameter in clef.model.parameters():
        parameter.requires_grad_(False)
    clef.model.language_model = get_peft_model(
        clef.model.language_model,
        LoraConfig(r=int(settings.get("rank", 8)), lora_alpha=int(settings.get("alpha", 16)),
                   target_modules=settings.get("target_modules", ["q_proj", "v_proj"]),
                   lora_dropout=float(settings.get("dropout", 0.05)), bias="none"),
    )
    parameters = [p for p in clef.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=list)
    best, history = -float("inf"), []
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(epochs):
        clef.model.train()
        total_loss = 0.0
        for batch in loader:
            images, labels = zip(*batch)
            optimizer.zero_grad(set_to_none=True)
            logits = class_logits(clef, list(images), class_names, prompt)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels, device=logits.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        clef.model.eval()
        metrics = score(val_data)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_data), "validation": metrics})
        if metrics[primary_metric] > best:
            best = metrics[primary_metric]
            clef.model.language_model.save_pretrained(output)
    # Restore the selected adapter before any test evaluation.
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    set_peft_model_state_dict(clef.model.language_model, load_peft_weights(str(output)))
    clef.model.eval()
    return {"history": history, "best_validation_score": best,
            "checkpoint": str(output), "trainable_parameters": sum(p.numel() for p in parameters)}
