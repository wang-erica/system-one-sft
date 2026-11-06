"""Supervised LoRA training for Clef's choice logits; checkpoint selection uses validation only."""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("flowers102", "food101"))
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--root", type=Path, help="dataset directory (default: repo data/)")
    parser.add_argument("--prompt", help="prompt ID; defaults to the frozen primary_prompt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, help="override the dataset config's seed")
    parser.add_argument("--limit", type=int, help="fixed-seed subset size for both train and validation")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "rank", "alpha", "limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a fresh directory")

    from dataclasses import replace
    from torch.utils.data import Subset
    from src.data import DEFAULT_ROOT, limit_indices, load_class_names, load_split
    from src.eval import environment, evaluate
    from src.metrics import PRIMARY_METRIC, compute_metrics
    from src.model import class_logits, get_prompt, load_config, load_model

    config = load_config(args.dataset)
    if args.seed is not None:
        config["seed"] = args.seed
    prompt = get_prompt(config, args.prompt)
    seed_everything(config["seed"])
    names = load_class_names(args.dataset)
    splits, indices = {}, {}
    for split in ("train", "val"):
        full = load_split(args.dataset, split, root=args.root or DEFAULT_ROOT)
        indices[split] = (limit_indices(len(full), args.limit, config["seed"])
                          if args.limit is not None else list(range(len(full))))
        if not indices[split]:
            raise ValueError(f"{split} split is empty")
        splits[split] = replace(full, data=Subset(full.data, indices[split]))

    settings = {key: getattr(args, key) for key in
                ("epochs", "batch_size", "learning_rate", "rank", "alpha", "dropout", "target_modules")}
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"dataset": args.dataset, "config": config, "prompt_id": prompt.prompt_id,
              "prompt": config["prompts"][prompt.prompt_id], "class_names": names,
              "settings": settings, "eval_batch_size": args.eval_batch_size,
              "seed": config["seed"], "limit": args.limit, "split_indices": indices,
              "data_root": str((args.root or DEFAULT_ROOT).resolve()), "device": args.device,
              "primary_metric": PRIMARY_METRIC[args.dataset],
              "started_at": datetime.now(timezone.utc).isoformat(), "environment": environment(),
              "status": "running"}
    log_path = args.output / "training.json"

    def save_log():
        log_path.write_text(json.dumps(record, indent=2) + "\n")

    save_log()
    started = time.perf_counter()
    try:
        clef = load_model(config, device=args.device)
        clef.model.eval()

        @torch.inference_mode()
        def scorer(images):
            return class_logits(clef, images, names, prompt).cpu().numpy()

        def validation(data):
            logits, labels = evaluate(data, scorer, list(range(len(data))),
                                      batch_size=args.eval_batch_size)
            return compute_metrics(labels, logits)

        # Check scoring on validation images before any weight updates.
        sanity = limit_indices(len(splits["val"]), 100, config["seed"])
        evaluate(splits["val"], scorer, sanity, batch_size=args.eval_batch_size)
        seed_everything(config["seed"])
        record["training"] = fit(clef, splits["train"], splits["val"], names, prompt,
                                  settings, args.output / "adapter", validation,
                                  PRIMARY_METRIC[args.dataset])
        logits, labels = evaluate(splits["val"], scorer, list(range(len(splits["val"]))),
                                  batch_size=args.eval_batch_size)
        np.savez_compressed(args.output / "validation.npz", logits=logits, labels=labels,
                            indices=np.asarray(indices["val"]), class_names=np.asarray(names))
        record["validation_metrics"] = compute_metrics(labels, logits)
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        save_log()
    print(f"Saved selected adapter to {args.output / 'adapter'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
