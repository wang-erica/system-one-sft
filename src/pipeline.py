"""Plan Kaggle experiments and run matched winner / zero-shot / SFT evaluations."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.kaggle import discover, parse_source


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def plan(url: str, output: Path, online: bool = True) -> dict:
    source = parse_source(url)
    output.mkdir(parents=True, exist_ok=False)
    evidence = discover(url) if online else {"source": source, "status": "not_queried"}
    evidence["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "discovery.json", evidence)
    config = {
        "source": source, "seed": 0, "primary_metric": "top1_accuracy",
        "data": {"root": "data/prepared", "class_names": ["REPLACE_WITH_CLASSES_IN_LABEL_ORDER"]},
        "model": {"repo": "Cloudflare/clef-flash", "revision": "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"},
        "prompt": {"state": "An image.", "instructions": "Which class is shown in the image?"},
        "baseline": {"factory": None, "solution_url": None, "checkpoint": None},
        "training": {"epochs": 3, "batch_size": 1, "learning_rate": 1e-4,
                     "rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        "batch_size": 1,
    }
    (output / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return evidence


def load_data(config: dict, base: Path):
    """Explicit ImageFolder train/val/test; all labels follow the declared class order."""
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Dataset

    names = config["data"]["class_names"]
    if len(names) < 2 or len(set(names)) != len(names) or any("REPLACE_" in n for n in names):
        raise ValueError("declare at least two unique class names in data.class_names")
    root = (base / config["data"]["root"]).resolve()
    splits, seen = {}, {}
    manifest = []

    class Remapped(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.mapping = [names.index(name) for name in dataset.classes]
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            image, label = self.dataset[index]
            return image, self.mapping[label]

    for split in ("train", "val", "test"):
        dataset = ImageFolder(root / split)
        if set(dataset.classes) != set(names):
            raise ValueError(f"{split}: class folders must exactly match data.class_names")
        for filename, label in dataset.samples:
            digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f"image content overlaps {seen[digest]} and {split}: {filename}")
            seen[digest] = split
            manifest.append({"split": split, "path": str(Path(filename).relative_to(root)),
                             "label": names.index(dataset.classes[label]), "sha256": digest})
        splits[split] = Remapped(dataset)
    return splits, manifest


def score_dataset(data, scorer, names, batch_size: int, limit: int | None = None):
    from src.data import Split
    from src.eval import evaluate
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch size and limit must be positive")
    indices = list(range(len(data)))
    # Validation/train smoke checks can be limited; the held-out comparison always uses all images.
    if limit is not None:
        indices = indices[:limit]
    logits, labels = evaluate(Split("custom", "evaluation", tuple(names), data), scorer,
                              indices, batch_size=batch_size)
    if not np.isfinite(logits).all():
        raise ValueError("scorer returned non-finite logits")
    return logits, labels, np.asarray(indices)


def run(config_path: Path, output: Path, device: str = "cuda", limit: int | None = None) -> dict:
    import torch
    from src.eval import environment
    from src.metrics import compute_metrics
    from src.model import Prompt, class_logits, load_model
    from src.train import fit, seed_everything

    config = yaml.safe_load(config_path.read_text())
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if config.get("batch_size", 1) < 1:
        raise ValueError("batch_size must be positive")
    parse_source(config["source"]["url"])
    metric = config["primary_metric"]
    if metric not in {"top1_accuracy", "mean_per_class_accuracy", "macro_f1"}:
        raise ValueError("primary_metric must be an accuracy or macro_f1")
    baseline = config["baseline"]
    if not baseline.get("factory") or not baseline.get("solution_url") or not baseline.get("checkpoint"):
        raise ValueError("baseline.factory, solution_url and checkpoint are required; a leaderboard row is not a runnable model")
    splits, manifest = load_data(config, config_path.resolve().parent)
    seed_everything(config["seed"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", manifest)
    started = time.perf_counter()
    names = config["data"]["class_names"]
    batch_size = config.get("batch_size", 1)
    records = {}
    write_json(output / "status.json", {"stage": "running", "started_at": datetime.now(timezone.utc).isoformat()})

    def save_scores(method, scorer):
        logits, labels, indices = score_dataset(splits["test"], scorer, names, batch_size)
        np.savez_compressed(output / f"{method}.npz", logits=logits, labels=labels,
                            indices=indices, class_names=np.asarray(names))
        records[method] = compute_metrics(labels, logits)
        write_json(output / "metrics.json", records)

    try:
        module, factory = baseline["factory"].split(":")
        scorer = getattr(importlib.import_module(module), factory)(baseline, names, device)
        save_scores("winning_solution", scorer)
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clef = load_model(config, device=device)
        clef.model.eval()
        prompt = Prompt("fixed", **config["prompt"])

        @torch.inference_mode()
        def clef_scorer(images):
            return class_logits(clef, images, names, prompt).cpu().numpy()

        # A validation sanity check precedes the expensive held-out runs.
        score_dataset(splits["val"], clef_scorer, names, batch_size, min(100, len(splits["val"])))
        save_scores("zeroshot", clef_scorer)

        def validation(data):
            logits, labels, _ = score_dataset(data, clef_scorer, names, batch_size, limit)
            return compute_metrics(labels, logits)

        train_data = splits["train"]
        if limit is not None:
            from torch.utils.data import Subset
            if limit < 1:
                raise ValueError("limit must be positive")
            train_data = Subset(train_data, range(min(limit, len(train_data))))
        seed_everything(config["seed"])
        training = fit(clef, train_data, splits["val"], names, prompt, config["training"],
                       output / "adapter", validation, metric)
        write_json(output / "training.json", training)
        save_scores("sft", clef_scorer)
        report = {"metrics": records, "primary_metric": metric,
                  "sft_minus_zeroshot": records["sft"][metric] - records["zeroshot"][metric],
                  "sft_minus_winning_solution": records["sft"][metric] - records["winning_solution"][metric],
                  "environment": environment(), "elapsed_seconds": time.perf_counter() - started,
                  "smoke_limit": limit}
        write_json(output / "report.json", report)
        rows = ["| Model | " + metric + " | Macro F1 | ECE |", "|---|---:|---:|---:|"]
        rows += [f"| {name} | {m[metric]:.4f} | {m['macro_f1']:.4f} | {m['ece']:.4f} |" for name, m in records.items()]
        (output / "summary.md").write_text("\n".join(rows) + "\n")
        write_json(output / "status.json", {"stage": "complete"})
        return report
    except Exception as error:
        write_json(output / "status.json", {"stage": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("url")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--offline", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("config", type=Path)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--limit", type=int, help="limit train/validation for smoke checks; test remains complete")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.url, args.output, not args.offline)
    else:
        result = run(args.config, args.output, args.device, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
