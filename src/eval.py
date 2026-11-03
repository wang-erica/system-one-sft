"""Evaluate a model on one split and save per-image logits plus a run log.

    python -m src.eval flowers102 --split val --prompt p0
    python -m src.eval food101 --split val --prompt p0 --limit 1000
    python -m src.eval flowers102 --split test        # requires primary_prompt in the config

Outputs, named by run ID:
    results/raw/<run_id>.npz     logits (float32, class-index order), labels, indices (gitignored)
    results/runs/<run_id>.json   metrics, config, prompt, git commit, versions, timing (committed)

``indices`` are positions in the full split, so a ``--limit`` run can be matched to images later.
The test split may only be evaluated with the config's frozen ``primary_prompt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import CLASSNAMES_DIR, REPO_ROOT, Split, limit_indices, load_split
from src.metrics import PRIMARY_METRIC, compute_metrics

RESULTS_DIR = REPO_ROOT / "results"

# Maps a batch of PIL images to logits of shape (batch, num_classes) in class-index order.
Scorer = Callable[[list[Any]], np.ndarray]


def check_split_allowed(config: dict[str, Any], split: str, prompt_id: str) -> None:
    if split != "test":
        return
    primary = config.get("primary_prompt")
    if primary is None:
        raise ValueError(f"{config['dataset']}: choose primary_prompt on validation before using the test split")
    if prompt_id != primary:
        raise ValueError(f"{config['dataset']}: test split only allowed with primary_prompt {primary!r}")


def make_run_id(dataset: str, split: str, method: str, prompt_id: str, limit: int | None, started: datetime) -> str:
    parts = [dataset, split, method, prompt_id]
    if limit is not None:
        parts.append(f"limit{limit}")
    parts.append(started.strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


def evaluate(
    split_data: Split,
    scorer: Scorer,
    indices: list[int],
    batch_size: int = 8,
    workers: int = 0,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Score ``indices`` of a split. Returns (logits float32 (N, C), labels int64 (N,))."""
    loader = DataLoader(
        Subset(split_data.data, indices),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=_keep_batch,
    )
    all_logits, all_labels = [], []
    for batch in tqdm(loader, disable=not progress, desc=f"{split_data.dataset_name}/{split_data.split}"):
        images, labels = zip(*batch)
        logits = np.asarray(scorer(list(images)), dtype=np.float32)
        if logits.shape != (len(images), len(split_data.class_names)):
            raise ValueError(f"scorer returned shape {logits.shape}")
        all_logits.append(logits)
        all_labels.extend(labels)
    return np.concatenate(all_logits), np.asarray(all_labels, dtype=np.int64)


def run(
    config: dict[str, Any],
    split: str,
    prompt_id: str,
    scorer: Scorer,
    split_data: Split | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    workers: int = 0,
    method: str = "zeroshot",
    results_dir: Path = RESULTS_DIR,
    extra: dict[str, Any] | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate, save logits and the run log, and return the run log."""
    dataset = config["dataset"]
    check_split_allowed(config, split, prompt_id)
    if split_data is None:
        split_data = load_split(dataset, split)
    indices = limit_indices(len(split_data), limit, config["seed"]) if limit else list(range(len(split_data)))

    started = datetime.now(timezone.utc)
    run_id = make_run_id(dataset, split, method, prompt_id, limit, started)
    raw_path = results_dir / "raw" / f"{run_id}.npz"
    run_path = results_dir / "runs" / f"{run_id}.json"
    if raw_path.exists() or run_path.exists():
        raise FileExistsError(f"outputs for run {run_id} already exist")
    timer = time.perf_counter()
    logits, labels = evaluate(split_data, scorer, indices, batch_size, workers, progress)
    eval_seconds = time.perf_counter() - timer

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, logits=logits, labels=labels, indices=np.asarray(indices, dtype=np.int64))

    metrics = compute_metrics(labels, logits)
    record = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split,
        "method": method,
        "prompt_id": prompt_id,
        "prompt": config["prompts"][prompt_id],
        "primary_prompt": config.get("primary_prompt"),
        "limit": limit,
        "seed": config["seed"],
        "primary_metric": PRIMARY_METRIC[dataset],
        "primary_metric_value": metrics[PRIMARY_METRIC[dataset]],
        "metrics": metrics,
        "model": config["model"],
        "class_names_sha256": _sha256(CLASSNAMES_DIR / f"{dataset}.txt"),
        "raw_predictions": _display_path(raw_path),
        "config": config,
        "started_at": started.isoformat(),
        "eval_seconds": round(eval_seconds, 2),
        "environment": environment(),
        **(extra or {}),
    }
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def environment() -> dict[str, Any]:
    """Git state, package versions and hardware, recorded with every run."""
    info: dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for package in ("torchvision", "transformers", "peft"):
        try:
            info[package] = __import__(package).__version__
        except ImportError:
            info[package] = None
    return info


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _display_path(path: Path) -> str:
    """Repo-relative path when inside the repo, so run logs don't record machine-specific paths."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keep_batch(batch: list[Any]) -> list[Any]:
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on one split.")
    parser.add_argument("dataset", choices=tuple(PRIMARY_METRIC))
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--prompt", default=None, help="prompt id (default: the config's primary_prompt)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from src.data import load_class_names
    from src.model import class_logits, get_prompt, load_config, load_model

    config = load_config(args.dataset)
    prompt = get_prompt(config, args.prompt)
    check_split_allowed(config, args.split, prompt.prompt_id)
    class_names = load_class_names(args.dataset)

    timer = time.perf_counter()
    clef = load_model(config, device=args.device)
    load_seconds = time.perf_counter() - timer

    @torch.inference_mode()
    def scorer(images: list[Any]) -> np.ndarray:
        return class_logits(clef, images, class_names, prompt).cpu().numpy()

    record = run(
        config,
        args.split,
        prompt.prompt_id,
        scorer,
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers,
        extra={"model_load_seconds": round(load_seconds, 2), "batch_size": args.batch_size},
    )
    print(json.dumps({"run_id": record["run_id"], **record["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
