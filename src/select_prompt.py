"""Rank prompt candidates by their zero-shot validation runs and report the best one.

    python -m src.select_prompt flowers102

Only runs that match the selection protocol count: zero-shot, validation split, the pinned model
revision, and the dataset's selection subset (full val for Flowers-102, the fixed-seed 1,000-image
subset for Food-101). This command only reports; setting ``primary_prompt`` in the config is a
deliberate manual edit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval import RESULTS_DIR
from src.metrics import PRIMARY_METRIC

SELECTION_LIMIT = {"flowers102": None, "food101": 1000}


def load_runs(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((results_dir / "runs").glob("*.json"))]


def select_prompt(config: dict[str, Any], runs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return the best prompt id and one row per candidate, best first.

    Uses the most recent qualifying run per prompt. Ties go to the lower prompt id.
    """
    dataset = config["dataset"]
    qualifying = [
        run
        for run in runs
        if run["dataset"] == dataset
        and run["split"] == "val"
        and run["method"] == "zeroshot"
        and run["limit"] == SELECTION_LIMIT[dataset]
        and run["model"]["revision"] == config["model"]["revision"]
        and run["prompt_id"] in config["prompts"]
    ]
    latest: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for run in sorted(qualifying, key=lambda run: run["started_at"]):
        latest[run["prompt_id"]] = run
        counts[run["prompt_id"]] = counts.get(run["prompt_id"], 0) + 1

    missing = [prompt_id for prompt_id in config["prompts"] if prompt_id not in latest]
    if missing:
        raise ValueError(f"{dataset}: no qualifying validation run for {', '.join(missing)}")

    metric = PRIMARY_METRIC[dataset]
    order = list(config["prompts"])
    rows = sorted(
        (
            {
                "prompt_id": prompt_id,
                metric: run["metrics"][metric],
                "macro_f1": run["metrics"]["macro_f1"],
                "ece": run["metrics"]["ece"],
                "run_id": run["run_id"],
                "runs_found": counts[prompt_id],
            }
            for prompt_id, run in latest.items()
        ),
        key=lambda row: (-row[metric], order.index(row["prompt_id"])),
    )
    return rows[0]["prompt_id"], rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=tuple(PRIMARY_METRIC))
    args = parser.parse_args()

    from src.model import load_config

    config = load_config(args.dataset)
    try:
        best, rows = select_prompt(config, load_runs())
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")
    metric = PRIMARY_METRIC[args.dataset]
    print(f"{'prompt':<8}{metric:>26}{'macro_f1':>10}{'ece':>8}  run")
    for row in rows:
        note = f"  ({row['runs_found']} runs, latest used)" if row["runs_found"] > 1 else ""
        print(f"{row['prompt_id']:<8}{row[metric]:>26.4f}{row['macro_f1']:>10.4f}{row['ece']:>8.4f}  {row['run_id']}{note}")
    if config.get("primary_prompt"):
        print(f"\nprimary_prompt is already frozen as {config['primary_prompt']!r}; not changing it.")
    else:
        print(f"\nBest: {best}. To freeze it, set `primary_prompt: {best}` in configs/{args.dataset}.yaml.")


if __name__ == "__main__":
    main()
