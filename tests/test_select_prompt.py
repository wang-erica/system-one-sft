import pytest

from src.select_prompt import select_prompt

REVISION = "rev"


def make_config(dataset="flowers102"):
    return {
        "dataset": dataset,
        "model": {"revision": REVISION},
        "prompts": {"p0": {}, "p1": {}, "p2": {}, "p3": {}},
    }


def make_run(prompt_id, score, dataset="flowers102", started="2026-11-03T00:00:00", **overrides):
    run = {
        "run_id": f"{dataset}_{prompt_id}_{started}",
        "dataset": dataset,
        "split": "val",
        "method": "zeroshot",
        "limit": None if dataset == "flowers102" else 1000,
        "model": {"revision": REVISION},
        "prompt_id": prompt_id,
        "started_at": started,
        "metrics": {"mean_per_class_accuracy": score, "top1_accuracy": score, "macro_f1": 0.0, "ece": 0.0},
    }
    run.update(overrides)
    return run


def test_picks_highest_primary_metric():
    runs = [make_run("p0", 0.5), make_run("p1", 0.7), make_run("p2", 0.6), make_run("p3", 0.1)]
    best, rows = select_prompt(make_config(), runs)
    assert best == "p1"
    assert [row["prompt_id"] for row in rows] == ["p1", "p2", "p0", "p3"]


def test_ties_go_to_lower_prompt_id():
    runs = [make_run("p0", 0.5), make_run("p1", 0.5), make_run("p2", 0.5), make_run("p3", 0.5)]
    assert select_prompt(make_config(), runs)[0] == "p0"


def test_ignores_runs_outside_the_selection_protocol():
    runs = [make_run(p, 0.5) for p in ("p0", "p1", "p2", "p3")] + [
        make_run("p3", 0.99, split="test"),
        make_run("p3", 0.99, limit=100),
        make_run("p3", 0.99, method="lora"),
        make_run("p3", 0.99, model={"revision": "other"}),
        make_run("p3", 0.99, dataset="food101"),
    ]
    assert select_prompt(make_config(), runs)[0] == "p0"


def test_food101_requires_the_1000_image_subset():
    full_val = [make_run(p, 0.5, dataset="food101", limit=None) for p in ("p0", "p1", "p2", "p3")]
    with pytest.raises(ValueError, match="no qualifying"):
        select_prompt(make_config("food101"), full_val)


def test_uses_latest_run_per_prompt_and_requires_every_candidate():
    runs = [make_run(p, 0.5) for p in ("p0", "p1", "p2")]
    with pytest.raises(ValueError, match="p3"):
        select_prompt(make_config(), runs)
    runs += [make_run("p3", 0.9, started="2026-11-03T00:00:00"), make_run("p3", 0.1, started="2026-11-04T00:00:00")]
    best, rows = select_prompt(make_config(), runs)
    assert best == "p0"
    assert next(row for row in rows if row["prompt_id"] == "p3")["runs_found"] == 2
