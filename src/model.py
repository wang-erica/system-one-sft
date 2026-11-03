"""Clef loading, classification record building, and scoring.

Each image becomes one Clef record with a single ``choice`` question whose option IDs are the class
names. The release code returns option logits sorted by option ID, so logits are always mapped back
to class-index order through the returned ``option_ids`` (see ``class_order``), never by position.

Quick check on a GPU node (never use the test split for this):
    python -m src.model flowers102 --prompt p0 --split val --limit 4
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import yaml

from src.data import REPO_ROOT, load_class_names, load_split

CONFIGS_DIR = REPO_ROOT / "configs"
QUESTION_ID = "label"
# Release files needed to build records without the weights (tokenizer, processor, release code).
RELEASE_CODE_FILES = [
    "joint_schema_model.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "processor_config.json",
    "joint_head_config.json",
    "generation_config.json",
]


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    state: Any
    instructions: str
    option_description: str | None = None

    def criteria(self, class_names: list[str]) -> dict[str, str | None]:
        """Option ID (class name) -> description, or None when the prompt has no description."""
        if self.option_description is None:
            return {name: None for name in class_names}
        return {name: self.option_description.format(name=name) for name in class_names}


@dataclass
class Clef:
    model: Any
    processor: Any
    release: ModuleType

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def load_config(dataset_name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS_DIR / f"{dataset_name}.yaml").read_text())


def get_prompt(config: dict[str, Any], prompt_id: str | None = None) -> Prompt:
    """Prompt by id, or the config's frozen ``primary_prompt`` when no id is given."""
    prompt_id = prompt_id or config.get("primary_prompt")
    if prompt_id is None:
        raise ValueError(f"{config['dataset']}: no primary_prompt chosen yet; pass a prompt id")
    if prompt_id not in config["prompts"]:
        raise ValueError(f"{config['dataset']}: unknown prompt id {prompt_id!r}")
    fields = config["prompts"][prompt_id]
    return Prompt(prompt_id, fields["state"], fields["instructions"], fields.get("option_description"))


def build_record(
    image: Any,
    class_names: list[str],
    prompt: Prompt,
    record_id: str | None = None,
) -> dict[str, Any]:
    """One Clef record for one image. ``image=None`` gives a text-only record (for length checks)."""
    record: dict[str, Any] = {
        "state": prompt.state,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": prompt.instructions,
                "criteria": prompt.criteria(class_names),
            }
        },
    }
    if image is not None:
        record["images"] = [image]
    if record_id is not None:
        record["id"] = record_id
    return record


def class_order(option_ids: tuple[str, ...] | list[str], class_names: list[str]) -> list[int]:
    """For each class index, the position of that class in the model's option order."""
    if sorted(option_ids) != sorted(class_names) or len(set(option_ids)) != len(option_ids):
        raise ValueError("model option IDs do not match the class names")
    position = {option_id: index for index, option_id in enumerate(option_ids)}
    return [position[name] for name in class_names]


def release_path(
    config: dict[str, Any],
    cache_dir: str | Path | None = None,
    weights: bool = True,
    local_files_only: bool = False,
) -> Path:
    """Download (or find) the pinned release. ``weights=False`` fetches only code and tokenizer."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            config["model"]["repo"],
            revision=config["model"]["revision"],
            cache_dir=cache_dir,
            allow_patterns=None if weights else RELEASE_CODE_FILES,
            local_files_only=local_files_only,
        )
    )


def import_release(path: Path) -> ModuleType:
    """Import ``joint_schema_model`` from a release snapshot."""
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    module = importlib.import_module("joint_schema_model")
    # Compare unresolved paths: snapshot files are symlinks into the cache's blobs/ directory.
    if Path(module.__file__).absolute().parent != path.absolute():
        raise RuntimeError(f"joint_schema_model already imported from {module.__file__}")
    return module


def load_model(config: dict[str, Any], device: str = "cuda", cache_dir: str | Path | None = None) -> Clef:
    path = release_path(config, cache_dir=cache_dir)
    release = import_release(path)
    model, processor = release.load_release_model(path, device=device)
    return Clef(model, processor, release)


def class_logits(
    clef: Clef,
    images: list[Any],
    class_names: list[str],
    prompt: Prompt,
) -> torch.Tensor:
    """Logits of shape ``(len(images), num_classes)`` in class-index order, float32.

    Gradients flow through this function, so it is shared by evaluation and training.
    """
    tokenizer = clef.processor.tokenizer
    encoded = [
        clef.release.encode_record(
            tokenizer, build_record(image, class_names, prompt), processor=clef.processor
        )
        for image in images
    ]
    batch = clef.release.collate_records(encoded, tokenizer.pad_token_id, clef.device)
    outputs = clef.model(batch)

    order = class_order(encoded[0].questions[0].option_ids, class_names)
    rows = []
    for record, record_logits in zip(encoded, outputs):
        question = record.questions[0]
        if question.question_id != QUESTION_ID or question.option_ids != encoded[0].questions[0].option_ids:
            raise RuntimeError("records in a batch have different option orders")
        rows.append(record_logits[0][order])
    return torch.stack(rows).float()


@torch.inference_mode()
def predict_proba(clef: Clef, images: list[Any], class_names: list[str], prompt: Prompt) -> torch.Tensor:
    return class_logits(clef, images, class_names, prompt).softmax(dim=-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a few images and print top-3 predictions.")
    parser.add_argument("dataset", choices=("flowers102", "food101"))
    parser.add_argument("--prompt", default=None, help="prompt id (default: the config's primary_prompt)")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.dataset)
    prompt = get_prompt(config, args.prompt)
    class_names = load_class_names(args.dataset)
    loaded = load_split(args.dataset, args.split, limit=args.limit, seed=config["seed"])
    images, labels = zip(*(loaded[index] for index in range(len(loaded))))

    clef = load_model(config, device=args.device)
    probabilities = predict_proba(clef, list(images), class_names, prompt)
    for label, row in zip(labels, probabilities):
        top = row.topk(3)
        predictions = ", ".join(
            f"{class_names[index]} ({value:.3f})" for value, index in zip(top.values.tolist(), top.indices.tolist())
        )
        print(f"true: {class_names[label]:<28} predicted: {predictions}")


if __name__ == "__main__":
    main()
