# system-one-sft

Fine-tuning System One decision models (Cloudflare Clef-flash / Clef) for image classification on
Oxford Flowers-102 and Food-101. CS 4644/7643 Deep Learning, Georgia Tech, Fall 2026.

## Setup

Requires a CUDA GPU with BF16 support (A100/H100/H200). Primary compute is the GT ICE cluster.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Results tables live in `results/summary.md`. Design decisions and open questions are logged in
`docs/decisions.md`.

## Training Flowers-102 and Food-101

Run from the repository root on a BF16-capable CUDA GPU. Prepare the datasets once:

```bash
python -m src.data download flowers102
python -m src.data make-food101-split --download
```

Start with a 100-image smoke check, then train on the full splits:

```bash
python -m src.train flowers102 --prompt p0 --epochs 1 --limit 100 --output outputs/flowers-smoke
python -m src.train flowers102 --prompt p0 --epochs 3 --output outputs/flowers-train
python -m src.train food101 --prompt p0 --epochs 3 --output outputs/food-train
```

`--prompt p0` explicitly chooses an experimental prompt. For final runs, select and freeze
`primary_prompt` using validation results, then omit `--prompt` to use it. Training uses
only train and validation, selects the adapter by the dataset's validation primary metric,
and saves `adapter/`, `training.json` (config, seed, subset indices, environment and history),
and `validation.npz` (selected-checkpoint logits, labels, class names and original indices).
The test split is never loaded by this command.

Each run needs a fresh output directory. `--limit N` takes fixed-seed random subsets of
both train and validation; omit it for full training. Use `--root` for a different data
directory. Batch size defaults to 1; other options include `--learning-rate`, `--rank`,
`--alpha`, `--dropout`, `--target-modules`, `--seed` and `--eval-batch-size`.
Run `python -m src.train --help` for the full interface.

If backpropagation runs out of GPU memory at batch size 1, try `--gradient-checkpointing`.
It uses non-reentrant activation checkpointing and disables the backbone's KV cache,
trading extra computation for lower activation memory. For example:

```bash
python -m src.train flowers102 --prompt p0 --epochs 1 --limit 100 --gradient-checkpointing --output outputs/flowers-smoke-gc
```

`--limit` reduces the number of examples, not the memory required for one training step.
Checkpointing does not guarantee that training fits a particular GPU; a larger-memory
GPU may still be needed. Training now displays a per-epoch progress bar and batch loss.

## Automated Kaggle experiments

```bash
python -m src.pipeline plan https://www.kaggle.com/competitions/COMPETITION --output outputs/plan
# Complete the generated YAML with prepared data and a published-solution adapter.
python -m src.pipeline run outputs/plan/experiment.yaml --output outputs/run
```

Runs a winning-solution baseline, zero-shot Clef, and validation-selected LoRA SFT on the
same held-out images, saving predictions and a comparison report. See
[the pipeline guide](docs/pipeline.md) for data preparation, adapter setup, and limitations.
