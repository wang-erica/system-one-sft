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

## Automated Kaggle experiments

```bash
python -m src.pipeline plan https://www.kaggle.com/competitions/COMPETITION --output outputs/plan
# Complete the generated YAML with prepared data and a published-solution adapter.
python -m src.pipeline run outputs/plan/experiment.yaml --output outputs/run
```

Runs a winning-solution baseline, zero-shot Clef, and validation-selected LoRA SFT on the
same held-out images, saving predictions and a comparison report. See
[the pipeline guide](docs/pipeline.md) for data preparation, adapter setup, and limitations.
