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
