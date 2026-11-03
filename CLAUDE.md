# Project: Fine-tuning System One models for image classification

CS 4644/7643 Deep Learning group project (Georgia Tech, Fall 2026).

## Research questions
1. Does supervised fine-tuning (SFT) significantly improve a System One model on image classification?
2. Can a fine-tuned System One model match or beat specialized CNN classifiers?
3. Which System One model (Clef-flash vs. Clef) improves most with SFT?

## Models
- **Primary:** Cloudflare Clef-flash (9B, Qwen3.5 backbone, Apache-2.0), Hugging Face `Cloudflare/clef-flash`.
- **Later / RQ3:** Cloudflare Clef (27B).
- These are *decision models*, not chat models: they score a fixed set of answer options in one pass and
  return a probability per option. Do not use `generate()` for classification.
- Load via the release code shipped in the model repo (`joint_schema_model.py`):
  `snapshot_download` → `load_release_model(path, device="cuda")` → `encode_record(...)` →
  `collate_records(...)` → `model(batch)` returns per-question logits over options.
- Images go in the record's `images` list (PIL images); pass `processor=processor` to `encode_record`.
- Classification uses a `choice` question with the class names as options. **Read the model card's
  input-format table for the exact field names before writing the record builder** — do not guess.
- Model card was tested with torch 2.11 and transformers 5.10.2. Requires a CUDA GPU with BF16 (A100/H100/H200; not T4).
- Do not use Jev: it is closed and API-only, so it cannot be fine-tuned.

## Datasets (always use official splits)
| Dataset | Loader | Train | Val | Test | Primary metric |
|---|---|---|---|---|---|
| Oxford Flowers-102 | `torchvision.datasets.Flowers102` | 1,020 | 1,020 (official) | 6,149 | mean per-class accuracy |
| Food-101 | `torchvision.datasets.Food101` | 75,750 | carve ~5% from train, fixed seed | 25,250 | top-1 accuracy |

- Never tune anything (prompt, hyperparameters, checkpoint choice) on a test set.
- Flowers-102 has only 10 training images per class: watch for overfitting.
- Food-101 training labels are intentionally noisy. Do not clean them.

## Baselines
1. **CNN baseline (reported, not run):** Tan & Le, EfficientNet, ICML 2019, Table 5.
   Flowers-102: Inception-v4 98.5%, EfficientNet-B7 98.8%. Food-101: Inception-v4 90.8%, EfficientNet-B7 93.0%.
2. **Zero-shot Clef-flash:** same model, same fixed prompt, no weight updates.
3. **Fine-tuned Clef-flash:** LoRA (via `peft`) first; linear probing / LP-FT as comparisons.

## Experimental rules
- One fixed prompt per dataset, stored in `configs/`, shared by the zero-shot and fine-tuned runs.
- Loss: cross-entropy over the model's logits for the candidate classes.
- Save the **full probability vector for every test image** (e.g. `.npy` or parquet) plus true labels,
  so confusion matrices, ECE and error analysis never need a rerun.
- Metrics: dataset primary metric (above), macro-F1, expected calibration error (ECE).
- Fix and log random seeds. Log every run's config, git commit hash, metrics and wall-clock time.
- Before any full run, do a 100-image sanity check and verify option index → class name mapping.

## Compute
- Georgia Tech ICE cluster (SLURM) or Colab Pro with an A100.
- GPU jobs go through `scripts/*.sbatch`. Code must also run on a tiny subset (`--limit N`) for quick checks.

## Repo layout
```
configs/        # per-dataset prompts and run configs (YAML)
src/
  data.py       # dataset loaders, official splits, class-name lists
  model.py      # Clef loading, record building, scoring
  eval.py       # zero-shot / checkpoint evaluation → saved probabilities
  train.py      # LoRA fine-tuning loop
  metrics.py    # per-class acc, top-1, macro-F1, ECE, confusion matrix
scripts/        # sbatch job scripts
results/        # metrics JSON + summary tables (committed); raw predictions (gitignored)
notebooks/      # analysis only
```

## Conventions
- Never commit datasets, model weights, checkpoints or large prediction files (see `.gitignore`).
- Keep the results table in `results/summary.md` up to date after each run.
- Small, reviewable commits

## Current milestone
Zero-shot Clef-flash baseline on Flowers-102 and Food-101 test sets.