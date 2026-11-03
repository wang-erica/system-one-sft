# Decision log

Project decisions and the reasons for them, newest last. Open questions are listed at the bottom
until they are decided.

## 2026-11-02

**Compute: ICE first.** GPU jobs target the Georgia Tech ICE cluster (SLURM). Colab is secondary.
Data code is developed and tested locally on CPU in a project `.venv`; nothing that needs the model
runs locally.

**Git workflow: small local commits on `main`.** One commit per feature or change so the log is easy
to audit. No feature branches or PRs for now.

**Model revision pinned.** `Cloudflare/clef-flash` at `fb4417d31c1d0f9e7e77f1141b40aaede7491e00`, so
every run uses identical weights and release code. See `docs/clef_interface.md`.

**Prompts: candidates per dataset with stable ids.** Each dataset config (`configs/<dataset>.yaml`)
holds prompt candidates keyed by stable id (`p0`, `p1`, ...). An id is never renumbered, reused or
edited once results exist for it. Every run logs its prompt id, and prediction files are named by it.
*(Revised: originally `p0` was to be primary by default; superseded by the selection rule below.)*

**Primary prompt chosen on validation.** Four candidates per dataset (`p0`–`p3`) are scored zero-shot
on validation: the full official Flowers-102 val set, and a fixed-seed 1,000-image subset of the
Food-101 val split. The best by the dataset's primary metric becomes `primary_prompt` in the config,
frozen before any test-set run, and is used for all headline zero-shot and fine-tuned numbers. The
other candidates feed the later prompt-sensitivity analysis and may not be used to re-pick the primary.

**Option IDs are class names.** Each class's name is its option ID in the `choice` question, because
the model reads the ID text. Whether options also get a description is a per-prompt setting
(`option_description`).

**Flowers-102 class names: CLIP list, unmodified.** torchvision returns integer labels only, so names
come from the list in OpenAI's CLIP repo (`data/prompts.md` at commit `e184f608`). No hand-editing of
individual names. Stored in `configs/classnames/flowers102.txt`, one name per line in label order.

**Food-101 class names: `meta/classes.txt` with underscores replaced by spaces.** No other changes.
Stored in `configs/classnames/food101.txt`, taken from the same CLIP file, whose list is exactly that
transformation. Food-101 is not downloaded locally, so the loader checks the file against torchvision's
`classes` every time the dataset is loaded and fails on any mismatch.

**Food-101 validation split: stratified 5% of train, fixed seed.** 38 of each class's 750 training
images (5% rounded up; 3,838 total) are held out with seed 0, so validation is exactly class-balanced
like the test set. The held-out image IDs are committed in `configs/splits/food101_val.json` so every
run uses the same split. That file is generated once on ICE (`python -m src.data make-food101-split`),
since Food-101 is not downloaded locally.

**Subsets for quick checks.** `--limit N` takes a fixed-seed random subset, not the first N images,
because Food-101 is stored grouped by class and the first N would all be one class.

**Local datasets: Flowers-102 only.** Flowers-102 (~350 MB) is downloaded locally for testing.
Food-101 (~5 GB) is only downloaded on ICE; locally we test class names and split logic.

**Metrics.** Computed in `src/metrics.py` from saved logits, so any metric can be recomputed later:
- Mean per-class accuracy: average recall over classes present in the labels (same as scikit-learn's
  `balanced_accuracy_score`). Primary metric for Flowers-102.
- Top-1 accuracy: primary metric for Food-101. Top-5 accuracy is also reported.
- Macro-F1: unweighted mean over classes present in labels or predictions (scikit-learn's default).
- ECE: top-label, 15 equal-width bins, bin m covers ((m-1)/15, m/15] (Guo et al., 2017). Other
  binning schemes can be computed later from the saved logits.
- Negative log-likelihood, for later calibration work.

**Save logits, not just probabilities.** Every evaluation saves full float32 logits per image (plus
labels and image indices). Probabilities are recomputed exactly with a softmax, tiny probabilities
are not lost to rounding, and post-hoc temperature scaling (fit on validation) stays possible without
rerunning the model.

**Dependencies.** Only torch 2.11 and transformers 5.10.2 are pinned (the versions the model card was
tested with). A full lock file will be generated from a working ICE install.

## Open questions

- **Prompt candidate wording:** `p0`–`p3` in the configs are drafts pending team review. They must be
  final before any validation run, since ids cannot be edited after results exist.
- **Scope of the prompt comparison:** zero-shot only, fine-tune with the primary prompt and evaluate
  all prompts, or fine-tune per prompt. Decide when training starts.
