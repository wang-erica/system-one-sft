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

**Prompts: list per dataset with stable ids, `p0` is primary.** Each dataset config holds a list of
prompts referenced by stable id (`p0`, `p1`, ...), never by list position. `p0` is frozen before any
results and is used for all headline zero-shot and fine-tuned numbers. Other prompts are a separate
prompt-sensitivity analysis and may not be used to re-pick the primary prompt. Every run logs its
prompt id, and prediction files are named by it.

**Flowers-102 class names: CLIP list, unmodified.** torchvision returns integer labels only, so names
come from the list in OpenAI's CLIP repo (`data/prompts.md` at commit `e184f608`). No hand-editing of
individual names. Stored in `configs/classnames/flowers102.txt`, one name per line in label order.

**Food-101 class names: `meta/classes.txt` with underscores replaced by spaces.** No other changes.
Stored in `configs/classnames/food101.txt`, taken from the same CLIP file, whose list is exactly that
transformation. Food-101 is not downloaded locally, so the loader checks the file against torchvision's
`classes` every time the dataset is loaded and fails on any mismatch.

**Food-101 validation split: stratified 5% of train, fixed seed.** Exactly 5% of each class's 750
training images (37–38 per class, ~3,800 total) is held out, so validation is class-balanced like
the test set. The split indices are committed so every run uses the same split.

**Local datasets: Flowers-102 only.** Flowers-102 (~350 MB) is downloaded locally for testing.
Food-101 (~5 GB) is only downloaded on ICE; locally we test class names and split logic.

**Dependencies.** Only torch 2.11 and transformers 5.10.2 are pinned (the versions the model card was
tested with). A full lock file will be generated from a working ICE install.

## Open questions

- **Option IDs for classes:** class name as option ID (recommended: the model reads the ID text) vs. a
  neutral code with the name in the description. Decide in Phase 3 as part of `p0`.
- **`state` text:** the record requires a text state alongside the image. Decide in Phase 3 as part
  of `p0`.
- **How `p0` is chosen:** hand-written once vs. picked from a few candidates on validation.
- **Scope of the prompt comparison:** zero-shot only, fine-tune with `p0` and evaluate all prompts,
  or fine-tune per prompt. Decide when training starts.
- **CLAUDE.md conventions:** still says "one feature per branch/PR", which conflicts with the git
  workflow above.
