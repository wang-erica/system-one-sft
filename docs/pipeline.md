# Kaggle comparison pipeline

This is a configurable image-classification pipeline. It automates leaderboard lookup,
matched evaluation, supervised Clef LoRA fine-tuning, validation checkpoint selection,
and reporting. Competition-specific data preparation and a runnable published solution
are still required. It does not infer an architecture or download a winner's weights
from a leaderboard entry. Dataset URLs have no winning model. Detection, segmentation,
multilabel tasks and hidden Kaggle test labels are outside this first implementation.

## Plan from a URL

Install the project requirements and the optional Kaggle CLI (`pip install kaggle`).
Configure Kaggle credentials and accept competition rules on Kaggle when needed.

```bash
python -m src.pipeline plan https://www.kaggle.com/competitions/COMPETITION --output outputs/competition-plan
```

This saves `discovery.json` with the server's first leaderboard row and raw CSV, and
`experiment.yaml` with explicit fields to complete. `--offline` writes a template without
network access. The CLI's `--show` uses final standings for concluded competitions;
a current leader in an active competition is provisional. Inspect the published winning
write-up and record its URL in `baseline.solution_url`; verify eligibility and rank.
See [Kaggle's competition CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md).
A leaderboard score is contextual evidence, never a locally measured baseline score.

Download with `kaggle competitions download COMPETITION -p data/COMPETITION` (or
`kaggle datasets download OWNER/DATASET -p data/DATASET`). Prepare labeled images as:

```text
data/prepared/
  train/class_a/*.jpg
  train/class_b/*.jpg
  val/class_a/*.jpg
  val/class_b/*.jpg
  test/class_a/*.jpg
  test/class_b/*.jpg
```

Preserve official splits where available. For datasets without labeled test images,
create a fixed held-out split from labeled training data before experimenting. Group
related subjects together where appropriate. The pipeline rejects identical image bytes
across splits, but cannot detect subject leakage or re-encoded duplicates.

Set `data.root` relative to the YAML file and declare `data.class_names` in label order.
Folders must match these names exactly in all splits. Fix the prompt before testing.
Use `mean_per_class_accuracy` for Flowers-102 and `top1_accuracy` for Food-101.

## Winning solution adapter

Set `baseline.factory` to a local trusted Python callable, e.g.
`examples.torchscript_baseline:create_scorer`. Its interface is:

```python
def create_scorer(baseline_config, class_names, device):
    # Load the published weights and exact inference preprocessing.
    def score(images):
        # Return finite numpy logits [batch, classes], in class_names order.
        ...
    return score
```

Record `baseline.checkpoint` and `baseline.solution_url`. Checkpoint paths are interpreted
by the adapter; the example uses paths relative to the working directory. The supplied
TorchScript example additionally requires `output_classes`, `resize`, `crop_size`, `mean`
and `std`, taken from the published solution. It is an integration example, not a claim
that any particular winning solution is TorchScript or uses standard preprocessing.
Use raw logits, not softmax probabilities. Ensembles require their own adapter.

## Execute on a BF16 CUDA GPU

```bash
python -m src.pipeline run outputs/competition-plan/experiment.yaml --output outputs/competition-run
```

Use a fresh output directory each time. `--limit 100` restricts training and validation
for a smoke check; test evaluation still uses every held-out image. Smoke runs must not
be treated as final research results. Start with a small separately prepared dataset for
an inexpensive full integration check.

The runner evaluates the published solution and zero-shot Clef, trains LoRA on the
language backbone with classification cross-entropy, restores the adapter with the best
validation primary metric, then evaluates SFT on the identical test set. The joint head
and vision encoder remain frozen. LoRA target modules are configurable and must exist
in the pinned model. It saves full logits, labels, indices, and class names for each
method; a hashed data manifest; copied config; validation history and adapter weights;
metrics; environment/git metadata; a Markdown table and primary-metric deltas. A failed
run keeps completed artifacts and a failure status; restarting requires a fresh directory.
Copy verified final results into `results/summary.md` after reviewing the run report.

GPU training against the actual Clef release must be validated on ICE. CPU tests use
synthetic models and images; they do not establish training quality, GPU memory needs,
or reproducibility of any published Kaggle solution.
