# EEG Seizure Detection — CNN · LSTM · GRU · TCN

Group 3 | COSC 4377: Data Science II | University of Houston | Dr. Nouhad Rizk

Khoa Anh Dao · John C Williams · Elias Arellano Campos

Automated binary seizure detection from scalp EEG using the CHB-MIT dataset.
Trains and compares four deep learning architectures to quantify the contribution
of temporal sequential modeling to seizure detection performance.

## Setup

### 1. Create and activate the conda environment

Python 3.12 is required.

```bash
conda create -n eeg python=3.12 -y
conda activate eeg
```

### 2. Install PyTorch with CUDA

For NVIDIA GPU systems:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install pyedflib
```

### 4. Download the dataset

CHB-MIT is Open Access, so no account is needed.
AWS S3 is the fastest method, typically around 30-60 minutes instead of days over HTTP.

Install AWS CLI from:

`https://aws.amazon.com/cli/`

Then run this from the project root:

```powershell
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ data/raw
```

This is safe to interrupt and re-run because `sync` skips files that are already downloaded.

## Project Overview

This repository contains a CHB-MIT EEG seizure detection workflow based on cached window tensors, multi-patient split merging, model training, and evaluation scripts.

## Models

| Model | Role | Key idea |
| --- | --- | --- |
| `cnn` | Ablation baseline | Local feature extraction only | Baseline model |
| `cnn_lstm` | Primary model | CNN features + LSTM sequential modeling |
| `cnn_gru` | Variant | CNN features + GRU (fewer parameters than LSTM) |
| `tcn` | Alternative | Dilated causal convolutions - fully parallel |

Hypothesis: CNN+LSTM and CNN+GRU outperform the CNN baseline in recall and F1, proving temporal sequential modeling adds value for seizure onset detection.

## Quick Start

Recommended end-to-end workflow:

1. Download `data/raw/`
2. Cache per-patient splits with `src/data/cache_patient_splits.py`
3. Merge patients into `data/processed/windowed_splits/multi_patient_4/`
4. Train a model from `src/training/`
5. Run `src/evaluation/confusion_matrix.py` to generate `predictions.pt`
6. Run comparison and analysis scripts on those saved predictions

Project convention:

- The current multi-patient workflow in this repo is centered on `multi_patient_4`
- The main checkpoints in `checkpoints/` are:
  `multi_patient_4_cnn.pt`, `multi_patient_4_cnn_lstm.pt`, `multi_patient_4_cnn_gru.pt`, and `multi_patient_4_tcn.pt`
- Most commands below assume you are running from the repo root with the environment already activated

## Data Layout

Expected raw data layout:

```text
data/raw/
  chb01/
    chb01_01.edf
    ...
    chb01-summary.txt
  chb02/
  ...
```

Processed cached splits are written under:

```text
data/processed/windowed_splits/
  chb01/
    train.pt
    val.pt
    test.pt
  ...
  multi_patient_4/
    train.pt
    val.pt
    test.pt
```

## Project Structure

```text
project_eeg/
  README.md
  requirements.txt
  checkpoints/
    multi_patient_4_cnn.pt
    multi_patient_4_cnn_lstm.pt
    multi_patient_4_cnn_gru.pt
    multi_patient_4_tcn.pt
  data/
    raw/
    processed/
      windowed_splits/
  src/
    data/
      cache_patient_splits.py
      merge_multi_patient_splits.py
      datasets.py
      chbmit_index.py
    models/
      cnn.py
      cnn_lstm.py
      cnn_gru.py
      tcn.py
    training/
      train_multi_patient_cnn.py
      train_multi_patient_cnn_lstm.py
      train_multi_patient_cnn_gru.py
      train_multi_patient_tcn.py
    evaluation/
      confusion_matrix.py
      explainability.py
      roc_pr_curves.py
      per_patient_analysis.py
      inspect_errors.py
```

## Preprocessing

*Patient Split*
- Patients with seizures (positives): 1, 2, 3, 4, 5, 24
- Patients without seizures (negatives): 6–21, 23

### 1. Cache per-patient splits

Use `src/data/cache_patient_splits.py` to build `train.pt`, `val.pt`, and `test.pt` for each patient.

Example:

```bash
python3 -m src.data.cache_patient_splits \
  --patients chb01 chb02 chb03 chb04 chb05 chb06
```

Useful options:

```bash
python3 -m src.data.cache_patient_splits \
  --patients chb14 chb15 \
  --data-root data/raw \
  --out-root data/processed/windowed_splits \
  --window-size-sec 10 \
  --stride-sec 5 \
  --overlap-threshold 0.0
```

Notes:

- The caching pipeline now handles channel mismatches inside a patient by aligning EDFs to shared valid channels.
- Placeholder channels like `--0`, `--1`, etc. are excluded.
- Subjects like `chb04`, `chb14`, and `chb15` needed this handling.

### 2. Merge multiple patients

Use `src/data/merge_multi_patient_splits.py` to combine several patient folders into one multi-patient split set.

Example for 4 patients:

```bash
python3 -m src.data.merge_multi_patient_splits \
  --patients chb01 chb02 chb03 chb04
```

This writes to:

```text
data/processed/windowed_splits/multi_patient_4/
```

You can also override the output folder:

```bash
python3 -m src.data.merge_multi_patient_splits \
  --patients chb01 chb02 chb03 chb04 \
  --output-dir data/processed/windowed_splits/my_custom_split
```

### Cached Split Statistics (Train Split)

The following cached training split statistics summarize dataset size and class imbalance for the current multi-patient workflow.

Our main merged split uses patients `1, 2, 3, 5`.

| Split | X shape | Windows | Positives | Negatives | Positive ratio | Positive weight |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `chb01` | `(16932, 23, 2560)` | 16932 | 44 | 16888 | 0.002599 | 383.8182 |
| `chb02` | `(15099, 23, 2560)` | 15099 | 18 | 15081 | 0.001192 | 837.8333 |
| `chb03` | `(15819, 23, 2560)` | 15819 | 36 | 15783 | 0.002276 | 438.4167 |
| `chb04` | `(58203, 23, 2560)` | 58203 | 12 | 58191 | 0.000206 | 4849.2500 |
| `chb05` | `(16539, 23, 2560)` | 16539 | 51 | 16488 | 0.003084 | 323.2941 |
| `chb24` | `(9347, 23, 2560)` | 9347 | 40 | 9307 | 0.004279 | 232.6750 |
| `multi_patient_4` | `(64389, 23, 2560)` | 64389 | 149 | 64240 | 0.002314 | 431.1409 |

These values show the severe class imbalance in seizure detection, especially for subjects like `chb04`. This is why the training scripts use threshold search and imbalance-aware losses or weighting.

### Patient Dataset Summary Script

Use `src/data/patient_dataset_summary.py` to print a quick summary of every cached training split inside `data/processed/windowed_splits/`.

Run:

```bash
python3 -m src.data.patient_dataset_summary
```

What it does:

- scans each patient or merged split folder inside `data/processed/windowed_splits/`
- reads `train.pt`
- prints:
  `X shape`
  `Windows`
  `Positives`
  `Negatives`
  `Pos ratio`
  `Pos weight`

This is useful for checking class imbalance and confirming that cached split generation succeeded before training.

## Training

Current multi-patient training scripts:

- `src/training/train_multi_patient_cnn.py`
- `src/training/train_multi_patient_cnn_lstm.py`
- `src/training/train_multi_patient_cnn_gru.py`
- `src/training/train_multi_patient_tcn.py`

### CNN baseline

```bash
python3 -m src.training.train_multi_patient_cnn
```

Current script defaults:

- input: `data/processed/windowed_splits/multi_patient_4`
- checkpoint: `checkpoints/multi_patient_4_cnn.pt`

### CNN-LSTM

```bash
python3 -m src.training.train_multi_patient_cnn_lstm
```

Current script defaults:

- input: `data/processed/windowed_splits/multi_patient_4`
- checkpoint: `checkpoints/multi_patient_4_cnn_lstm.pt`

### CNN-GRU

```bash
python3 -m src.training.train_multi_patient_cnn_gru
```

Current script defaults:

- input: `data/processed/windowed_splits/multi_patient_4`
- checkpoint: `checkpoints/multi_patient_4_cnn_gru.pt`

### TCN

```bash
python3 -m src.training.train_multi_patient_tcn
```

Current script defaults:

- input: `data/processed/windowed_splits/multi_patient_4`
- checkpoint: `checkpoints/multi_patient_4_tcn.pt`

## Evaluation

Evaluation scripts are in `src/evaluation/`.

### 1. Confusion matrix + predictions export (use the best threshold per model)

Use `src/evaluation/confusion_matrix.py` to:

- run inference
- print metrics
- save confusion matrix plots
- save `predictions.pt` for downstream analysis

Supported models:

- `cnn`
- `cnn_lstm`
- `cnn_gru`
- `tcn`

Example commands:

```bash
python3 -m src.evaluation.confusion_matrix \
  --model cnn \
  --checkpoint checkpoints/multi_patient_4_cnn.pt \
  --split-path data/processed/windowed_splits/multi_patient_4/test.pt \
  --threshold 0.99 \
  --device cpu
```

```bash
python3 -m src.evaluation.confusion_matrix \
  --model cnn_lstm \
  --checkpoint checkpoints/multi_patient_4_cnn_lstm.pt \
  --split-path data/processed/windowed_splits/multi_patient_4/test.pt \
  --threshold 0.20
```

```bash
python3 -m src.evaluation.confusion_matrix \
  --model cnn_gru \
  --checkpoint checkpoints/multi_patient_4_cnn_gru.pt \
  --split-path data/processed/windowed_splits/multi_patient_4/test.pt \
  --threshold 0.50
```

```bash
python3 -m src.evaluation.confusion_matrix \
  --model tcn \
  --checkpoint checkpoints/multi_patient_4_tcn.pt \
  --split-path data/processed/windowed_splits/multi_patient_4/test.pt \
  --threshold 0.60
```

Outputs go to:

```text
outputs/evaluation/<model>/
  confusion_matrix_counts.png
  confusion_matrix_normalized.png
  predictions.pt
```

Notes:

- `cnn_lstm` and `cnn_gru` are evaluated on reconstructed sequences.
- The CHB01 LSTM checkpoint support includes a legacy architecture path because the original source file is no longer present.
- Use the threshold that matches the saved experiment summary for each model when reproducing final test metrics.


### 2. ROC / PR comparison plots

Use `src/evaluation/roc_pr_curves.py` on saved `predictions.pt` files.

Required:

- `--cnn-preds`
- `--cnn-lstm-preds`

Optional:

- `--cnn-gru-preds`
- `--tcn-preds`

Example:

```bash
python3 -m src.evaluation.roc_pr_curves \
  --cnn-preds outputs/evaluation/cnn/predictions.pt \
  --cnn-lstm-preds outputs/evaluation/cnn_lstm/predictions.pt \
  --cnn-gru-preds outputs/evaluation/cnn_gru/predictions.pt \
  --tcn-preds outputs/evaluation/tcn/predictions.pt
```

Outputs:

```text
outputs/evaluation/comparison/
  roc_comparison.png
  pr_comparison.png
```

### 3. Per-patient analysis

Use `src/evaluation/per_patient_analysis.py` on a `predictions.pt` file.

Generic pattern:

```bash
python3 -m src.evaluation.per_patient_analysis \
  --predictions outputs/evaluation/<model>/predictions.pt
```

Outputs now go into a model-specific folder:

```text
outputs/evaluation/per_patient/<model>/
  per_patient_metrics.csv
  per_patient_f1.png
  per_patient_recall.png
  per_patient_precision.png
  per_patient_support.png
```
### 4. Error inspection

Use `src/evaluation/inspect_errors.py` on a `predictions.pt` file.

Generic pattern:

```bash
python3 -m src.evaluation.inspect_errors \
  --predictions outputs/evaluation/<model>/predictions.pt
```

Outputs now go into a model-specific folder:

```text
outputs/evaluation/error_inspection/<model>/
  all_errors.csv
  top_false_positives.csv
  top_false_negatives.csv
  errors_by_patient.csv
```

### 5. Explainability / saliency maps

Use `src/evaluation/explainability.py`.

Supported models:

- `cnn`
- `cnn_lstm`
- `cnn_gru`
- `tcn`

Example commands:

```bash

# CNN baseline: FP for chb03 (idx 17075), FN for chb02 (idx 7146)
python3 -m src.evaluation.explainability \
  --model cnn \
  --checkpoint checkpoints/multi_patient_4_cnn.pt \
  --split data/processed/windowed_splits/multi_patient_4/test.pt \
  --sample_idx 17075
```

```bash
# CNN+LSTM: FP for chb03 (idx 15749), FN for chb03 (idx 16900)
python3 -m src.evaluation.explainability \
  --model cnn_lstm \
  --checkpoint checkpoints/multi_patient_4_cnn_lstm.pt \
  --split data/processed/windowed_splits/multi_patient_4/test.pt \
  --sample_idx 15749
```

```bash
# CNN+GRU: FP for chb03 (idx 17821), FN for chb02 (idx 7076)
python3 -m src.evaluation.explainability \
  --model cnn_gru \
  --checkpoint checkpoints/multi_patient_4_cnn_gru.pt \
  --split data/processed/windowed_splits/multi_patient_4/test.pt \
  --sample_idx 17821
```

```bash
# TCN: FP for chb03 (idx 17069), FN for chb02 (idx 7146)
python3 -m src.evaluation.explainability \
  --model tcn \
  --checkpoint checkpoints/multi_patient_4_tcn.pt \
  --split data/processed/windowed_splits/multi_patient_4/test.pt \
  --sample_idx 17069
```

Outputs go to:

```text
outputs/evaluation/explainability/
  <model>_sample_<idx>.png
```

`sample_idx` meaning:

- For `cnn` and `tcn`, `sample_idx` is the index of a single cached window in the split tensor.
- For `cnn_lstm` and `cnn_gru`, `sample_idx` is the index of a reconstructed sequence built from consecutive windows within the same recording.

## Current Checkpoints In Repo

Present checkpoint files:

- `checkpoints/multi_patient_4_cnn.pt`
- `checkpoints/multi_patient_4_cnn_lstm.pt`
- `checkpoints/multi_patient_4_cnn_gru.pt`
- `checkpoints/multi_patient_4_tcn.pt`

If you retrain or switch datasets, verify the hardcoded split/checkpoint paths inside each training script before running.

## Known Project-Specific Notes

- Some evaluation scripts rely on saved checkpoint metadata under `ckpt["config"]`.
- The CHB01 LSTM checkpoint uses a legacy architecture path during evaluation because the original source file is not present in `src/models/`.
- Large merged `.pt` files can be memory-heavy, especially for explainability and full inference on multi-patient splits.
- `per_patient_analysis.py` and `inspect_errors.py` are model-agnostic once `predictions.pt` exists.
- `explainability.py` can be especially expensive on large multi-patient splits because it still needs to load from large cached tensors.
- CPU-only inference works, but large evaluations can take a while.
- For sequence models, evaluation is slower than CNN/TCN because sequences must be rebuilt from `recording_id` and `window_idx`.

## Results

### Multi-Patient Test Results (Patients 1, 2, 3, 5)

| Model | Threshold | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN+GRU | 0.50 | 0.8974 | 0.8140 | 0.8537 | 0.9924 |
| CNN+LSTM | 0.20 | 0.7802 | 0.8256 | 0.8023 | 0.9915 |
| TCN | 0.60 | 0.7349 | 0.7093 | 0.7219 | 0.9813 |
| CNN baseline | 0.99 | 0.6118 | 0.6047 | 0.6082 | 0.9695 |

CNN+GRU achieved the best overall F1 (0.8537). CNN+LSTM achieved the highest recall (0.8256) at a lower threshold of 0.20, trading some precision for fewer missed seizures. Both significantly outperformed the CNN baseline, confirming the hypothesis that temporal sequential modeling adds value.

## Output Charts

The evaluation scripts generate charts and analysis files under `outputs/evaluation/`.

### 1. Confusion matrix plots

Files:

- `outputs/evaluation/<model>/confusion_matrix_counts.png`
- `outputs/evaluation/<model>/confusion_matrix_normalized.png`

What they explain:

- The count plot shows the raw number of true negatives, false positives, false negatives, and true positives.
- The normalized plot shows the fraction of each true class that was classified correctly or incorrectly.

What to look for:

- High `TP` and low `FN` if seizure detection recall matters most.
- High `TN` and low `FP` if avoiding false alarms matters most.
- Compare normalized matrices across models, because raw counts can hide class imbalance.

### 2. ROC and Precision-Recall comparison plots

Files:

- `outputs/evaluation/comparison/roc_comparison.png`
- `outputs/evaluation/comparison/pr_comparison.png`

What they explain:

- The ROC curve shows the tradeoff between true positive rate and false positive rate across thresholds.
- The PR curve shows the tradeoff between precision and recall across thresholds.

What to look for:

- Curves closer to the top-left on ROC are better.
- Curves closer to the top-right on PR are better.
- Under strong class imbalance, PR is often more informative than ROC.
- The model with the largest AUC or AP is usually stronger overall, but threshold-specific operating points still matter.

### 3. Per-patient metric plots

Files:

- `outputs/evaluation/per_patient/<model>/per_patient_f1.png`
- `outputs/evaluation/per_patient/<model>/per_patient_recall.png`
- `outputs/evaluation/per_patient/<model>/per_patient_precision.png`
- `outputs/evaluation/per_patient/<model>/per_patient_support.png`

What they explain:

- These plots break performance down by patient instead of only reporting one global score.
- The support plot shows how many positive and negative samples each patient contributes.

What to look for:

- Large variation across patients can indicate poor generalization.
- Patients with very low recall may be clinically important failure cases.
- High performance on patients with very small support should be interpreted carefully.
- Compare metric plots with the support plot to avoid over-interpreting unstable small-sample results.

### 4. Explainability / saliency plots

Files:

- `outputs/evaluation/explainability/<model>_sample_<idx>.png`

What they explain:

- These highlight which input channels, timesteps, or sequence steps most influenced the model's prediction for one sample.
- For sequence models, the plots also summarize which windows in the sequence mattered most.

What to look for:

- Whether the model focuses on plausible EEG regions rather than diffuse noise.
- Whether seizure-positive samples show concentrated attention around meaningful temporal segments.
- Whether false positives and false negatives reveal unstable or misleading saliency patterns.

### 5. Error inspection CSV outputs

Files:

- `outputs/evaluation/error_inspection/<model>/all_errors.csv`
- `outputs/evaluation/error_inspection/<model>/top_false_positives.csv`
- `outputs/evaluation/error_inspection/<model>/top_false_negatives.csv`
- `outputs/evaluation/error_inspection/<model>/errors_by_patient.csv`

What they explain:

- These files list the specific samples the model got wrong, ranked by confidence or severity.
- They also summarize which patients contribute the most false positives or false negatives.

What to look for:

- Repeated failures from the same patient or recording.
- High-confidence false positives, which suggest over-triggering.
- High-confidence false negatives, which are usually the most serious missed detections.
- Whether certain patients dominate the error distribution.

## Dataset

CHB-MIT Scalp EEG Database — PhysioNet (Open Access)

- 23 pediatric epilepsy patients
- ~1,000 hours
- 198 labeled seizure events
- 23 channels at 256 Hz
- EDF format

https://physionet.org/content/chbmit/1.0.0/

## Project Artifacts

Large project artifacts are not tracked in GitHub due to size constraints.
These are shared separately in the team folder:

- `data/raw/`
- `data/processed/`
- `checkpoints/`
- `outputs/`

After cloning the repo, place those folders in the project root before running training or evaluation scripts.

## References

- Goldberger et al. (2000) — CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
- Bai et al. (2018) — TCN: https://arxiv.org/abs/1803.01271
- Hochreiter & Schmidhuber (1997) — LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf
- Cho et al. (2014) — GRU: https://arxiv.org/abs/1406.1078
