# EEG Seizure Detection — CNN · LSTM · GRU · TCN

Group 3 | COSC 4377: Data Science II | University of Houston | Dr. Nouhad Rizk
Khoa Anh Dao · John C Williams · Elias Arellano Campos

Automated binary seizure detection from scalp EEG using the CHB-MIT dataset.
Trains and compares four deep learning architectures to quantify the contribution
of temporal sequential modeling to seizure detection performance.

---

## Setup

**1. Create and activate the conda environment (Python 3.12 required)**
```bash
conda create -n egg python=3.12 -y
conda activate# EEG Seizure Detection — CNN · LSTM · GRU · TCN

Group 3 | COSC 4377: Data Science II | University of Houston | Dr. Nouhad Rizk
Khoa Anh Dao · John C Williams · Elias Arellano Campos

Automated binary seizure detection from scalp EEG using the CHB-MIT dataset.
Trains and compares four deep learning architectures to quantify the contribution
of temporal sequential modeling to seizure detection performance.

---

## Setup

**1. Create and activate the conda environment (Python 3.12 required)**
```bash
conda create -n egg python=3.12 -y
conda activate egg
```

**2. Install PyTorch with CUDA (NVIDIA GPU)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install remaining dependencies**
```bash
pip install -r requirements.txt
pip install pyedflib
```

**4. Download the dataset**

CHB-MIT is Open Access — no account needed. AWS S3 is the fastest method (~30-60 min vs days via HTTP).

Install AWS CLI from https://aws.amazon.com/cli/ then run from the project root:
```powershell
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ data/raw
```

Safe to interrupt and re-run — sync skips already-downloaded files.

**5. Run preprocessing**
```bash
python src/data/preprocess.py --config configs/config.yaml
```

> Every time you open VSCode, activate the environment first:
> `conda activate egg`

---

## Running

**Preprocess** (run once, or again when changing patient count or stride):
```bash
python src/data/preprocess.py --config configs/config.yaml
```

**Train:**
```bash
python train.py                   # all 4 models
python train.py --model cnn_lstm  # one model
```

**Evaluate:**
```bash
python evaluate.py                   # all 4 models
python evaluate.py --model cnn_lstm  # one model
```

All outputs go to `checkpoints/`, `logs/`, `results/`.

---

## Scaling up

Edit `configs/config.yaml` — these are the only values you need to change:

| Phase | n_train | n_val | n_test | stride_sec | Notes |
|---|---|---|---|---|---|
| Pipeline test | 2 | 1 | 1 | 5.0 | Verify everything runs |
| Small training | 8 | 2 | 2 | 5.0 | Real learning begins |
| Full training | 16 | 3 | 5 | 5.0 | Full experiment |
| Full + overlap | 16 | 3 | 5 | 2.5 | Best model, 2× disk space |

After changing patients or stride, delete processed data and reprocess:
```powershell
Remove-Item -Recurse -Force data/processed
python src/data/preprocess.py --config configs/config.yaml
```

---

## Project Structure

```
eeg-seizure-detection/
├── configs/
│   └── config.yaml              ← all hyperparameters in one place
├── data/
│   ├── raw/                     ← CHB-MIT .edf files (gitignored, ~42GB)
│   └── processed/               ← windowed .npy arrays (gitignored)
├── scripts/
│   ├── setup_data.py            ← downloads dataset and runs preprocessing
│   └── learning_curve.py        ← plots F1/Recall vs patient count
├── src/
│   ├── data/
│   │   ├── preprocess.py        ← EDF loading, windowing, patient-level split
│   │   └── dataset.py           ← EEGDataset class and DataLoader factory
│   ├── models/
│   │   ├── cnn_baseline.py      ← CNN-only ablation baseline
│   │   ├── cnn_lstm.py          ← CNN + LSTM (primary model)
│   │   ├── cnn_gru.py           ← CNN + GRU (faster LSTM variant)
│   │   └── tcn.py               ← Temporal Convolutional Network
│   ├── training/
│   │   └── trainer.py           ← Trainer class: loop, early stopping, checkpointing
│   └── evaluation/
│       ├── metrics.py           ← recall, F1, ROC-AUC, PR-AUC, threshold sweep
│       └── visualize.py         ← all diagnostic and evaluation plots
├── checkpoints/                 ← saved model weights (gitignored)
├── logs/                        ← per-epoch CSV logs (gitignored)
├── results/                     ← all generated charts and metrics (gitignored)
├── train.py                     ← training entry point
├── evaluate.py                  ← evaluation entry point
└── requirements.txt
```

---

## Models

| Model | Role | Key idea |
|---|---|---|
| `cnn_baseline` | Ablation baseline | Local feature extraction only |
| `cnn_lstm` | Primary model | CNN features + LSTM sequential modeling |
| `cnn_gru` | Variant | CNN features + GRU (fewer parameters than LSTM) |
| `tcn` | Alternative | Dilated causal convolutions — fully parallel |

**Hypothesis:** CNN+LSTM and CNN+GRU outperform the CNN baseline in recall and F1,
proving temporal sequential modeling adds value for seizure onset detection.

**Patient-level split:** train on patients 1–16, val on 17–19, test on 20–24.
The model is tested on patients it has **never seen** during training.

---

## Output Charts

| File | What to look for |
|---|---|
| `*_training_curves.png` | Val diverging from train → overfitting |
| `*_grad_norms.png` | Spikes → exploding gradients (LSTM/GRU debug) |
| `*_roc_curve.png` | AUC > 0.85 = good discrimination |
| `*_pr_curve.png` | More reliable than ROC under class imbalance |
| `*_confusion_matrix.png` | FN = missed seizures → lower threshold |
| `*_threshold_sweep.png` | Pick threshold for your recall target |
| `ablation_comparison.png` | All 4 models side by side |
| `metrics_summary.csv` | Full metrics for all models |
| `learning_curve.png` | F1/Recall vs patients (run scripts/learning_curve.py) |

---

## Dataset

**CHB-MIT Scalp EEG Database** — PhysioNet (Open Access)
23 pediatric epilepsy patients · ~1,000 hours · 198 labeled seizure events
23 channels at 256 Hz · EDF format
https://physionet.org/content/chbmit/1.0.0/

---



## References

- Goldberger et al. (2000) — CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
- Bai et al. (2018) — TCN: https://arxiv.org/abs/1803.01271
- Hochreiter & Schmidhuber (1997) — LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf
- Cho et al. (2014) — GRU: https://arxiv.org/abs/1406.1078 eeg
```

**2. Install PyTorch with CUDA (NVIDIA GPU)**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install remaining dependencies**
```bash
pip install -r requirements.txt
pip install pyedflib
```

**4. Download and preprocess the dataset**
```bash
python scripts/setup_data.py
```
CHB-MIT is Open Access — no account needed. Uses AWS S3 if AWS CLI is installed
(fast), otherwise falls back to HTTP. ~42GB download.

> Every time you open VSCode, activate the environment first:
> `conda activate eeg`

---

## Running

**Preprocess** (run once, or again when changing patient count or stride):
```bash
python src/data/preprocess.py --config configs/config.yaml
```

**Train:**
```bash
python train.py                   # all 4 models
python train.py --model cnn_lstm  # one model
```

**Evaluate:**
```bash
python evaluate.py                   # all 4 models
python evaluate.py --model cnn_lstm  # one model
```

All outputs go to `checkpoints/`, `logs/`, `results/`.

---

## Scaling up

Edit `configs/config.yaml` — these are the only values you need to change:

| Phase | n_train | n_val | n_test | stride_sec | Notes |
|---|---|---|---|---|---|
| Pipeline test | 2 | 1 | 1 | 5.0 | Verify everything runs |
| Small training | 8 | 2 | 2 | 5.0 | Real learning begins |
| Full training | 16 | 3 | 5 | 5.0 | Full experiment |
| Full + overlap | 16 | 3 | 5 | 2.5 | Best model, 2× disk space |

After changing patients or stride, delete processed data and reprocess:
```powershell
Remove-Item -Recurse -Force data/processed
python src/data/preprocess.py --config configs/config.yaml
```

---

## Project Structure

```
eeg-seizure-detection/
├── configs/
│   └── config.yaml              ← all hyperparameters in one place
├── data/
│   ├── raw/                     ← CHB-MIT .edf files (gitignored, ~42GB)
│   └── processed/               ← windowed .npy arrays (gitignored)
├── scripts/
│   ├── setup_data.py            ← downloads dataset and runs preprocessing
│   └── learning_curve.py        ← plots F1/Recall vs patient count
├── src/
│   ├── data/
│   │   ├── preprocess.py        ← EDF loading, windowing, patient-level split
│   │   └── dataset.py           ← EEGDataset class and DataLoader factory
│   ├── models/
│   │   ├── cnn_baseline.py      ← CNN-only ablation baseline
│   │   ├── cnn_lstm.py          ← CNN + LSTM (primary model)
│   │   ├── cnn_gru.py           ← CNN + GRU (faster LSTM variant)
│   │   └── tcn.py               ← Temporal Convolutional Network
│   ├── training/
│   │   └── trainer.py           ← Trainer class: loop, early stopping, checkpointing
│   └── evaluation/
│       ├── metrics.py           ← recall, F1, ROC-AUC, PR-AUC, threshold sweep
│       └── visualize.py         ← all diagnostic and evaluation plots
├── checkpoints/                 ← saved model weights (gitignored)
├── logs/                        ← per-epoch CSV logs (gitignored)
├── results/                     ← all generated charts and metrics (gitignored)
├── train.py                     ← training entry point
├── evaluate.py                  ← evaluation entry point
└── requirements.txt
```

---

## Models

| Model | Role | Key idea |
|---|---|---|
| `cnn_baseline` | Ablation baseline | Local feature extraction only |
| `cnn_lstm` | Primary model | CNN features + LSTM sequential modeling |
| `cnn_gru` | Variant | CNN features + GRU (fewer parameters than LSTM) |
| `tcn` | Alternative | Dilated causal convolutions — fully parallel |

**Hypothesis:** CNN+LSTM and CNN+GRU outperform the CNN baseline in recall and F1,
proving temporal sequential modeling adds value for seizure onset detection.

**Patient-level split:** train on patients 1–16, val on 17–19, test on 20–24.
The model is tested on patients it has **never seen** during training.

---

## Output Charts

| File | What to look for |
|---|---|
| `*_training_curves.png` | Val diverging from train → overfitting |
| `*_grad_norms.png` | Spikes → exploding gradients (LSTM/GRU debug) |
| `*_roc_curve.png` | AUC > 0.85 = good discrimination |
| `*_pr_curve.png` | More reliable than ROC under class imbalance |
| `*_confusion_matrix.png` | FN = missed seizures → lower threshold |
| `*_threshold_sweep.png` | Pick threshold for your recall target |
| `ablation_comparison.png` | All 4 models side by side |
| `metrics_summary.csv` | Full metrics for all models |
| `learning_curve.png` | F1/Recall vs patients (run scripts/learning_curve.py) |

---

## Dataset

**CHB-MIT Scalp EEG Database** — PhysioNet (Open Access)
23 pediatric epilepsy patients · ~1,000 hours · 198 labeled seizure events
23 channels at 256 Hz · EDF format
https://physionet.org/content/chbmit/1.0.0/


## References

- Goldberger et al. (2000) — CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
- Bai et al. (2018) — TCN: https://arxiv.org/abs/1803.01271
- Hochreiter & Schmidhuber (1997) — LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf
- Cho et al. (2014) — GRU: https://arxiv.org/abs/1406.1078
