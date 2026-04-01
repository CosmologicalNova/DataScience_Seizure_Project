# EEG Seizure Detection — CNN · LSTM · GRU · TCN

Group 3 | COSC 4377: Data Science II | University of Houston | Dr. Nouhad Rizk
Khoa Anh Dao · John C Williams · Elias Arellano Campos

Automated binary seizure detection from scalp EEG using the CHB-MIT dataset.
Trains and compares four deep learning architectures to quantify the contribution
of temporal sequential modeling to seizure detection performance.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
For PyTorch with NVIDIA GPU (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**2. Get your PhysioNet credentials**
- Create a free account at https://physionet.org/register/
- Accept the CHB-MIT data use agreement at https://physionet.org/content/chbmit/1.0.0/

**3. Create a `.env` file in the project root**
```
PHYSIONET_USERNAME=your_username
PHYSIONET_PASSWORD=your_password
```

**4. Download and preprocess the dataset**
```bash
python scripts/setup_data.py
```
Downloads the full ~45GB CHB-MIT dataset and runs preprocessing automatically.
Returns instantly if already downloaded.

---

## Running

**Quick test run — edit `configs/config.yaml` first:**
```yaml
training:
  epochs: 2
data:
  n_train_patients: 4
```

**Train all models:**
```bash
python train.py
```

**Evaluate and generate all charts:**
```bash
python evaluate.py
```

**Learning curve across multiple patient counts:**

After each training run at a different `n_train_patients`, rename the logs:
```bash
# After 4-patient run:
mv logs/cnn_lstm_log.csv logs/cnn_lstm_4pat.csv
mv logs/cnn_gru_log.csv  logs/cnn_gru_4pat.csv
# ... repeat for other patient counts ...
python scripts/learning_curve.py
```

---

## Project Structure

```
eeg-seizure-detection/
├── configs/
│   └── config.yaml              — all hyperparameters in one place
├── data/
│   ├── raw/                     — CHB-MIT .edf files (gitignored, ~45GB)
│   └── processed/               — windowed .npy arrays (gitignored)
├── scripts/
│   ├── setup_data.py            — downloads dataset and runs preprocessing
│   └── learning_curve.py        — plots F1/Recall vs patient count
├── src/
│   ├── data/
│   │   ├── preprocess.py        — EDF loading, windowing, patient-level split
│   │   └── dataset.py           — EEGDataset class and DataLoader factory
│   ├── models/
│   │   ├── cnn_baseline.py      — CNN-only ablation baseline
│   │   ├── cnn_lstm.py          — CNN + LSTM (primary model)
│   │   ├── cnn_gru.py           — CNN + GRU (faster LSTM variant)
│   │   └── tcn.py               — Temporal Convolutional Network
│   ├── training/
│   │   └── trainer.py           — Trainer class: loop, early stopping, checkpointing
│   └── evaluation/
│       ├── metrics.py           — recall, F1, ROC-AUC, PR-AUC, threshold sweep
│       └── visualize.py         — all diagnostic and evaluation plots
├── checkpoints/                 — saved model weights (gitignored)
├── logs/                        — per-epoch CSV logs (gitignored)
├── results/                     — all generated charts and metrics (gitignored)
├── notebooks/
│   └── eda.ipynb                — interactive data exploration
├── .env                         — PhysioNet credentials (never pushed to GitHub)
├── train.py                     — main training entry point
├── evaluate.py                  — main evaluation entry point
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
proving that temporal sequential modeling adds value for seizure onset detection.

**Patient-level split:** Train on patients 1–16, validate on 17–19, test on 20–24.
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

## Config Reference

| Parameter | What it controls |
|---|---|
| `data.n_train_patients` | Use first N of 16 training patients (for learning curve) |
| `data.stride_sec` | `2.5` = 50% overlap · `5.0` = non-overlapping |
| `training.monitor_metric` | `"f1"` or `"recall"` for early stopping |
| `training.grad_clip` | Keep enabled for LSTM/GRU — prevents exploding gradients |
| `evaluation.threshold` | Lower (e.g. `0.3`) = higher recall, more false alarms |
| `cnn_lstm.bidirectional` | `true` for offline analysis (better accuracy, not real-time) |

---

## Team Contributions

| Member | Responsibility |
|---|---|
| Khoa Dao | CNN feature extraction · CNN+GRU · training infrastructure |
| John Williams | Data preprocessing · patient-level split · TCN |
| Elias Arellano | LSTM temporal modeling · evaluation · visualization |

All members: proposal writing, experiment discussion, final report.

---

## Dataset

**CHB-MIT Scalp EEG Database** — PhysioNet
23 pediatric epilepsy patients · ~1,000 hours · 198 labeled seizure events
23 channels at 256 Hz · EDF format
https://physionet.org/content/chbmit/1.0.0/

---

## References

- Goldberger et al. (2000) — CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
- Bai et al. (2018) — TCN: https://arxiv.org/abs/1803.01271
- Hochreiter & Schmidhuber (1997) — LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf
- Cho et al. (2014) — GRU: https://arxiv.org/abs/1406.1078
