"""
train.py — Main entry point for training all four EEG seizure detection models
===============================================================================
Run from project root:
    python train.py

What happens:
  1. Loads config from configs/config.yaml
  2. Prints GPU info
  3. Builds DataLoaders (patient-level split: train/val/test on different patients)
  4. Computes pos_weight for class imbalance handling
  5. Trains CNNBaseline → CNNLSTM → CNNGRU → TCNModel sequentially
  6. Saves best checkpoint per model to checkpoints/
  7. Saves test_loader.pt for evaluate.py to use the identical test split

Quick test run — edit configs/config.yaml first:
    epochs: 2

Full training:
    epochs: 50   (early stopping will trigger before this if val F1 stalls)
"""

import os
import argparse
import yaml
import torch
import numpy as np

from src.data.dataset import get_dataloaders, load_pos_weight
from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_lstm     import CNNLSTM
from src.models.cnn_gru      import CNNGRU
from src.models.tcn          import TCNModel
from src.training.trainer    import Trainer

def get_device():
    """
    Cross-platform device selection:
    - CUDA for Windows/Linux with NVIDIA
    - MPS for Apple Silicon Mac
    - CPU otherwise
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_pos_weight(train_loader, device):
    """
    Computes pos_weight for BCEWithLogitsLoss from the training set.

    CHB-MIT seizure windows are ~0.5–5% of all windows depending on the patient.
    Without pos_weight, the model learns to always predict "normal" and still
    achieves high accuracy — but completely fails at seizure detection.

    pos_weight[i] = (# normal windows) / (# seizure windows)
    This penalizes missing a seizure proportionally to its rarity.
    """
    print("Computing pos_weight from training set...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    n_pos = max(int(all_labels.sum()), 1)
    n_neg = len(all_labels) - n_pos
    weight = n_neg / n_pos
    print(f"  Seizure windows: {n_pos:,} / {len(all_labels):,} total "
          f"({100*n_pos/len(all_labels):.2f}%)  →  pos_weight = {weight:.1f}")

    return torch.tensor([weight], dtype=torch.float32).to(device)


def main(selected_model: str = "all"):
    # ── Config ────────────────────────────────────────────────────────────────
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("GPU: Apple Silicon (MPS)")
    else:
        print("GPU: None (running on CPU)")

    # ── Output directories ────────────────────────────────────────────────────
    for path in config["paths"].values():
        os.makedirs(path, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    pos_weight = compute_pos_weight(train_loader, device)

    # Save test split so evaluate.py always uses the identical test patients
    torch.save(test_loader, os.path.join(config["paths"]["checkpoints"],
                                          "test_loader.pt"))
    print(f"Test loader saved to {config['paths']['checkpoints']}test_loader.pt")

    num_channels   = config["data"]["num_channels"]
    window_samples = int(config["data"]["window_size_sec"] * config["data"]["sampling_rate"])

    # ══════════════════════════════════════════════════════════════════════════
    # Model 1: CNN Baseline (ablation — no temporal modeling)
    # ══════════════════════════════════════════════════════════════════════════
    if selected_model in ("all", "cnn_baseline"):
        print("\n" + "="*60)
        print("MODEL 1: CNN Baseline (Ablation — No Temporal Modeling)")
        print("="*60)

        cnn_cfg = config["cnn_baseline"]
        cnn = CNNBaseline(
            num_channels=num_channels,
            window_samples=window_samples,
            out_channels=cnn_cfg["out_channels"],
            kernel_size=cnn_cfg["kernel_size"],
            dropout=cnn_cfg["dropout"],
        )
        cnn_trainer = Trainer(cnn, train_loader, val_loader, config, "cnn_baseline", device)
        cnn_trainer.criterion.pos_weight = pos_weight
        cnn_trainer.train()

    # ══════════════════════════════════════════════════════════════════════════
    # Model 2: CNN + LSTM (primary hypothesis model)
    # ══════════════════════════════════════════════════════════════════════════
    if selected_model in ("all", "cnn_lstm"):
        print("\n" + "="*60)
        print("MODEL 2: CNN + LSTM (Primary Model)")
        print("="*60)

        lstm_cfg = config["cnn_lstm"]
        cnn_lstm = CNNLSTM(
            num_channels=num_channels,
            window_samples=window_samples,
            out_channels=lstm_cfg["out_channels"],
            kernel_size=lstm_cfg["kernel_size"],
            lstm_hidden=lstm_cfg["lstm_hidden"],
            lstm_layers=lstm_cfg["lstm_layers"],
            bidirectional=lstm_cfg["bidirectional"],
            dropout=lstm_cfg["dropout"],
        )
        lstm_trainer = Trainer(cnn_lstm, train_loader, val_loader, config, "cnn_lstm", device)
        lstm_trainer.criterion.pos_weight = pos_weight
        lstm_trainer.train()

    # ══════════════════════════════════════════════════════════════════════════
    # Model 3: CNN + GRU (faster LSTM variant)
    # ══════════════════════════════════════════════════════════════════════════
    if selected_model in ("all", "cnn_gru"):
        print("\n" + "="*60)
        print("MODEL 3: CNN + GRU (Faster LSTM Variant)")
        print("="*60)

        gru_cfg = config["cnn_gru"]
        cnn_gru = CNNGRU(
            num_channels=num_channels,
            window_samples=window_samples,
            out_channels=gru_cfg["out_channels"],
            kernel_size=gru_cfg["kernel_size"],
            gru_hidden=gru_cfg["gru_hidden"],
            gru_layers=gru_cfg["gru_layers"],
            bidirectional=gru_cfg["bidirectional"],
            dropout=gru_cfg["dropout"],
        )
        gru_trainer = Trainer(cnn_gru, train_loader, val_loader, config, "cnn_gru", device)
        gru_trainer.criterion.pos_weight = pos_weight
        gru_trainer.train()

    # ══════════════════════════════════════════════════════════════════════════
    # Model 4: TCN (parallel dilated convolutions)
    # ══════════════════════════════════════════════════════════════════════════
    if selected_model in ("all", "tcn"):
        print("\n" + "="*60)
        print("MODEL 4: TCN (Temporal Convolutional Network)")
        print("="*60)

        tcn_cfg = config["tcn"]
        tcn = TCNModel(
            num_channels=num_channels,
            window_samples=window_samples,
            num_block_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
        )
        tcn_trainer = Trainer(tcn, train_loader, val_loader, config, "tcn", device)
        tcn_trainer.criterion.pos_weight = pos_weight
        tcn_trainer.train()

    print("\n" + "="*60)
    if selected_model == "all":
        print("All models trained. Run evaluate.py for full evaluation and charts.")
    else:
        print(f"{selected_model} trained. Run evaluate.py --model {selected_model} to evaluate it.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "cnn_baseline", "cnn_lstm", "cnn_gru", "tcn"],
        help="Train all models or only one specific model.",
    )
    args = parser.parse_args()
    main(args.model)
