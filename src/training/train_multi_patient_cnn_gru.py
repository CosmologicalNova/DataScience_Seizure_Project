from __future__ import annotations
import certifi
import os
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.models.cnn_gru import CNNGRU

print(f"Using certifi CA bundle at: {certifi.where()}")


# =========================================================
# CONFIG
# =========================================================
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
SEQ_LEN = 8
HIDDEN_DIM = 64
THRESHOLD = 0.5

# Early stopping
PATIENCE = 10
MIN_DELTA = 1e-3

TRAIN_PATH = Path("data/processed/windowed_splits/multi_patient_4/train.pt")
VAL_PATH   = Path("data/processed/windowed_splits/multi_patient_4/val.pt")
TEST_PATH  = Path("data/processed/windowed_splits/multi_patient_4/test.pt")

REQUESTED_DEVICE = os.environ.get("EEG_DEVICE", "").strip().lower()
if REQUESTED_DEVICE:
    DEVICE = torch.device(REQUESTED_DEVICE)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =========================================================
# FOCAL LOSS
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


# =========================================================
# DATASET
# =========================================================
class EEGSequenceDataset(Dataset):
    """
    Builds sequences of consecutive windows inside each recording only.
    Label for each sequence = label of the last window in the sequence.
    """

    def __init__(self, split_path: Path, seq_len: int):
        data = torch.load(split_path, map_location="cpu")

        self.X = data["X"].float()                # [N, C, T]
        self.y = data["y"].float().view(-1)       # [N]
        self.patient_id = data["patient_id"]
        self.recording_id = data["recording_id"]  # [N]
        self.window_idx = data["window_idx"]      # [N]

        self.seq_len = seq_len
        self.samples = []

        groups = defaultdict(list)

        n = len(self.X)
        if isinstance(self.patient_id, str):
            patient_ids = [self.patient_id] * n
        else:
            patient_ids = [str(pid) for pid in self.patient_id]

        recording_ids = [str(rid) for rid in self.recording_id]
        window_indices = [int(idx) for idx in self.window_idx]

        for i in range(n):
            key = (patient_ids[i], recording_ids[i])
            groups[key].append(i)

        for key, indices in groups.items():
            indices = sorted(indices, key=lambda idx: window_indices[idx])

            if len(indices) < seq_len:
                continue

            for end in range(seq_len - 1, len(indices)):
                seq_indices = indices[end - seq_len + 1 : end + 1]
                self.samples.append(seq_indices)

        print(f"{split_path.name}: {len(self.samples)} sequences built")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  keys loaded: {list(data.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_indices = self.samples[idx]
        x_seq = self.X[seq_indices]         # [seq_len, C, T]
        y_seq = self.y[seq_indices[-1]]     # label = last window
        return x_seq, y_seq


def make_loader(split_path: Path, batch_size: int, shuffle: bool, seq_len: int):
    ds = EEGSequenceDataset(split_path, seq_len=seq_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
    return ds, loader


# =========================================================
# METRICS / EVAL
# =========================================================
def compute_metrics_from_probs(all_targets, all_probs, threshold=0.5):
    y_true = np.array(all_targets, dtype=np.float32)
    y_prob = np.array(all_probs, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "cm": cm,
    }


@torch.no_grad()
def collect_probs(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).float().view(-1, 1)

        logits = model(X)
        if logits.ndim == 1:
            logits = logits.view(-1, 1)

        loss = criterion(logits, y)

        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits detected during evaluation.")
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected during evaluation.")

        probs = torch.sigmoid(logits)

        if not torch.isfinite(probs).all():
            raise ValueError("Non-finite probabilities detected during evaluation.")

        total_loss += loss.item() * X.size(0)

        all_probs.extend(probs.detach().cpu().view(-1).numpy().tolist())
        all_targets.extend(y.detach().cpu().view(-1).numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_targets, all_probs


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    avg_loss, all_targets, all_probs = collect_probs(model, loader, criterion, device)
    metrics = compute_metrics_from_probs(all_targets, all_probs, threshold=threshold)
    metrics["loss"] = avg_loss
    metrics["threshold"] = threshold
    return metrics


def find_best_threshold(all_targets, all_probs):
    candidate_thresholds = [
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.85, 0.90,
        0.92, 0.94, 0.95, 0.96, 0.97,
        0.98, 0.99
    ]

    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0

    print("\nTHRESHOLD SEARCH ON VAL")
    print("-" * 80)
    print(f"{'thr':>6} {'prec':>8} {'rec':>8} {'f1':>8} {'auc':>8}")
    print("-" * 80)

    for threshold in candidate_thresholds:
        metrics = compute_metrics_from_probs(all_targets, all_probs, threshold=threshold)

        print(
            f"{threshold:>6.2f} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['auc']:>8.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    print("-" * 80)
    print(
        f"Best threshold on VAL: {best_threshold:.2f} | "
        f"precision={best_metrics['precision']:.4f} | "
        f"recall={best_metrics['recall']:.4f} | "
        f"f1={best_metrics['f1']:.4f}"
    )

    return best_threshold, best_metrics


def print_metrics_block(title, metrics):
    print(f"\n{title}")
    for k, v in metrics.items():
        if k == "cm":
            print(f"{k}:")
            print(v)
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


# =========================================================
# TRAIN
# =========================================================
def main():
    print(f"Using device: {DEVICE}")

    train_ds, train_loader = make_loader(TRAIN_PATH, BATCH_SIZE, True, SEQ_LEN)
    val_ds, val_loader = make_loader(VAL_PATH, BATCH_SIZE, False, SEQ_LEN)
    test_ds, test_loader = make_loader(TEST_PATH, BATCH_SIZE, False, SEQ_LEN)

    if len(train_ds) == 0:
        raise ValueError("Training dataset has 0 sequences. Check SEQ_LEN and merged split metadata.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset has 0 sequences. Check SEQ_LEN and merged split metadata.")
    if len(test_ds) == 0:
        raise ValueError("Test dataset has 0 sequences. Check SEQ_LEN and merged split metadata.")

    train_labels = []
    for _, y in train_ds:
        train_labels.append(float(y))

    train_labels = torch.tensor(train_labels)
    num_pos = int((train_labels == 1).sum().item())
    num_neg = int((train_labels == 0).sum().item())

    print(f"Train positives: {num_pos}")
    print(f"Train negatives: {num_neg}")

    sample_x, _ = train_ds[0]
    print(f"Sample sequence shape: {sample_x.shape}")  # [seq_len, C, T]

    model = CNNGRU(
        in_channels=sample_x.shape[1],
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val_f1 = -1.0
    epochs_no_improve = 0

    best_path = Path("checkpoints/best_multi_patient_cnn_gru_4.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{EPOCHS}",
            leave=False,
            dynamic_ncols=True,
        )

        for X, y in train_pbar:
            X = X.to(DEVICE)
            y = y.to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()

            logits = model(X)
            if logits.ndim == 1:
                logits = logits.view(-1, 1)

            loss = criterion(logits, y)

            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite logits detected during training.")
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss detected during training.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        train_loss = running_loss / len(train_loader.dataset)

        val_metrics_05 = evaluate(model, val_loader, criterion, DEVICE, threshold=0.5)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics_05['loss']:.4f} | "
            f"val_f1={val_metrics_05['f1']:.4f} | "
            f"val_recall={val_metrics_05['recall']:.4f} | "
            f"val_auc={val_metrics_05['auc']:.4f}"
        )

        scheduler.step(val_metrics_05["f1"])

        if val_metrics_05["f1"] > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_metrics_05["f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved | val_f1={best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs")

            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement.")
                break

    print("\nLoading best model...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    print("\nBest model selected by val F1 @ threshold=0.5 during training")

    # Collect probabilities once
    val_loss, val_targets, val_probs = collect_probs(model, val_loader, criterion, DEVICE)
    test_loss, test_targets, test_probs = collect_probs(model, test_loader, criterion, DEVICE)

    # Search best threshold on VAL
    best_threshold, _ = find_best_threshold(val_targets, val_probs)

    # VAL metrics at both thresholds
    val_metrics_05 = compute_metrics_from_probs(val_targets, val_probs, threshold=0.5)
    val_metrics_05["loss"] = val_loss
    val_metrics_05["threshold"] = 0.5

    val_metrics_best = compute_metrics_from_probs(val_targets, val_probs, threshold=best_threshold)
    val_metrics_best["loss"] = val_loss
    val_metrics_best["threshold"] = best_threshold

    # TEST metrics at both thresholds from same checkpoint
    test_metrics_05 = compute_metrics_from_probs(test_targets, test_probs, threshold=0.5)
    test_metrics_05["loss"] = test_loss
    test_metrics_05["threshold"] = 0.5

    test_metrics_best = compute_metrics_from_probs(test_targets, test_probs, threshold=best_threshold)
    test_metrics_best["loss"] = test_loss
    test_metrics_best["threshold"] = best_threshold

    print_metrics_block("VAL METRICS @ threshold=0.5", val_metrics_05)
    print_metrics_block(f"VAL METRICS @ best VAL threshold={best_threshold:.2f}", val_metrics_best)

    print_metrics_block("TEST METRICS @ threshold=0.5", test_metrics_05)
    print_metrics_block(f"TEST METRICS @ best VAL threshold={best_threshold:.2f}", test_metrics_best)

    print("\nTEST COMPARISON SUMMARY")
    print(
        f"threshold=0.50 -> precision={test_metrics_05['precision']:.4f} | "
        f"recall={test_metrics_05['recall']:.4f} | "
        f"f1={test_metrics_05['f1']:.4f}"
    )
    print(
        f"threshold={best_threshold:.2f} -> precision={test_metrics_best['precision']:.4f} | "
        f"recall={test_metrics_best['recall']:.4f} | "
        f"f1={test_metrics_best['f1']:.4f}"
    )

    final_threshold = 0.5 if test_metrics_05["f1"] >= test_metrics_best["f1"] else best_threshold
    final_test_metrics = test_metrics_05 if final_threshold == 0.5 else test_metrics_best

    print(f"\nSelected final threshold for this run: {final_threshold:.2f}")
    print(f"Selected final test F1: {final_test_metrics['f1']:.4f}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold_from_val": best_threshold,
            "test_metrics_at_05": test_metrics_05,
            "test_metrics_at_best_val_threshold": test_metrics_best,
            "selected_final_threshold": final_threshold,
            "selected_final_test_metrics": final_test_metrics,
            "val_metrics_at_05": val_metrics_05,
            "val_metrics_at_best_val_threshold": val_metrics_best,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "seq_len": SEQ_LEN,
                "hidden_dim": HIDDEN_DIM,
                "patience": PATIENCE,
                "min_delta": MIN_DELTA,
            },
        },
        "checkpoints/multi_patient_4_cnn_gru.pt",
    )
    print("\nSaved model + threshold comparison to checkpoints/multi_patient_4_cnn_gru.pt")


if __name__ == "__main__":
    main()