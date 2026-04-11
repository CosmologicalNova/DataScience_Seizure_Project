from __future__ import annotations
import certifi
import os
import torch.nn.functional as F
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.models.tcn import TCN

print(f"Using certifi CA bundle at: {certifi.where()}")


# =========================================================
# CONFIG
# =========================================================
BATCH_SIZE      = 64
EPOCHS          = 50          # was 20 — early stopping handles termination
LR              = 3e-4
THRESHOLD       = 0.5

# Early stopping
PATIENCE        = 10
MIN_DELTA       = 1e-3
PRECISION_FLOOR = 0.75

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
    """
    alpha=0.25 matches CNN-LSTM run 2 (best confirmed config).
    Do NOT raise alpha to push recall — prior experiments showed
    higher alpha degrades precision severely and is a net negative.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t   = torch.exp(-bce)
        alpha = torch.where(targets == 1,
                            torch.full_like(p_t, self.alpha),
                            torch.full_like(p_t, 1 - self.alpha))
        focal = alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


# =========================================================
# DATA
# =========================================================
def load_cached_split(split_path: Path, batch_size: int, shuffle: bool):
    data = torch.load(split_path, map_location="cpu")

    if "X" not in data or "y" not in data:
        raise ValueError(f"Invalid split file format: {split_path}")

    X = data["X"].float()
    y = data["y"].float().view(-1)

    if len(X) != len(y):
        raise ValueError(
            f"Mismatched X/y lengths in {split_path}: "
            f"len(X)={len(X)}, len(y)={len(y)}"
        )
    if X.ndim != 3:
        raise ValueError(
            f"TCN expects [N, C, T], got {tuple(X.shape)} in {split_path}"
        )

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=0, pin_memory=False)

    print(f"{split_path.name}:  X={tuple(X.shape)}  y={tuple(y.shape)}")
    return ds, loader


# =========================================================
# METRICS / EVAL
# =========================================================
def compute_metrics_from_probs(all_targets, all_probs, threshold=0.5):
    y_true = np.array(all_targets, dtype=np.float32)
    y_prob = np.array(all_probs,   dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    try:
        auc    = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        auc = pr_auc = 0.0

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       auc,
        "pr_auc":    pr_auc,   # primary metric under class imbalance
        "cm":        confusion_matrix(y_true, y_pred),
    }


@torch.no_grad()
def collect_probs(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_probs   = []
    all_targets = []

    for X, y in loader:
        X      = X.to(device)
        y      = y.to(device).float().view(-1, 1)
        logits = model(X)
        if logits.ndim == 1:
            logits = logits.view(-1, 1)

        loss = criterion(logits, y)

        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits detected during evaluation.")
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected during evaluation.")

        probs = torch.sigmoid(logits)
        total_loss  += loss.item() * X.size(0)
        all_probs.extend(probs.detach().cpu().view(-1).numpy().tolist())
        all_targets.extend(y.detach().cpu().view(-1).numpy().tolist())

    return total_loss / len(loader.dataset), all_targets, all_probs


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    avg_loss, all_targets, all_probs = collect_probs(model, loader, criterion, device)
    metrics          = compute_metrics_from_probs(all_targets, all_probs, threshold)
    metrics["loss"]  = avg_loss
    metrics["threshold"] = threshold
    return metrics


def find_best_threshold(all_targets, all_probs):
    """Finds threshold maximising F1 on validation set only."""
    candidate_thresholds = [
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.85, 0.90,
        0.92, 0.94, 0.95, 0.96, 0.97,
        0.98, 0.99,
    ]

    best_threshold = 0.5
    best_metrics   = None
    best_f1        = -1.0

    print("\nTHRESHOLD SEARCH ON VAL")
    print("-" * 70)
    print(f"{'thr':>6} {'prec':>8} {'rec':>8} {'f1':>8} {'pr_auc':>8}")
    print("-" * 70)

    for threshold in candidate_thresholds:
        m = compute_metrics_from_probs(all_targets, all_probs, threshold=threshold)
        print(f"{threshold:>6.2f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['pr_auc']:>8.4f}")
        if m["f1"] > best_f1:
            best_f1        = m["f1"]
            best_threshold = threshold
            best_metrics   = m

    print("-" * 70)
    print(f"Best val threshold: {best_threshold:.2f} | "
          f"precision={best_metrics['precision']:.4f} | "
          f"recall={best_metrics['recall']:.4f} | "
          f"f1={best_metrics['f1']:.4f}")
    return best_threshold, best_metrics


def print_metrics_block(title, metrics):
    print(f"\n{title}")
    for k, v in metrics.items():
        if k == "cm":
            print(f"  confusion_matrix:\n{v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# =========================================================
# TRAIN
# =========================================================
def main():
    print(f"Using device: {DEVICE}")

    train_ds, train_loader = load_cached_split(TRAIN_PATH, BATCH_SIZE, True)
    val_ds,   val_loader   = load_cached_split(VAL_PATH,   BATCH_SIZE, False)
    test_ds,  test_loader  = load_cached_split(TEST_PATH,  BATCH_SIZE, False)

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if len(ds) == 0:
            raise ValueError(f"{name} dataset has 0 samples.")

    train_labels = train_ds.tensors[1]
    num_pos = int((train_labels == 1).sum().item())
    num_neg = int((train_labels == 0).sum().item())
    print(f"Train positives: {num_pos}  |  negatives: {num_neg}  |  "
          f"seizure rate: {100*num_pos/(num_pos+num_neg):.2f}%")

    sample_x, _ = train_ds[0]
    model = TCN(
        in_channels=sample_x.shape[0],
        num_channels=[64, 64, 128, 128],
        kernel_size=3,
        dropout=0.3,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TCN parameters: {n_params:,}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # AdamW — proper weight decay decoupling, matches CNN-LSTM run 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler and early stopping both monitor val F1 — consistent with CNN-LSTM
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_f1       = -1.0
    epochs_no_improve = 0

    best_path = Path("checkpoints/best_multi_patient_tcn_4.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch:02d}/{EPOCHS}",
                          leave=False, dynamic_ncols=True)

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
        val_m      = evaluate(model, val_loader, criterion, DEVICE, threshold=0.5)

        val_f1        = val_m["f1"]
        val_recall    = val_m["recall"]
        val_precision = val_m["precision"]
        current_lr    = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_m['loss']:.4f} | "
            f"val_f1={val_f1:.4f} | val_recall={val_recall:.4f} | "
            f"val_precision={val_precision:.4f} | val_pr_auc={val_m['pr_auc']:.4f}"
        )

        scheduler.step(val_f1)

        qualifies = val_precision >= PRECISION_FLOOR
        improves  = val_f1 > best_val_f1 + MIN_DELTA

        if qualifies and improves:
            best_val_f1       = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best | val_f1={best_val_f1:.4f} | "
                  f"val_recall={val_recall:.4f} | val_precision={val_precision:.4f}")
        else:
            epochs_no_improve += 1
            if not qualifies:
                print(f"  Precision floor not met "
                      f"({val_precision:.4f} < {PRECISION_FLOOR}) "
                      f"({epochs_no_improve}/{PATIENCE})")
            else:
                print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val_f1={best_val_f1:.4f})")
                break

    # ── Post-training evaluation ───────────────────────────────────────────────
    print("\nLoading best checkpoint...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    val_loss,  val_targets,  val_probs  = collect_probs(
        model, val_loader,  criterion, DEVICE)
    test_loss, test_targets, test_probs = collect_probs(
        model, test_loader, criterion, DEVICE)

    # Threshold selected on val only — never use test set to pick threshold
    best_threshold, _ = find_best_threshold(val_targets, val_probs)

    val_metrics_05    = compute_metrics_from_probs(val_targets,  val_probs,  0.5)
    val_metrics_best  = compute_metrics_from_probs(val_targets,  val_probs,  best_threshold)
    test_metrics_05   = compute_metrics_from_probs(test_targets, test_probs, 0.5)
    test_metrics_best = compute_metrics_from_probs(test_targets, test_probs, best_threshold)

    for m, loss, thr in [
        (val_metrics_05,   val_loss,  0.5),
        (val_metrics_best, val_loss,  best_threshold),
        (test_metrics_05,  test_loss, 0.5),
        (test_metrics_best,test_loss, best_threshold),
    ]:
        m["loss"]      = loss
        m["threshold"] = thr

    print_metrics_block("VAL  @ threshold=0.50",                          val_metrics_05)
    print_metrics_block(f"VAL  @ best val threshold={best_threshold:.2f}",val_metrics_best)
    print_metrics_block("TEST @ threshold=0.50",                          test_metrics_05)
    print_metrics_block(f"TEST @ best val threshold={best_threshold:.2f}",test_metrics_best)

    # Final reported numbers use the val-selected threshold only
    print(f"\nFINAL TEST RESULTS (threshold={best_threshold:.2f}, selected from val)")
    print(f"  precision : {test_metrics_best['precision']:.4f}")
    print(f"  recall    : {test_metrics_best['recall']:.4f}")
    print(f"  f1        : {test_metrics_best['f1']:.4f}")
    print(f"  pr_auc    : {test_metrics_best['pr_auc']:.4f}")

    torch.save(
        {
            "model_state_dict":                   model.state_dict(),
            "best_threshold_from_val":            best_threshold,
            "test_metrics_at_05":                 test_metrics_05,
            "test_metrics_at_best_val_threshold": test_metrics_best,
            "val_metrics_at_05":                  val_metrics_05,
            "val_metrics_at_best_val_threshold":  val_metrics_best,
            "config": {
                "batch_size":      BATCH_SIZE,
                "epochs":          EPOCHS,
                "lr":              LR,
                "patience":        PATIENCE,
                "min_delta":       MIN_DELTA,
                "precision_floor": PRECISION_FLOOR,
                "focal_alpha":     0.25,
                "focal_gamma":     2.0,
                "tcn_channels":    [64, 64, 128, 128],
                "kernel_size":     3,
                "dropout":         0.3,
            },
        },
        "checkpoints/multi_patient_4_tcn.pt",
    )

    final_model_path = Path("checkpoints/final_multi_patient_tcn_4.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model state dict saved to {final_model_path}")
    print("Full checkpoint saved to checkpoints/multi_patient_4_tcn.pt")


if __name__ == "__main__":
    main()