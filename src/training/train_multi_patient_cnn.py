# src/training/train_chb01_cnn.py

from __future__ import annotations
import certifi

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.models.cnn import SimpleEEGCNN

print(f"Using certifi CA bundle at: {certifi.where()}")


def load_cached_split(split_path, batch_size, shuffle):
    data = torch.load(split_path, map_location="cpu")

    X = data["X"].float()
    y = data["y"].float()

    ds = TensorDataset(X, y)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

    return ds, loader


def compute_metrics_from_probs(all_targets, all_probs, threshold=0.5):
    all_preds = [1.0 if p >= threshold else 0.0 for p in all_probs]

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * x.size(0)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics_from_probs(all_targets, all_probs, threshold=threshold)
    metrics["loss"] = avg_loss
    metrics["threshold"] = threshold

    return metrics


@torch.no_grad()
def collect_probs(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * x.size(0)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_targets, all_probs


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
    print("-" * 72)
    print(f"{'thr':>6} {'prec':>8} {'rec':>8} {'f1':>8} {'auc':>8}")
    print("-" * 72)

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

    print("-" * 72)
    print(
        f"Best threshold on VAL: {best_threshold:.2f} | "
        f"precision={best_metrics['precision']:.4f} | "
        f"recall={best_metrics['recall']:.4f} | "
        f"f1={best_metrics['f1']:.4f}"
    )

    return best_threshold, best_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using device:", device)

    batch_size = 64
    epochs = 10
    lr = 1e-3

    base = Path("data/processed/windowed_splits/multi_patient_4")

    train_ds, train_loader = load_cached_split(base / "train.pt", batch_size, shuffle=True)
    val_ds, val_loader = load_cached_split(base / "val.pt", batch_size, shuffle=False)
    test_ds, test_loader = load_cached_split(base / "test.pt", batch_size, shuffle=False)

    train_pos = int(train_ds.tensors[1].sum().item())
    train_neg = len(train_ds) - train_pos
    pos_weight = train_neg / max(train_pos, 1)

    print(f"Train positives: {train_pos}")
    print(f"Train negatives: {train_neg}")
    print(f"pos_weight: {pos_weight:.4f}")

    model = SimpleEEGCNN(in_channels=23).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nBest model selected by val F1 @ threshold=0.5 during training")

    val_loss, val_targets, val_probs = collect_probs(model, val_loader, criterion, device)
    best_threshold, best_val_threshold_metrics = find_best_threshold(val_targets, val_probs)

    val_metrics = compute_metrics_from_probs(val_targets, val_probs, threshold=best_threshold)
    val_metrics["loss"] = val_loss
    val_metrics["threshold"] = best_threshold

    test_loss, test_targets, test_probs = collect_probs(model, test_loader, criterion, device)
    test_metrics = compute_metrics_from_probs(test_targets, test_probs, threshold=best_threshold)
    test_metrics["loss"] = test_loss
    test_metrics["threshold"] = best_threshold

    print("\nVAL METRICS (best threshold from VAL)")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nTEST METRICS (using VAL-selected threshold)")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        "checkpoints/multi_patient_4_cnn.pt",
    )
    print("\nSaved model + threshold to checkpoints/multi_patient_4_cnn.pt")


if __name__ == "__main__":
    main()
