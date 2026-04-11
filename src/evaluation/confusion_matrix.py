# src/evaluation/evaluate_confusion_matrix.py

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Support direct execution like:
#   python src/evaluation/evaluate_confusion_matrix.py ...
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.cnn import SimpleEEGCNN
from src.models.cnn_lstm import CNNLSTM
from src.models.cnn_gru import CNNGRU
from src.models.tcn import TCN


def build_model(model_name: str, checkpoint: dict | None = None) -> torch.nn.Module:
    model_name = model_name.lower()

    if model_name == "cnn":
        return SimpleEEGCNN()

    elif model_name == "cnn_lstm":
        hidden_dim = 64
        feature_dim = 128
        num_layers = 2
        dropout = 0.2
        in_channels = 23

        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})
            hidden_dim = int(config.get("hidden_dim", hidden_dim))
            feature_dim = int(config.get("feature_dim", feature_dim))
            num_layers = int(config.get("num_layers", num_layers))
            dropout = float(config.get("dropout", dropout))

        return CNNLSTM(
            in_channels=in_channels,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "cnn_gru":
        hidden_dim = 64
        num_layers = 1
        dropout = 0.3
        in_channels = 23

        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})
            hidden_dim = int(config.get("hidden_dim", hidden_dim))
            num_layers = int(config.get("num_layers", num_layers))
            dropout = float(config.get("dropout", dropout))

        return CNNGRU(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "tcn":
        in_channels = 23
        num_channels = [64, 64, 128, 128]
        kernel_size = 3
        dropout = 0.3
        use_last_timestep = False

        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})
            num_channels = list(config.get("tcn_channels", num_channels))
            kernel_size = int(config.get("kernel_size", kernel_size))
            dropout = float(config.get("dropout", dropout))
            use_last_timestep = bool(config.get("use_last_timestep", use_last_timestep))

        return TCN(
            in_channels=in_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_last_timestep=use_last_timestep,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_split(split_path: Path):
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    data = torch.load(split_path, map_location="cpu")

    if "X" not in data or "y" not in data:
        raise ValueError(f"Split file missing X/y: {split_path}")

    X = data["X"].float()
    y = data["y"].float().view(-1)
    patient_ids = data.get("patient_id", None)

    return X, y, patient_ids, data


class EEGSequenceDataset(Dataset):
    """
    Rebuilds sequences exactly like training:
    - groups by (patient_id, recording_id)
    - sorts by window_idx
    - creates rolling sequences of length seq_len
    - uses the label of the last window in the sequence
    """

    def __init__(self, data: dict, seq_len: int):
        self.X = data["X"].float()
        self.y = data["y"].float().view(-1)
        self.patient_id = data["patient_id"]
        self.recording_id = data["recording_id"]
        self.window_idx = data["window_idx"]
        self.seq_len = seq_len
        self.samples = []

        n = len(self.X)

        if isinstance(self.patient_id, str):
            patient_ids = [self.patient_id] * n
        else:
            patient_ids = [str(pid) for pid in self.patient_id]

        recording_ids = [str(rid) for rid in self.recording_id]
        window_indices = [int(idx) for idx in self.window_idx]

        groups = {}
        for i in range(n):
            key = (patient_ids[i], recording_ids[i])
            groups.setdefault(key, []).append(i)

        for indices in groups.values():
            indices = sorted(indices, key=lambda idx: window_indices[idx])

            if len(indices) < seq_len:
                continue

            for end in range(seq_len - 1, len(indices)):
                seq_indices = indices[end - seq_len + 1 : end + 1]
                self.samples.append(seq_indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_indices = self.samples[idx]
        x_seq = self.X[seq_indices]               # [S, C, T]
        y_seq = self.y[seq_indices[-1]]           # label of last window

        if isinstance(self.patient_id, str):
            pid = self.patient_id
        else:
            pid = self.patient_id[seq_indices[-1]]

        return x_seq, y_seq, pid


@torch.no_grad()
def predict(model, X, batch_size: int, device: torch.device):
    model.eval()
    probs = []

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        xb = X[start:end].to(device)

        logits = model(xb)

        if logits.ndim > 1:
            logits = logits.view(-1)

        batch_probs = torch.sigmoid(logits).cpu()
        probs.append(batch_probs)

    probs = torch.cat(probs, dim=0).numpy()
    return probs


@torch.no_grad()
def predict_sequences(
    model,
    data: dict,
    seq_len: int,
    batch_size: int,
    device: torch.device,
):
    model.eval()

    ds = EEGSequenceDataset(data, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    probs = []
    targets = []
    patient_ids = []

    print(f"Sequence dataset built: {len(ds)} sequences")
    print(f"Sequence length used: {seq_len}")

    for xb, yb, pid in loader:
        xb = xb.to(device)

        logits = model(xb)
        if logits.ndim > 1:
            logits = logits.view(-1)

        probs.append(torch.sigmoid(logits).cpu())
        targets.append(yb.view(-1).cpu())

        if isinstance(pid, torch.Tensor):
            patient_ids.extend(pid.cpu().numpy().tolist())
        else:
            patient_ids.extend(list(pid))

    probs = torch.cat(probs, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy().astype(int)

    return probs, targets, patient_ids


def save_confusion_matrix_plot(cm, labels, title: str, out_path: Path, normalize: bool = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    values_format = ".3f" if normalize else "d"
    disp.plot(ax=ax, cmap="Blues", values_format=values_format, colorbar=False)

    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "cnn_lstm", "cnn_gru", "tcn"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--outdir", type=str, default="outputs/evaluation")
    
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda, mps, or cpu). If None, auto-detects.")
    
    args = parser.parse_args()


    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint)
    split_path = Path(args.split_path)
    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading split: {split_path}")
    X, y, patient_ids, raw_data = load_split(split_path)

    print(f"Raw X shape: {tuple(X.shape)}")
    print(f"Raw y shape: {tuple(y.shape)}")
    print(f"Raw positives: {int(y.sum().item())}")
    print(f"Raw negatives: {int((y == 0).sum().item())}")

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    print(f"\nBuilding model: {args.model}")
    model = build_model(args.model, checkpoint=ckpt if isinstance(ckpt, dict) else None)

    # Handle either:
    # - raw state_dict
    # - dict with model_state_dict
    # - dict with state_dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("\n[ERROR] Failed to load state_dict into model.")
        print("This usually means the checkpoint architecture does not match the current model.")
        print("Checkpoint path:", checkpoint_path)
        print("Model selected:", args.model)
        print("Error details:\n", e)
        raise

    model.to(device)

    print("\nRunning inference...")

    if args.model in {"cnn_lstm", "cnn_gru"}:
        if "recording_id" not in raw_data or "window_idx" not in raw_data:
            raise ValueError(
                f"{args.model} evaluation requires recording_id and window_idx in the split file."
            )

        if isinstance(ckpt, dict):
            seq_len = int(ckpt.get("config", {}).get("seq_len", 8))
        else:
            seq_len = 8

        probs, targets, patient_ids_for_save = predict_sequences(
            model=model,
            data=raw_data,
            seq_len=seq_len,
            batch_size=args.batch_size,
            device=device,
        )

    else:
        probs = predict(model, X, batch_size=args.batch_size, device=device)
        targets = y.numpy().astype(int)
        patient_ids_for_save = (
            patient_ids.cpu().numpy() if isinstance(patient_ids, torch.Tensor) else patient_ids
        )

    preds = (probs >= args.threshold).astype(int)

    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)

    print("\n================ CONFUSION MATRIX ================")
    print(cm)
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    print("\n================ METRICS ================")
    print(f"Threshold : {args.threshold:.3f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    cm_norm = confusion_matrix(targets, preds, normalize="true")

    save_confusion_matrix_plot(
        cm,
        labels=["Non-seizure", "Seizure"],
        title=f"{args.model.upper()} Confusion Matrix",
        out_path=outdir / "confusion_matrix_counts.png",
        normalize=False,
    )

    save_confusion_matrix_plot(
        cm_norm,
        labels=["Non-seizure", "Seizure"],
        title=f"{args.model.upper()} Confusion Matrix (Normalized)",
        out_path=outdir / "confusion_matrix_normalized.png",
        normalize=True,
    )

    save_dict = {
        "probs": probs,
        "preds": preds,
        "targets": targets,
        "threshold": args.threshold,
        "patient_id": patient_ids_for_save,
        "split_path": str(split_path),
        "checkpoint": str(checkpoint_path),
        "model": args.model,
    }
    torch.save(save_dict, outdir / "predictions.pt")

    print(f"\nSaved:")
    print(f"  {outdir / 'confusion_matrix_counts.png'}")
    print(f"  {outdir / 'confusion_matrix_normalized.png'}")
    print(f"  {outdir / 'predictions.pt'}")


if __name__ == "__main__":
    main()
