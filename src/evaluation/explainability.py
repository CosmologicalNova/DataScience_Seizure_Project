# src/evaluation/explainability.py

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.cnn import SimpleEEGCNN
from src.models.cnn_gru import CNNGRU
from src.models.tcn import TCN


# ── Legacy architecture reconstructed from checkpoint keys/shapes ─────────────

class _AttentionPool(nn.Module):
    """
    Soft attention over time steps → weighted sum → fixed-size vector.
    attn: Linear(128 → 1, bias=True) produces a score per time step.
    Scores are softmax-normalized, then used to sum across time.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attn = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, feature_dim, time)
        x_t = x.permute(0, 2, 1)                    # (batch, time, feature_dim)
        scores = self.attn(x_t)                      # (batch, time, 1)
        weights = torch.softmax(scores, dim=1)       # (batch, time, 1)
        pooled = (x_t * weights).sum(dim=1)          # (batch, feature_dim)
        return pooled


class LegacyCNNLSTM(nn.Module):
    """
    Exact architecture that produced checkpoints like multi_patient_4_cnn_lstm.pt.

    Reconstructed from state dict inspection:
      encoder.features:
        [Conv1d(23,32,7), BN, ReLU, MaxPool,          ← block 0  (indices 0-3)
         Conv1d(32,64,5), BN, ReLU, MaxPool,           ← block 1  (indices 4-7)
         Conv1d(64,128,5), BN, ReLU,                   ← block 2  (indices 8-10)
         Conv1d(128,128,3), BN, ReLU]                  ← block 3  (indices 11-13, no pool)
      encoder.pool:  AttentionPool(128)
      encoder.proj:  Linear(128, 128)
      lstm:          LSTM(input=128, hidden=64, layers=2, batch_first=True)
      classifier:    Linear(64,64) → ReLU → Dropout → Linear(64,1)
    """
    def __init__(self, in_channels: int = 23, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.ModuleDict({
            "features": nn.Sequential(
                # block 0
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                # block 1
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                # block 2
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                # block 3 — no pooling
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            "pool": _AttentionPool(128),
            "proj": nn.Linear(128, 128),
        })

        # hidden=64 confirmed from weight_hh_l0 shape (256, 64) → 4*hidden × hidden
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels, time)  for CNN-LSTM
        #    (batch, channels, time)            for single-window CNN path
        if x.dim() == 4:
            B, S, C, T = x.shape
            x = x.view(B * S, C, T)
            feats = self.encoder["features"](x)          # (B*S, 128, T')
            pooled = self.encoder["pool"](feats)         # (B*S, 128)
            proj = self.encoder["proj"](pooled)          # (B*S, 128)
            proj = proj.view(B, S, 128)                  # (B, S, 128)
            _, (h_n, _) = self.lstm(proj)                # h_n: (2, B, 64)
            h = h_n[-1]                                  # (B, 64)
        else:
            feats = self.encoder["features"](x)
            pooled = self.encoder["pool"](feats)
            proj = self.encoder["proj"](pooled)
            proj = proj.unsqueeze(1)                     # (B, 1, 128)
            _, (h_n, _) = self.lstm(proj)
            h = h_n[-1]

        return self.classifier(h)


# ── Dataset ───────────────────────────────────────────────────────────────────

class EEGSequenceDataset:
    def __init__(self, split_path: Path, seq_len: int):
        data = torch.load(split_path, map_location="cpu", weights_only=False)

        self.X = data["X"]
        self.y = data["y"].float().view(-1)
        self.patient_id = data["patient_id"]
        self.recording_id = data["recording_id"]
        self.window_idx = data["window_idx"]

        self.seq_len = seq_len
        self.samples = []

        n = len(self.X)
        patient_ids   = [str(pid) for pid in (self.patient_id if not isinstance(self.patient_id, str) else [self.patient_id] * n)]
        recording_ids = [str(rid) for rid in self.recording_id]
        window_indices = [int(idx) for idx in self.window_idx]

        groups = defaultdict(list)
        for i in range(n):
            groups[(patient_ids[i], recording_ids[i])].append(i)

        for _, indices in groups.items():
            indices = sorted(indices, key=lambda idx: window_indices[idx])
            if len(indices) < seq_len:
                continue
            for end in range(seq_len - 1, len(indices)):
                self.samples.append(indices[end - seq_len + 1 : end + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_indices = self.samples[idx]
        x_seq = self.X[seq_indices].float()   # (S, C, T)
        y_seq = self.y[seq_indices[-1]]
        return x_seq, y_seq


# ── Model loading ─────────────────────────────────────────────────────────────

def get_checkpoint_value(ckpt, key: str, default):
    if not isinstance(ckpt, dict):
        return default
    if key in ckpt:
        return ckpt[key]
    config = ckpt.get("config")
    if isinstance(config, dict) and key in config:
        return config[key]
    return default


def _is_legacy_checkpoint(state_dict: dict) -> bool:
    """Returns True if state dict uses encoder.* keys (legacy architecture)."""
    return any(k.startswith("encoder.") for k in state_dict)


def load_model(model_name: str, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if model_name == "cnn":
        model = SimpleEEGCNN()
        model.load_state_dict(state_dict)

    elif model_name == "cnn_lstm":
        if _is_legacy_checkpoint(state_dict):
            print("  Detected legacy checkpoint (encoder.* keys) — using LegacyCNNLSTM")
            model = LegacyCNNLSTM(in_channels=23)
        else:
            # Current architecture from src/models/cnn_lstm.py
            from src.models.cnn_lstm import CNNLSTM
            hidden_dim = int(get_checkpoint_value(ckpt, "hidden_dim", 128))
            model = CNNLSTM(
                in_channels=23,
                feature_dim=128,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=0.0,
            )
        model.load_state_dict(state_dict)
    elif model_name == "cnn_gru":
        hidden_dim = int(get_checkpoint_value(ckpt, "hidden_dim", 64))
        num_layers = int(get_checkpoint_value(ckpt, "num_layers", 1))
        dropout = float(get_checkpoint_value(ckpt, "dropout", 0.3))
        model = CNNGRU(
            in_channels=23,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(state_dict)
    elif model_name == "tcn":
        model = TCN(
            in_channels=23,
            num_channels=list(get_checkpoint_value(ckpt, "tcn_channels", [64, 64, 128, 128])),
            kernel_size=int(get_checkpoint_value(ckpt, "kernel_size", 3)),
            dropout=float(get_checkpoint_value(ckpt, "dropout", 0.3)),
            use_last_timestep=bool(get_checkpoint_value(ckpt, "use_last_timestep", False)),
        )
        model.load_state_dict(state_dict)
    else:
        raise ValueError("model must be cnn, cnn_lstm, cnn_gru, or tcn")

    model.to(device)
    model.eval()
    return model, ckpt


# ── Saliency ──────────────────────────────────────────────────────────────────

def compute_saliency(model, x: torch.Tensor, device, model_name: str):
    x = x.to(device).unsqueeze(0)   # add batch dim
    x.requires_grad_(True)

    output = model(x)
    prob = torch.sigmoid(output).squeeze()

    model.zero_grad()
    prob.backward()

    saliency = x.grad.abs().detach().cpu().squeeze(0)
    return saliency, prob.item()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_saliency_cnn(x, saliency, save_path, title):
    x        = x.cpu().numpy()
    saliency = saliency.cpu().numpy()
    channel_importance = saliency.mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].imshow(x,        aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title(f"{title} — EEG Window")
    axes[0].set_ylabel("Channel")
    axes[1].imshow(saliency, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title("Saliency Heatmap")
    axes[1].set_ylabel("Channel")
    axes[2].plot(channel_importance)
    axes[2].set_title("Average Saliency per Channel")
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_saliency_cnn_lstm(x, saliency, save_path, title):
    x        = x.cpu().numpy()        # (S, C, T)
    saliency = saliency.cpu().numpy()

    x_mean         = x.mean(axis=0)                   # (C, T)
    saliency_mean  = saliency.mean(axis=0)             # (C, T)
    seq_importance = saliency.mean(axis=(1, 2))        # (S,)
    chan_importance = saliency.mean(axis=(0, 2))       # (C,)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    axes[0].imshow(x_mean,       aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title(f"{title} — Mean EEG Across Sequence")
    axes[0].set_ylabel("Channel")
    axes[1].imshow(saliency_mean, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title("Mean Saliency Heatmap Across Sequence")
    axes[1].set_ylabel("Channel")
    axes[2].plot(seq_importance)
    axes[2].set_title("Average Saliency per Sequence Step")
    axes[2].set_xlabel("Sequence step")
    axes[2].set_ylabel("Importance")
    axes[3].plot(chan_importance)
    axes[3].set_title("Average Saliency per Channel")
    axes[3].set_xlabel("Channel")
    axes[3].set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=["cnn", "cnn_lstm", "cnn_gru", "tcn"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      required=True)
    parser.add_argument("--sample_idx", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, ckpt = load_model(args.model, args.checkpoint, device)

    if args.model in {"cnn", "tcn"}:
        data   = torch.load(args.split, map_location="cpu", weights_only=False)
        x      = data["X"][args.sample_idx].float()
        target = int(data["y"][args.sample_idx].item())
        print(f"{args.model.upper()} sample shape: {tuple(x.shape)}")
    else:
        seq_len = int(get_checkpoint_value(ckpt, "seq_len", 8))
        ds      = EEGSequenceDataset(Path(args.split), seq_len=seq_len)
        x, y    = ds[args.sample_idx]
        target  = int(y.item())
        print(f"CNN-LSTM sample shape: {tuple(x.shape)}")

    saliency, prob = compute_saliency(model, x, device, args.model)

    output_dir = Path("outputs/evaluation/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{args.model}_sample_{args.sample_idx}.png"

    title = (f"{args.model.upper()} | idx={args.sample_idx} | "
             f"target={target} | prob={prob:.4f}")

    if args.model in {"cnn", "tcn"}:
        plot_saliency_cnn(x, saliency, save_path, title)
    else:
        plot_saliency_cnn_lstm(x, saliency, save_path, title)

    print(f"Saved saliency map to: {save_path}")


if __name__ == "__main__":
    main()
