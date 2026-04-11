from __future__ import annotations
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 128, T']
        x = x.permute(0, 2, 1)                          # [B, T', 128]
        weights = torch.softmax(self.attn(x), dim=1)    # [B, T', 1]
        return (x * weights).sum(dim=1)                  # [B, 128]


class WindowCNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 23, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = AttentionPool(128)
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        returns: [B, feature_dim]
        """
        x = self.features(x)    # [B, 128, T']
        x = self.pool(x)        # [B, 128]
        x = self.proj(x)        # [B, feature_dim]
        return x


class CNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 23,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = WindowCNNEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, C, T]
        returns: [B]
        """
        B, S, C, T = x.shape
        x = x.view(B * S, C, T)         # [B*S, C, T]
        feats = self.encoder(x)         # [B*S, feature_dim]
        feats = feats.view(B, S, -1)    # [B, S, feature_dim]
        lstm_out, _ = self.lstm(feats)  # [B, S, hidden_dim]
        last_out = lstm_out[:, -1, :]   # [B, hidden_dim]
        logits = self.classifier(last_out).squeeze(-1)  # [B]
        return logits