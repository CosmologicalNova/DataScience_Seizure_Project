from __future__ import annotations

import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    def __init__(
        self,
        in_channels: int = 23,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.feature_dim = 128

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, C, T]
        returns: [B, 1]
        """
        if x.ndim != 4:
            raise ValueError(f"CNNGRU expects input [B, S, C, T], got {tuple(x.shape)}")

        b, s, c, t = x.shape

        x = x.view(b * s, c, t)          # [B*S, C, T]
        feats = self.cnn(x)              # [B*S, 128, 1]
        feats = feats.squeeze(-1)        # [B*S, 128]
        feats = feats.view(b, s, -1)     # [B, S, 128]

        _, h_n = self.gru(feats)         # h_n: [num_layers, B, hidden_dim]
        last_hidden = h_n[-1]            # [B, hidden_dim]

        logits = self.classifier(last_hidden)   # [B, 1]
        return logits