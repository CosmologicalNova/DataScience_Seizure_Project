"""
src/models/cnn_baseline.py — CNN-only ablation baseline
=========================================================
Purpose: Performance floor. If temporal models beat this, they add real value.

Architecture:
    Input (batch, 23, 1280)
    → 3× [Conv1d → BatchNorm → ReLU → MaxPool]
    → AdaptiveAvgPool1d(1)
    → Dropout → Linear(64) → ReLU → Linear(1)

Why this is the baseline:
    The CNN learns to recognize LOCAL EEG patterns (spike shapes, ripples)
    at each compressed time step. It does NOT model how those patterns
    evolve over the 5-second window — that's what LSTM/GRU/TCN adds.

    After AdaptiveAvgPool, all temporal ordering information is discarded.
    The model answers "were there seizure-like shapes anywhere in this window?"
    but NOT "were they building progressively like an ictal onset?"

What to change (in configs/config.yaml → cnn_baseline):
    out_channels: [32,64,128]    → add more entries or increase values for capacity
    kernel_size: 5               → larger = each filter sees wider local context
    dropout: 0.3                 → increase to 0.5 if train metrics >> val metrics
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):

    def __init__(
        self,
        num_channels:  int,
        window_samples: int,
        out_channels:  list,
        kernel_size:   int,
        dropout:       float,
    ):
        """
        Args:
            num_channels:   Number of EEG electrode channels (23 for CHB-MIT)
            window_samples: Samples per window (window_size_sec × sampling_rate = 1280)
            out_channels:   Filters per conv layer, e.g. [32, 64, 128]
            kernel_size:    Temporal filter width (must be odd for symmetric padding)
            dropout:        Dropout probability applied after pooling and in classifier
        """
        super().__init__()

        layers = []
        in_ch  = num_channels

        for out_ch in out_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        # Collapses entire remaining time axis into one value per channel.
        # Makes the model length-agnostic (can handle different window sizes).
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),   # Raw logit — no sigmoid (BCEWithLogitsLoss handles it)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_channels, window_samples)
        Returns:
            logit: (batch, 1)
        """
        x = self.cnn(x)
        x = self.pool(x)
        return self.classifier(x)
