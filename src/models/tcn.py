"""
src/models/tcn.py — Temporal Convolutional Network (TCN)
=========================================================
Fully parallel alternative to LSTM/GRU using dilated causal convolutions.
Based on: Bai et al. (2018) https://arxiv.org/abs/1803.01271

Key advantage over LSTM:
    All time steps processed simultaneously — no sequential dependency.
    Faster training on GPU. Stable gradients via residual connections.

Key concept — dilation:
    Block 0: dilation=1 → filter sees [t-2, t-1, t]
    Block 1: dilation=2 → filter sees [t-4, t-2, t]      (wider view, same params)
    Block 2: dilation=4 → filter sees [t-8, t-4, t]
    Block 3: dilation=8 → filter sees [t-16, t-8, t]
    Each block doubles the receptive field.

What to change (in configs/config.yaml → tcn):
    num_channels: [64,64,64,64]   → add more entries to grow receptive field
    kernel_size: 3                → 5 or 7 for faster RF growth per block
    dropout: 0.2                  → keep lower than LSTM, TCN is less prone to overfit
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _Chomp1d(nn.Module):
    """Removes right-side causal padding to enforce that position t only sees t and earlier."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """
    One TCN residual block: 2× dilated causal Conv1d + residual connection.

    Weight normalization (weight_norm) instead of BatchNorm:
        BatchNorm depends on batch statistics — awkward for sequential data.
        Weight_norm normalizes the filter weights directly, more stable for TCN.
        Replace with BatchNorm1d if you find weight_norm unstable.

    Residual connection:
        Same idea as ResNet. output = ReLU(conv(x) + x).
        If input/output channels differ, a 1×1 conv adapts the residual branch.
        Prevents vanishing gradients in deep stacks.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_inputs,  n_outputs, kernel_size,
                                  padding=padding, dilation=dilation)),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                  padding=padding, dilation=dilation)),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if hasattr(m, "weight"):
                m.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):

    def __init__(
        self,
        num_channels:   int,
        window_samples: int,
        num_block_channels: list,
        kernel_size:    int,
        dropout:        float,
    ):
        """
        Args:
            num_block_channels: Filters per block, e.g. [64, 64, 64, 64].
                                 Dilation doubles each block: 1, 2, 4, 8, ...
        """
        super().__init__()

        blocks = []
        in_ch  = num_channels

        for i, out_ch in enumerate(num_block_channels):
            dilation = 2 ** i
            blocks.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn  = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.pool(x)
        return self.classifier(x)
