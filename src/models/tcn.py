from __future__ import annotations
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1   = nn.Conv1d(in_channels,  out_channels, kernel_size=kernel_size,
                                  padding=padding, dilation=dilation)
        self.chomp1  = Chomp1d(padding)
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.relu1   = nn.ReLU()
        self.drop1   = nn.Dropout(dropout)

        self.conv2   = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, dilation=dilation)
        self.chomp2  = Chomp1d(padding)
        self.bn2     = nn.BatchNorm1d(out_channels)
        self.relu2   = nn.ReLU()
        self.drop2   = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.bn1(self.chomp1(self.conv1(x)))))
        out = self.drop2(self.relu2(self.bn2(self.chomp2(self.conv2(out)))))
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for EEG seizure detection.

    Changes vs original:
      - Deeper classifier head (Linear→ReLU→Dropout→Linear) matching CNN-LSTM
      - Last-timestep pooling option (use_last_timestep=True) instead of
        AdaptiveAvgPool — the final causal position has seen the full receptive
        field, which is more informative for onset detection than the average
      - Falls back to AdaptiveAvgPool when use_last_timestep=False (default
        kept False so existing checkpoints are not broken)
    """

    def __init__(
        self,
        in_channels:        int         = 23,
        num_channels:       list[int]   = None,
        kernel_size:        int         = 3,
        dropout:            float       = 0.2,
        use_last_timestep:  bool        = False,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128]

        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch     = in_channels if i == 0 else num_channels[i - 1]
            dilation  = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network           = nn.Sequential(*layers)
        self.use_last_timestep = use_last_timestep
        self.pool              = nn.AdaptiveAvgPool1d(1)

        feat_dim = num_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        returns logits: [B]
        """
        if x.ndim != 3:
            raise ValueError(f"TCN expects input [B, C, T], got {tuple(x.shape)}")

        x = self.network(x)                           # [B, F, T]

        if self.use_last_timestep:
            x = x[:, :, -1]                           # [B, F] — final causal position
        else:
            x = self.pool(x).squeeze(-1)              # [B, F] — average over time

        return self.classifier(x).squeeze(-1)         # [B]