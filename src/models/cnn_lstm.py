"""
src/models/cnn_lstm.py — CNN + LSTM hybrid (primary model)
===========================================================
CNN extracts local features from each compressed time step.
LSTM models how those features evolve over the 5-second window.

This is the model most likely to outperform the CNN baseline because
seizures are TEMPORAL events — their onset is defined by how EEG patterns
change over time, not just what they look like at one instant.

What to change (in configs/config.yaml → cnn_lstm):
    lstm_hidden: 128     → 256 for more memory, 64 for faster training
    lstm_layers: 2       → 1 is often enough for 5s windows
    bidirectional: false → true for offline analysis (better, not real-time)
    dropout: 0.3         → applied between stacked LSTM layers
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(
        self,
        num_channels:   int,
        window_samples: int,
        out_channels:   list,
        kernel_size:    int,
        lstm_hidden:    int,
        lstm_layers:    int,
        bidirectional:  bool,
        dropout:        float,
    ):
        super().__init__()

        # CNN block — identical to CNNBaseline for fair ablation comparison.
        # The ONLY difference between this model and the baseline is the LSTM.
        cnn_layers = []
        in_ch      = num_channels

        for out_ch in out_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM reads CNN feature frames as a temporal sequence.
        # input_size = in_ch (128): each time step is a 128-dim CNN feature vector.
        # batch_first=True: input shape is (batch, seq_len, features).
        # dropout between stacked layers (not applied on final layer — PyTorch behavior).
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.bidirectional  = bidirectional
        lstm_out_size       = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                      # (batch, 128, compressed_time)
        x = x.permute(0, 2, 1)              # (batch, compressed_time, 128) for LSTM
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers * dirs, batch, hidden)

        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]                      # Final hidden state (batch, hidden)

        return self.classifier(h)
