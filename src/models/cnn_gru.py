"""
src/models/cnn_gru.py — CNN + GRU hybrid
==========================================
Identical to CNN+LSTM but GRU replaces LSTM.

GRU vs LSTM:
    GRU has 2 gates (reset, update) vs LSTM's 3 (input, forget, output).
    GRU has no separate cell state — only hidden state.
    Result: ~30% fewer parameters, similar accuracy on most tasks.
    Trains faster. Good baseline to test whether LSTM's extra complexity is worth it.

    Key code difference from CNNLSTM:
        GRU returns: (output, h_n)
        LSTM returns: (output, (h_n, c_n))  ← extra cell state
    So the forward() call unpacks differently.

What to change (in configs/config.yaml → cnn_gru):
    Same as cnn_lstm: gru_hidden, gru_layers, bidirectional, dropout
"""

import torch
import torch.nn as nn


class CNNGRU(nn.Module):

    def __init__(
        self,
        num_channels:   int,
        window_samples: int,
        out_channels:   list,
        kernel_size:    int,
        gru_hidden:     int,
        gru_layers:     int,
        bidirectional:  bool,
        dropout:        float,
    ):
        super().__init__()

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

        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.bidirectional = bidirectional
        gru_out_size       = gru_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)               # GRU: no cell state, just h_n

        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]

        return self.classifier(h)
