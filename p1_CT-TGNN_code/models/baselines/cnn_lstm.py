"""
CNN-LSTM for Encrypted Traffic Analysis

Baseline: Spatial CNN + Temporal LSTM without graph structure
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN-LSTM baseline for sequence classification."""

    def __init__(self, input_dim=87, hidden_dim=256, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, **kwargs):
        # x: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]

        h, _ = self.lstm(x)
        h = h[:, -1, :]  # Last timestep

        logits = self.classifier(h)
        return {'logits': logits, 'embeddings': h}
