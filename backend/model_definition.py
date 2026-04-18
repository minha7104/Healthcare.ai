from __future__ import annotations

import torch.nn as nn


class TabularBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 32,
        hidden2: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.network(x)
