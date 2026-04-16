from __future__ import annotations

import torch.nn as nn


class BinaryLinearHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_hidden_states):
        return self.classifier(self.dropout(pooled_hidden_states))
