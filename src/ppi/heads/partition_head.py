"""Partition head: BN-FC-BN projection for per-partition embeddings."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PartitionHead(nn.Module):
    """BN → Linear → BN projection producing a K-dim partition embedding."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn2(self.fc(self.bn1(x)))
