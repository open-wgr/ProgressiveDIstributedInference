"""PartitionCombiner: mask-conditional MLP combining partition embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PartitionCombiner(nn.Module):
    """Small MLP combining partition embeddings with an explicit presence mask.

    Input:
        embeddings: (B, num_partitions * partition_dim)  — per-partition L2-normed,
                    zero-padded for absent partitions
        mask:       (B, num_partitions)                  — float 0/1 presence flags

    The mask is concatenated to the embedding rather than applied multiplicatively
    so the network can learn subset-conditional projections: zero padding alone is
    ambiguous between "absent" and "present with near-zero embedding."

    Output: L2-normalised combined embedding of shape (B, output_dim).
    No batch norm — partition statistics differ per subset, and BN would couple
    different subsets seen in the same mini-batch.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = num_partitions * partition_dim + num_partitions
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        """Return L2-normalised combined embedding of shape (B, output_dim)."""
        x = torch.cat([embeddings, mask], dim=1)
        return F.normalize(self.net(x), dim=1, eps=1e-12)
