"""Orthogonality regularisation loss for partition outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrthogonalityLoss(nn.Module):
    """Penalise correlation between partition outputs via Gram matrix.

    Stacks partition outputs into ``[B, N, K]``, L2-normalises each
    partition, computes the ``N×N`` Gram matrix, and penalises the
    squared off-diagonal elements.
    """

    def __init__(self, lambda_orth: float = 0.1) -> None:
        super().__init__()
        self.lambda_orth = lambda_orth

    def forward(self, partition_outputs: list[Tensor]) -> Tensor:
        """Compute orthogonality loss.

        Parameters
        ----------
        partition_outputs : list of Tensor[B, K]
            Raw partition outputs (before positional encoding).
        """
        # Stack to [B, N, K]
        stacked = torch.stack(partition_outputs, dim=1)
        # L2-normalise each partition independently
        normed = F.normalize(stacked, dim=2, eps=1e-12)
        # Gram matrix [B, N, N]
        gram = torch.bmm(normed, normed.transpose(1, 2))
        # Penalise off-diagonal elements
        N = gram.size(1)
        eye = torch.eye(N, device=gram.device).unsqueeze(0)
        off_diag = gram * (1.0 - eye)
        loss = (off_diag ** 2).sum() / (gram.size(0) * N * (N - 1))
        return self.lambda_orth * loss
