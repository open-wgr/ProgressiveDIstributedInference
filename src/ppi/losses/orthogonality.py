"""Orthogonality and diversity regularisation losses for partition outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrthogonalityLoss(nn.Module):
    """Penalise redundancy between partition outputs.

    Two modes:

    ``"cosine"`` (original)
        Gram matrix of L2-normalised partition vectors.  Penalises geometric
        overlap, but can be trivially satisfied in high dimensions without
        actually encouraging information diversity.

    ``"correlation"`` (default)
        Mean-centres each partition's output **across the batch** before
        computing the Gram matrix.  This measures statistical correlation:
        two partitions that encode the same class structure will have
        correlated activations across samples even if their raw vectors
        point in different directions.  Much harder to satisfy trivially.
    """

    def __init__(
        self,
        lambda_orth: float = 0.1,
        mode: str = "correlation",
    ) -> None:
        super().__init__()
        self.lambda_orth = lambda_orth
        if mode not in ("cosine", "correlation"):
            raise ValueError(f"Unknown orthogonality mode: '{mode}'")
        self.mode = mode

    def forward(self, partition_outputs: list[Tensor]) -> Tensor:
        """Compute orthogonality / diversity loss.

        Parameters
        ----------
        partition_outputs : list of Tensor[B, K]
            Raw partition outputs (before positional encoding).
        """
        if self.lambda_orth == 0:
            return torch.tensor(0.0, device=partition_outputs[0].device)

        if len(partition_outputs) <= 1:
            return torch.tensor(0.0, device=partition_outputs[0].device)

        # Stack to [B, N, K]
        stacked = torch.stack(partition_outputs, dim=1)

        if self.mode == "correlation":
            # Centre across the batch dimension so we measure covariation
            centred = stacked - stacked.mean(dim=0, keepdim=True)
            normed = F.normalize(centred, dim=2, eps=1e-12)
        else:
            normed = F.normalize(stacked, dim=2, eps=1e-12)

        # Gram matrix [B, N, N]
        gram = torch.bmm(normed, normed.transpose(1, 2))

        # Penalise off-diagonal elements
        N = gram.size(1)
        eye = torch.eye(N, device=gram.device).unsqueeze(0)
        off_diag = gram * (1.0 - eye)
        loss = (off_diag ** 2).sum() / (gram.size(0) * N * (N - 1))
        return self.lambda_orth * loss
