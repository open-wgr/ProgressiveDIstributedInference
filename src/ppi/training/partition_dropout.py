"""Partition dropout and embedding assembly."""

from __future__ import annotations

import random as _random
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PartitionDropout(nn.Module):
    """Stochastically zero entire partitions during training.

    The dropout distribution is ``[p_1part, p_2part, p_3part, p_0part]``
    controlling how often each width configuration is sampled.  The same
    mask is applied to every sample in the batch.

    In eval mode, all partitions are passed through unchanged.
    """

    def __init__(
        self,
        num_partitions: int = 3,
        distribution: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_partitions = num_partitions
        if distribution is None:
            distribution = [0.4, 0.3, 0.2, 0.1]
        if abs(sum(distribution) - 1.0) > 1e-6:
            raise ValueError(
                f"Dropout distribution must sum to 1.0, got {sum(distribution)}"
            )
        self.distribution = distribution
        self._last_chosen_width: int = num_partitions

        # Pre-compute all possible configs per width category
        indices = list(range(num_partitions))
        self._configs_by_width: list[list[set[int]]] = []
        # width 1, 2, ..., num_partitions
        for w in range(1, num_partitions + 1):
            self._configs_by_width.append(
                [set(c) for c in combinations(indices, w)]
            )
        # width 0 (all dropped)
        # distribution order: [1-part, 2-part, ..., N-part, 0-part]

    @property
    def last_chosen_width(self) -> int:
        """The width category sampled in the last forward call."""
        return self._last_chosen_width

    def forward(self, partition_outputs: list[Tensor]) -> list[Tensor]:
        if not self.training:
            self._last_chosen_width = self.num_partitions
            return partition_outputs

        # Sample a width category
        # distribution: [p_1, p_2, ..., p_N, p_0]
        r = _random.random()
        cumsum = 0.0
        chosen_width = 0  # default: all dropped
        for i, p in enumerate(self.distribution):
            cumsum += p
            if r < cumsum:
                if i < self.num_partitions:
                    chosen_width = i + 1  # 1-part, 2-part, etc.
                else:
                    chosen_width = 0  # 0-part
                break

        self._last_chosen_width = chosen_width

        if chosen_width == 0:
            # All partitions dropped — use p * 0.0 instead of zeros_like
            # to preserve the autograd graph for gradient flow
            return [p * 0.0 for p in partition_outputs]

        # Pick a random config of that width
        configs = self._configs_by_width[chosen_width - 1]
        active = _random.choice(configs)

        result = []
        for idx, p in enumerate(partition_outputs):
            if idx in active:
                result.append(p)
            else:
                result.append(p * 0.0)
        return result


def assemble_embedding(partition_outputs: list[Tensor]) -> Tensor:
    """Concatenate partition outputs and L2-normalise the result.

    Handles the all-zeros case gracefully (returns zeros, no NaN).

    Parameters
    ----------
    partition_outputs:
        List of ``Tensor[B, K]`` — one per partition.  Missing partitions
        should already be zero-tensors.

    Returns
    -------
    Tensor[B, N*K]
        L2-normalised assembled embedding.
    """
    full = torch.cat(partition_outputs, dim=1)
    return F.normalize(full, dim=1, eps=1e-12)
