"""Abstract partition strategy interface and default implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class PartitionStrategy(ABC):
    """Interface that all partition variants implement.

    The trainer calls these hooks without knowing which variant is running.
    """

    def compute_auxiliary_loss(self, partition_outputs: list[Tensor]) -> Tensor:
        """Return variant-specific auxiliary loss (e.g. orthogonality).

        Default returns zero.
        """
        return torch.tensor(0.0)

    def pre_training_setup(self, model: nn.Module, config: dict[str, Any]) -> None:
        """Called once before the training loop starts. Default no-op."""

    def get_trainable_parameters(
        self,
        model: nn.Module,
        phase: int | None = None,
    ) -> list[nn.Parameter]:
        """Return parameters that should be optimised. Default: all."""
        return list(model.parameters())

    def process_partitions(self, partition_outputs: list[Tensor]) -> list[Tensor]:
        """Transform partition outputs before dropout/assembly. Default: identity."""
        return partition_outputs

    def post_epoch_hook(self, epoch: int, model: nn.Module) -> None:
        """Called at the end of each epoch. Default no-op."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PartitionStrategy:
        """Factory: build the right strategy from the config's ``variant`` key."""
        variant = config.get("variant")
        if variant is None:
            return DefaultStrategy()
        if variant == "orthogonal":
            from ppi.partitions.orthogonal import OrthogonalPartitionStrategy
            return OrthogonalPartitionStrategy(config)
        if variant == "nested":
            from ppi.partitions.nested import NestedPartitionStrategy
            return NestedPartitionStrategy(config)
        if variant == "residual":
            from ppi.partitions.residual import ResidualPartitionStrategy
            return ResidualPartitionStrategy(config)
        if variant == "combined":
            from ppi.partitions.combined import CombinedPartitionStrategy
            return CombinedPartitionStrategy(config)
        raise ValueError(f"Unknown variant: '{variant}'")


class DefaultStrategy(PartitionStrategy):
    """No-op strategy used when no variant is specified."""
