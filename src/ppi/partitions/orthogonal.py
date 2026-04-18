"""Variant A: Orthogonal partition strategy with learned positional encoding."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ppi.losses.orthogonality import OrthogonalityLoss
from ppi.partitions.base import PartitionStrategy


class OrthogonalPartitionStrategy(PartitionStrategy, nn.Module):
    """Orthogonal partitions with learned positional encoding.

    - Adds a learned ``nn.Embedding(num_partitions, K)`` element-wise
      to each partition's output before assembly.
    - Computes an orthogonality regularisation loss on the *raw*
      (pre-encoding) partition outputs.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        nn.Module.__init__(self)
        partitions_cfg = config["partitions"]
        self.num_partitions = partitions_cfg["num_partitions"]
        K = partitions_cfg["K"]

        orth_cfg = config.get("orthogonality", {})
        self.orth_loss = OrthogonalityLoss(
            lambda_orth=orth_cfg.get("lambda", 0.1),
            mode=orth_cfg.get("mode", "correlation"),
        )

        # Learned positional encoding
        pos_cfg = config.get("positional_encoding", {})
        pos_type = pos_cfg.get("type", "learned")
        if pos_type == "learned":
            self.pos_embedding = nn.Embedding(self.num_partitions, K)
        else:
            raise ValueError(f"Unsupported positional encoding type: {pos_type}")
        self.register_buffer(
            "_pos_indices", torch.arange(self.num_partitions),
        )

    def process_partitions(self, partition_outputs: list[Tensor]) -> list[Tensor]:
        """Add learned positional encoding element-wise."""
        pos_all = self.pos_embedding(self._pos_indices)
        result = []
        for idx, p in enumerate(partition_outputs):
            result.append(p + pos_all[idx])
        return result

    def compute_auxiliary_loss(self, partition_outputs: list[Tensor]) -> Tensor:
        """Orthogonality loss on raw partition outputs."""
        return self.orth_loss(partition_outputs)

    def get_trainable_parameters(
        self,
        model: nn.Module,
        phase: int | None = None,
    ) -> list[nn.Parameter]:
        """Backbone parameters only; strategy params are added by the trainer."""
        return list(model.parameters())
