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
        self.orth_loss = OrthogonalityLoss(lambda_orth=orth_cfg.get("lambda", 0.1))

        # Learned positional encoding
        pos_cfg = config.get("positional_encoding", {})
        pos_type = pos_cfg.get("type", "learned")
        if pos_type == "learned":
            self.pos_embedding = nn.Embedding(self.num_partitions, K)
        else:
            raise ValueError(f"Unsupported positional encoding type: {pos_type}")

    def process_partitions(self, partition_outputs: list[Tensor]) -> list[Tensor]:
        """Add learned positional encoding element-wise."""
        result = []
        for idx, p in enumerate(partition_outputs):
            pos = self.pos_embedding(
                torch.tensor(idx, device=p.device)
            )
            result.append(p + pos)
        return result

    def compute_auxiliary_loss(self, partition_outputs: list[Tensor]) -> Tensor:
        """Orthogonality loss on raw partition outputs."""
        return self.orth_loss(partition_outputs)

    def get_trainable_parameters(
        self,
        model: nn.Module,
        phase: int | None = None,
    ) -> list[nn.Parameter]:
        """All model params + positional embedding params."""
        return list(model.parameters())
