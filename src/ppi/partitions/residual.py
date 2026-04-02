"""Variant C: Residual boosting partition strategy — placeholder."""

from __future__ import annotations

from typing import Any

from ppi.partitions.base import PartitionStrategy


class ResidualPartitionStrategy(PartitionStrategy):
    def __init__(self, config: dict[str, Any]) -> None:
        raise NotImplementedError("Residual partition strategy not yet implemented")
