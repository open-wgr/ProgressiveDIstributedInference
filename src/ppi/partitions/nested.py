"""Variant B: Nested/slimmable partition strategy — placeholder."""

from __future__ import annotations

from typing import Any

from ppi.partitions.base import PartitionStrategy


class NestedPartitionStrategy(PartitionStrategy):
    def __init__(self, config: dict[str, Any]) -> None:
        raise NotImplementedError("Nested partition strategy not yet implemented")
