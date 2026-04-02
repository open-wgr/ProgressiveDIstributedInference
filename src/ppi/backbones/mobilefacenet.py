"""MobileFaceNet backbone — placeholder for future implementation."""

from __future__ import annotations

import torch.nn as nn


class MobileFaceNet(nn.Module):
    """MobileFaceNet with partitioned output heads.

    Not yet implemented — will share the same forward interface as
    ``PartitionedResNet``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("MobileFaceNet not yet implemented")
