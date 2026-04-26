"""ResNet backbone with modified stem and partitioned output heads."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from torch import Tensor
from torchvision import models

from ppi.heads.partition_head import PartitionHead

_BACKBONE_DIM = {
    "resnet18": 512,
    "resnet50": 2048,
}

_MODEL_FN = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


class PartitionedResNet(nn.Module):
    """ResNet backbone with a small-input stem and partitioned output heads.

    The standard 7×7 stride-2 conv + maxpool is replaced with a 3×3 stride-1
    conv (no maxpool) so the network works on 112×112 face crops or 32×32
    CIFAR images without losing too much spatial information.

    Each partition head output is L2-normalised before being returned.  This
    makes every partition a unit vector in K-dim space regardless of which
    other partitions are active, so the assembled embedding's distribution is
    determined solely by the count of active partitions — not their identities.
    This is the correct behaviour for arbitrary-subset inference (any subset
    containing P0 must work, including non-prefix failure-mode configs).

    Gradient checkpointing (layer1–layer4) is available via the
    ``gradient_checkpointing`` flag.  It trades ~2× backward compute for
    ~4× reduction in activation memory, allowing larger batch sizes.
    Only active during training; eval is unaffected.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_partitions: int = 3,
        K: int = 128,
        pretrained: bool = False,
        input_size: int = 112,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if backbone_name not in _MODEL_FN:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {sorted(_MODEL_FN)}"
            )

        self._backbone_dim = _BACKBONE_DIM[backbone_name]
        self._grad_checkpoint = gradient_checkpointing

        base = _MODEL_FN[backbone_name](
            weights="IMAGENET1K_V1" if pretrained else None,
        )

        # Replace the aggressive stem with a face-friendly 3×3 conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.partition_heads = nn.ModuleList(
            [PartitionHead(self._backbone_dim, K) for _ in range(num_partitions)]
        )

    @property
    def backbone_dim(self) -> int:
        return self._backbone_dim

    def _ckpt(self, layer: nn.Module, x: Tensor) -> Tensor:
        """Run layer with gradient checkpointing (training only)."""
        if self._grad_checkpoint and self.training:
            return grad_ckpt.checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self._ckpt(self.layer1, x)
        x = self._ckpt(self.layer2, x)
        x = self._ckpt(self.layer3, x)
        x = self._ckpt(self.layer4, x)
        x = self.flatten(self.pool(x))

        # L2-normalise each partition independently so every active partition
        # contributes a unit vector to the assembled embedding, regardless of
        # which other partitions are present or absent.
        partitions = [
            F.normalize(head(x), dim=1, eps=1e-12)
            for head in self.partition_heads
        ]
        return {"features": x, "partitions": partitions}
