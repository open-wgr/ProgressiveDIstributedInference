"""Backbone network architectures with partitioned output heads."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ppi.backbones.resnet import PartitionedResNet


def build_backbone(config: dict[str, Any]) -> nn.Module:
    """Build a backbone from a config dict."""
    name = config["backbone"]["name"]
    partitions = config["partitions"]
    input_size = config.get("data", {}).get("input_size", 112)

    if name in ("resnet18", "resnet50"):
        return PartitionedResNet(
            backbone_name=name,
            num_partitions=partitions["num_partitions"],
            K=partitions["K"],
            pretrained=config["backbone"].get("pretrained", False),
            input_size=input_size,
            gradient_checkpointing=config["backbone"].get("gradient_checkpointing", False),
        )
    elif name == "mobilefacenet":
        from ppi.backbones.mobilefacenet import MobileFaceNet
        return MobileFaceNet()
    else:
        raise ValueError(f"Unknown backbone: '{name}'")
