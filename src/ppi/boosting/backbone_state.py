"""Backbone parameter group configuration for boosting phase transitions."""

from __future__ import annotations

import torch.nn as nn


def set_backbone_state(
    backbone: nn.Module,
    state: str,
    frozen_stages: int = 3,
    backbone_lr_multiplier: float = 0.1,
    base_lr: float = 0.1,
) -> list[dict]:
    """Configure backbone parameter groups for the optimizer.

    Returns a list of parameter group dicts (compatible with torch.optim).
    Frozen parameters are excluded from optimizer groups entirely (not set to
    lr=0, which would still waste memory on gradient buffers).

    For "partial": ResNet-50 is split into freezable groups:
      0: stem (conv1 + bn1)
      1: layer1 (residual stage 1)
      2: layer2 (residual stage 2)
      3: layer3 (residual stage 3)
      4: layer4 (residual stage 4)
      5: final projection / pooling
    Groups 0..frozen_stages-1 are frozen; groups frozen_stages..5 are trainable.
    Default frozen_stages=3 freezes stem + first two residual stages.
    """
    if state not in ("frozen", "fine_tuned", "partial"):
        raise ValueError(
            f"Unknown backbone state '{state}'. "
            "Choose from: frozen, fine_tuned, partial"
        )

    # First unfreeze everything so we start from a clean slate
    for p in backbone.parameters():
        p.requires_grad_(True)

    if state == "frozen":
        for p in backbone.parameters():
            p.requires_grad_(False)
        return []

    backbone_lr = base_lr * backbone_lr_multiplier

    if state == "fine_tuned":
        return [{"params": list(backbone.parameters()), "lr": backbone_lr}]

    # state == "partial"
    stage_groups = _resnet_stage_groups(backbone)
    param_groups: list[dict] = []
    for stage_idx, params in enumerate(stage_groups):
        if stage_idx < frozen_stages:
            for p in params:
                p.requires_grad_(False)
        else:
            trainable = [p for p in params if p.requires_grad]
            if trainable:
                param_groups.append({"params": trainable, "lr": backbone_lr})

    return param_groups


def _resnet_stage_groups(backbone: nn.Module) -> list[list[nn.Parameter]]:
    """Return parameter lists for each ResNet-50 stage group.

    Groups:
      0: stem (conv1, bn1, relu — no maxpool since PartitionedResNet uses 3×3)
      1: layer1
      2: layer2
      3: layer3
      4: layer4
      5: pool, flatten, partition_heads (final projection)
    """
    def _params(module: nn.Module) -> list[nn.Parameter]:
        return list(module.parameters())

    groups: list[list[nn.Parameter]] = []

    # Group 0: stem
    stem_params: list[nn.Parameter] = []
    for attr in ("conv1", "bn1", "relu"):
        mod = getattr(backbone, attr, None)
        if mod is not None:
            stem_params.extend(_params(mod))
    groups.append(stem_params)

    # Groups 1–4: residual stages
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(backbone, layer_name, None)
        groups.append(_params(layer) if layer is not None else [])

    # Group 5: pooling + partition heads
    tail_params: list[nn.Parameter] = []
    for attr in ("pool", "flatten", "partition_heads"):
        mod = getattr(backbone, attr, None)
        if mod is not None:
            tail_params.extend(_params(mod))
    groups.append(tail_params)

    return groups
