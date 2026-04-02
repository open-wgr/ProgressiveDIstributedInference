"""Learning rate schedulers: warmup + cosine decay."""

from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler


def build_scheduler(optimizer: Optimizer, config: dict[str, Any]) -> LRScheduler:
    """Build a warmup + cosine annealing schedule from config."""
    sched_cfg = config["training"]["scheduler"]
    warmup_epochs = sched_cfg.get("warmup_epochs", 1)
    total_epochs = config["training"]["epochs"]

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
