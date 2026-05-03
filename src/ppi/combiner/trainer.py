"""Combiner training loop: ArcFace loss over frozen partition embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ppi.combiner.mlp import PartitionCombiner
from ppi.heads.arcface import ArcFaceHead
from ppi.losses.arcface_loss import ArcFaceLoss
from ppi.utils.logging import ExperimentLogger


def train_combiner(
    combiner: PartitionCombiner,
    arcface_head: ArcFaceHead,
    arcface_loss: ArcFaceLoss,
    train_dataset: Dataset,
    config: dict[str, Any],
    logger: ExperimentLogger,
    device: torch.device,
) -> PartitionCombiner:
    """Train the combiner with ArcFace loss on combined embeddings.

    The combiner and a freshly-initialised arcface_head are optimised jointly
    with AdamW.  The frozen backbone is not touched — only combiner weights and
    the new ArcFace head are updated.

    Returns the combiner at its best epoch (lowest training loss).  If training
    loss does not decrease from epoch 1 to epoch max_epochs, a warning is logged
    and the best checkpoint is returned without raising.
    """
    train_cfg = config.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})
    max_epochs: int = train_cfg.get("epochs", 5)
    batch_size: int = train_cfg.get("batch_size", 512)
    lr: float = opt_cfg.get("lr", 1e-3)
    weight_decay: float = opt_cfg.get("weight_decay", 1e-4)

    combiner = combiner.to(device)
    arcface_head = arcface_head.to(device)
    arcface_loss = arcface_loss.to(device)

    optimizer = torch.optim.AdamW(
        list(combiner.parameters()) + list(arcface_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # num_workers=0 keeps the seeded generator in CachedPartitionDataset reproducible
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    best_loss = float("inf")
    best_combiner_state: dict = {}
    epoch1_loss: float | None = None
    final_loss: float = float("inf")
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        combiner.train()
        arcface_head.train()
        running_loss = 0.0
        n_steps = 0

        for embeddings, mask, labels in loader:
            embeddings = embeddings.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            combined = combiner(embeddings, mask)
            cosine = arcface_head(combined)
            loss = arcface_loss(cosine, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(combiner.parameters()) + list(arcface_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            step_loss = loss.item()
            running_loss += step_loss
            n_steps += 1
            logger.log_scalar("combiner/loss_step", step_loss, global_step)
            logger.log_scalar("combiner/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        mean_loss = running_loss / max(n_steps, 1)
        final_loss = mean_loss
        logger.log_epoch({"combiner/loss_epoch": mean_loss}, epoch)
        print(f"[CombinerTrainer] Epoch {epoch}/{max_epochs}  loss={mean_loss:.4f}", flush=True)

        # Persist epoch checkpoint
        ckpt_path = Path(logger.run_dir) / f"combiner_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "combiner_state_dict": combiner.state_dict(),
                "arcface_head_state_dict": arcface_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": mean_loss,
            },
            ckpt_path,
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_combiner_state = {k: v.clone() for k, v in combiner.state_dict().items()}

        if epoch == 1:
            epoch1_loss = mean_loss

    if epoch1_loss is not None and final_loss >= epoch1_loss:
        print(
            f"[CombinerTrainer] WARNING: loss did not decrease from epoch 1 "
            f"({epoch1_loss:.4f}) to epoch {max_epochs} ({final_loss:.4f}). "
            "Returning best checkpoint.",
            flush=True,
        )

    combiner.load_state_dict(best_combiner_state)
    return combiner
