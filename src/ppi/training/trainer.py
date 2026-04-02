"""Main training loop for Progressive Partitioned Inference."""

from __future__ import annotations

import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ppi.backbones import build_backbone
from ppi.data import build_dataloader
from ppi.heads.arcface import ArcFaceHead
from ppi.losses.arcface_loss import ArcFaceLoss
from ppi.partitions.base import PartitionStrategy
from ppi.training.partition_dropout import PartitionDropout, assemble_embedding
from ppi.training.schedulers import build_scheduler
from ppi.utils.logging import ExperimentLogger


class Trainer:
    """Orchestrates PPI training with variant-agnostic hooks."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._seed_everything(config["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_loader, num_classes = build_dataloader(config, split="train")
        # Allow config override for num_classes
        num_classes = config.get("arcface", {}).get("num_classes", num_classes)

        # Model
        self.backbone = build_backbone(config).to(self.device)
        K = config["partitions"]["K"]
        num_partitions = config["partitions"]["num_partitions"]
        embedding_dim = num_partitions * K
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes).to(self.device)

        # Partition mechanics
        dropout_cfg = config["partitions"].get("dropout", {})
        self.partition_dropout = PartitionDropout(
            num_partitions=num_partitions,
            distribution=dropout_cfg.get("distribution", [0.4, 0.3, 0.2, 0.1]),
        ).to(self.device)

        # Variant strategy
        self.strategy = PartitionStrategy.from_config(config)
        if isinstance(self.strategy, nn.Module):
            self.strategy = self.strategy.to(self.device)

        # Loss
        arcface_cfg = config.get("arcface", {})
        self.arcface_loss = ArcFaceLoss(
            s=arcface_cfg.get("s", 64.0),
            m=arcface_cfg.get("m", 0.5),
        )

        # Optimizer
        params = self.strategy.get_trainable_parameters(self.backbone)
        # Include arcface head and any strategy parameters
        params = params + list(self.arcface_head.parameters())
        if isinstance(self.strategy, nn.Module):
            params = params + list(self.strategy.parameters())

        opt_cfg = config["training"]["optimizer"]
        self.optimizer = torch.optim.SGD(
            params,
            lr=opt_cfg.get("lr", 0.1),
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 5e-4),
        )

        # Scheduler
        self.scheduler = build_scheduler(self.optimizer, config)

        # Logger
        self.logger = ExperimentLogger(config)

        self._total_batches = len(self.train_loader)
        print(
            f"[Trainer] device={self.device}, backbone={config['backbone']['name']}, "
            f"partitions={num_partitions}×{K}, classes={num_classes}, "
            f"batches/epoch={self._total_batches}"
        )

    def train(self) -> None:
        epochs = self.config["training"]["epochs"]
        checkpoint_interval = self.config["training"].get("checkpoint_interval", 1)

        self.strategy.pre_training_setup(self.backbone, self.config)
        print(f"[Trainer] Starting training for {epochs} epochs")

        global_step = 0
        for epoch in range(1, epochs + 1):
            self.backbone.train()
            self.arcface_head.train()
            self.partition_dropout.train()

            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()
            log_interval = max(1, self._total_batches // 5)  # ~5 updates per epoch

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                out = self.backbone(images)
                partition_outputs = out["partitions"]

                # Strategy processing (e.g. positional encoding)
                partition_outputs = self.strategy.process_partitions(partition_outputs)

                # Partition dropout
                partition_outputs = self.partition_dropout(partition_outputs)

                # Assemble and classify
                embedding = assemble_embedding(partition_outputs)
                cosine = self.arcface_head(embedding)

                # Losses
                loss = self.arcface_loss(cosine, labels)
                aux_loss = self.strategy.compute_auxiliary_loss(out["partitions"])
                total_loss = loss + aux_loss

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1
                global_step += 1

                self.logger.log_scalar("train/loss", total_loss.item(), global_step)
                if aux_loss.item() > 0:
                    self.logger.log_scalar("train/aux_loss", aux_loss.item(), global_step)

                if num_batches % log_interval == 0:
                    avg = epoch_loss / num_batches
                    print(
                        f"  epoch {epoch}/{epochs}  "
                        f"batch {num_batches}/{self._total_batches}  "
                        f"loss={avg:.4f}",
                        flush=True,
                    )

            self.scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch}/{epochs} done  "
                f"loss={avg_loss:.4f}  lr={lr:.6f}  "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            self.logger.log_scalar("train/epoch_loss", avg_loss, epoch)

            self.strategy.post_epoch_hook(epoch, self.backbone)

            if epoch % checkpoint_interval == 0:
                self.logger.save_checkpoint(
                    model_state={
                        "backbone": self.backbone.state_dict(),
                        "arcface_head": self.arcface_head.state_dict(),
                    },
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    metrics={"loss": avg_loss},
                )

        self.logger.close()
        print(f"[Trainer] Training complete. Logs at {self.logger.run_dir}")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
