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

        self._all_params = params
        self._grad_clip = config["training"].get("grad_clip", 5.0)

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

            epoch_totals = {"arcface": 0.0, "aux": 0.0, "total": 0.0}
            num_batches = 0
            epoch_start = time.time()
            # ~10 updates per epoch, but cap at every 500 batches for large datasets
            log_interval = max(1, min(self._total_batches // 10, 500))

            for images, labels in self.train_loader:
                if num_batches == 0 and epoch == 1:
                    print(
                        f"  First batch loaded, training started "
                        f"(logging every {log_interval} batches)...",
                        flush=True,
                    )
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                out = self.backbone(images)
                partition_outputs = out["partitions"]

                # Strategy processing (e.g. positional encoding)
                partition_outputs = self.strategy.process_partitions(partition_outputs)

                # Partition dropout
                dropped_outputs = self.partition_dropout(partition_outputs)

                # Assemble and classify
                embedding = assemble_embedding(dropped_outputs)
                cosine = self.arcface_head(embedding)

                # Losses
                arcface_loss = self.arcface_loss(cosine, labels)
                aux_loss = self.strategy.compute_auxiliary_loss(out["partitions"])
                total_loss = arcface_loss + aux_loss

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping — essential for ArcFace stability with
                # many classes (10k+) and high learning rates
                torch.nn.utils.clip_grad_norm_(
                    self._all_params, max_norm=self._grad_clip,
                )
                self.optimizer.step()

                # Track per-component losses
                arcface_val = arcface_loss.item()
                aux_val = aux_loss.item()
                total_val = total_loss.item()
                epoch_totals["arcface"] += arcface_val
                epoch_totals["aux"] += aux_val
                epoch_totals["total"] += total_val
                num_batches += 1
                global_step += 1

                # Per-step TensorBoard logging
                self.logger.log_scalar("train/loss_total", total_val, global_step)
                self.logger.log_scalar("train/loss_arcface", arcface_val, global_step)
                if aux_val > 0:
                    self.logger.log_scalar("train/loss_aux", aux_val, global_step)

                # Log diagnostics less frequently to avoid overhead
                if num_batches % log_interval == 0:
                    # Partition activation: how many partitions were active this batch
                    n_active = sum(1 for d in dropped_outputs if d.abs().sum() > 0)
                    # Embedding L2 norm (pre-normalisation would be more informative,
                    # but post-normalisation should be ~1.0 for non-zero embeddings)
                    emb_norm = embedding.detach().norm(dim=1).mean().item()

                    self.logger.log_scalar("train/active_partitions", n_active, global_step)
                    self.logger.log_scalar("train/embedding_norm", emb_norm, global_step)

                    avg_t = epoch_totals["total"] / num_batches
                    avg_a = epoch_totals["arcface"] / num_batches
                    avg_x = epoch_totals["aux"] / num_batches
                    parts_str = f"  aux={avg_x:.4f}" if avg_x > 0 else ""
                    print(
                        f"  epoch {epoch}/{epochs}  "
                        f"batch {num_batches}/{self._total_batches}  "
                        f"total={avg_t:.4f}  arcface={avg_a:.4f}{parts_str}  "
                        f"active={n_active}/3",
                        flush=True,
                    )

            self.scheduler.step()
            avg_total = epoch_totals["total"] / max(num_batches, 1)
            avg_arcface = epoch_totals["arcface"] / max(num_batches, 1)
            avg_aux = epoch_totals["aux"] / max(num_batches, 1)
            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            aux_str = f"  aux={avg_aux:.4f}" if avg_aux > 0 else ""
            print(
                f"  epoch {epoch}/{epochs} done  "
                f"total={avg_total:.4f}  arcface={avg_arcface:.4f}{aux_str}  "
                f"lr={lr:.6f}  time={elapsed:.1f}s",
                flush=True,
            )
            epoch_metrics = {
                "train/epoch_loss_total": avg_total,
                "train/epoch_loss_arcface": avg_arcface,
                "train/epoch_time_s": elapsed,
                "train/lr": lr,
            }
            if avg_aux > 0:
                epoch_metrics["train/epoch_loss_aux"] = avg_aux
            self.logger.log_epoch(epoch_metrics, epoch)

            self.strategy.post_epoch_hook(epoch, self.backbone)

            if epoch % checkpoint_interval == 0:
                self.logger.save_checkpoint(
                    model_state={
                        "backbone": self.backbone.state_dict(),
                        "arcface_head": self.arcface_head.state_dict(),
                    },
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    metrics={
                        "loss_total": avg_total,
                        "loss_arcface": avg_arcface,
                        "loss_aux": avg_aux,
                    },
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
