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

            # Loss tracking buckets.  The default path uses "arcface" and
            # "aux"; the custom-step path logs whatever keys the strategy
            # returns (e.g. arcface_full, arcface_narrow, kd, width).
            epoch_totals: dict[str, float] = {"total": 0.0}
            # Whether the strategy owns the forward pass this epoch.
            custom_step = False
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

                # Strategy may override the entire forward + loss computation
                step_result = self.strategy.training_step(
                    out, labels, self.arcface_head, self.arcface_loss,
                    self.partition_dropout,
                )

                if step_result is not None:
                    # --- Custom step path (Variant B, etc.) ---
                    custom_step = True
                    total_loss, step_metrics = step_result
                    total_val = total_loss.item()
                    dropped_outputs = None
                    embedding = None
                    # Accumulate every metric the strategy reports
                    for k, v in step_metrics.items():
                        epoch_totals.setdefault(k, 0.0)
                        epoch_totals[k] += v
                else:
                    # --- Default path (Variant A, baseline, etc.) ---
                    partition_outputs = out["partitions"]

                    # Strategy processing (e.g. positional encoding,
                    # or prefix masking for Variant B)
                    partition_outputs = self.strategy.process_partitions(
                        partition_outputs,
                    )

                    # Partition dropout (skipped if strategy handles its own)
                    if self.strategy.handles_own_dropout:
                        dropped_outputs = partition_outputs
                    else:
                        dropped_outputs = self.partition_dropout(partition_outputs)

                    # Assemble, optional strategy transform, and classify
                    embedding = assemble_embedding(dropped_outputs)
                    embedding = self.strategy.post_assembly(embedding)
                    cosine = self.arcface_head(embedding)

                    # Losses
                    arcface_loss = self.arcface_loss(cosine, labels)
                    aux_loss = self.strategy.compute_auxiliary_loss(
                        out["partitions"],
                    )
                    total_loss = arcface_loss + aux_loss
                    total_val = total_loss.item()
                    step_metrics = {
                        "arcface": arcface_loss.item(),
                        "aux": aux_loss.item(),
                    }
                    for k, v in step_metrics.items():
                        epoch_totals.setdefault(k, 0.0)
                        epoch_totals[k] += v

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping — essential for ArcFace stability with
                # many classes (10k+) and high learning rates
                torch.nn.utils.clip_grad_norm_(
                    self._all_params, max_norm=self._grad_clip,
                )
                self.optimizer.step()

                # Track total loss
                epoch_totals["total"] += total_val
                num_batches += 1
                global_step += 1

                # Per-step TensorBoard logging — write every metric
                self.logger.log_scalar("train/loss_total", total_val, global_step)
                for k, v in step_metrics.items():
                    if k == "width":
                        continue  # not a loss
                    self.logger.log_scalar(f"train/loss_{k}", v, global_step)

                # Log diagnostics less frequently to avoid overhead
                if num_batches % log_interval == 0:
                    if not custom_step:
                        n_active = self.partition_dropout.last_chosen_width
                    else:
                        n_active = int(step_metrics.get("width", 3))
                    if embedding is not None:
                        emb_norm = embedding.detach().norm(dim=1).mean().item()
                    else:
                        emb_norm = 1.0  # post-L2-norm is ~1.0 by construction

                    self.logger.log_scalar("train/active_partitions", n_active, global_step)
                    self.logger.log_scalar("train/embedding_norm", emb_norm, global_step)

                    avg_t = epoch_totals["total"] / num_batches
                    # Build per-component averages string
                    parts = []
                    for k in sorted(epoch_totals):
                        if k == "total" or k == "width":
                            continue
                        avg_k = epoch_totals[k] / num_batches
                        if avg_k > 0 or k not in ("aux",):
                            parts.append(f"{k}={avg_k:.4f}")
                    parts_str = "  ".join(parts)
                    print(
                        f"  epoch {epoch}/{epochs}  "
                        f"batch {num_batches}/{self._total_batches}  "
                        f"total={avg_t:.4f}  {parts_str}  "
                        f"active={n_active}/3",
                        flush=True,
                    )

            self.scheduler.step()
            n = max(num_batches, 1)
            avg_total = epoch_totals["total"] / n
            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            # Build component summary for end-of-epoch line
            parts = []
            for k in sorted(epoch_totals):
                if k == "total" or k == "width":
                    continue
                avg_k = epoch_totals[k] / n
                if avg_k > 0 or k not in ("aux",):
                    parts.append(f"{k}={avg_k:.4f}")
            parts_str = "  ".join(parts)
            print(
                f"  epoch {epoch}/{epochs} done  "
                f"total={avg_total:.4f}  {parts_str}  "
                f"lr={lr:.6f}  time={elapsed:.1f}s",
                flush=True,
            )
            epoch_metrics = {
                "train/epoch_loss_total": avg_total,
                "train/epoch_time_s": elapsed,
                "train/lr": lr,
            }
            for k in epoch_totals:
                if k == "total" or k == "width":
                    continue
                epoch_metrics[f"train/epoch_loss_{k}"] = epoch_totals[k] / n
            self.logger.log_epoch(epoch_metrics, epoch)

            self.strategy.post_epoch_hook(epoch, self.backbone)

            if epoch % checkpoint_interval == 0:
                model_state = {
                    "backbone": self.backbone.state_dict(),
                    "arcface_head": self.arcface_head.state_dict(),
                }
                if isinstance(self.strategy, nn.Module):
                    model_state["strategy"] = self.strategy.state_dict()
                ckpt_metrics = {"loss_total": avg_total}
                for k in epoch_totals:
                    if k == "total" or k == "width":
                        continue
                    ckpt_metrics[f"loss_{k}"] = epoch_totals[k] / n
                self.logger.save_checkpoint(
                    model_state=model_state,
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    metrics=ckpt_metrics,
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
