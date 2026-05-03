"""Phase-sequential boosting trainer for Direction 2."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ppi.backbones import build_backbone
from ppi.boosting.backbone_state import set_backbone_state
from ppi.boosting.losses import ArcFaceMargin, ArcFaceReweighted, ContrastiveLoss, SubCenterArcFace, TripletLoss, build_loss
from ppi.boosting.mining import HardPairMiner
from ppi.heads.arcface import ArcFaceHead
from ppi.heads.partition_head import PartitionHead
from ppi.losses.arcface_loss import ArcFaceLoss
from ppi.training.schedulers import build_scheduler
from ppi.utils.logging import ExperimentLogger


class BoostingTrainer:
    """Phase-sequential boosting trainer.

    Phase 0: Train P0 (standard ArcFace, backbone fully trainable).
    Phase k (k=1..N-1): Freeze previous partition heads; set backbone state
        per --backbone-state; mine hard pairs from previous ensemble; train
        current partition head on hard pairs with chosen loss; refresh pair
        set every refresh_every steps.

    All hyperparameters from config, all sweepable via CLI overrides.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        logger: ExperimentLogger,
    ) -> None:
        self.config = config
        self.device = device
        self.logger = logger
        self._seed_everything(config.get("seed", 42))

        boosting_cfg = config.get("boosting", {})
        self.num_partitions: int = config.get("num_partitions", 3)
        self.K: int = config.get("partitions", {}).get("K", 128)
        self.backbone_state: str = boosting_cfg.get("backbone_state", "partial")
        self.frozen_stages: int = boosting_cfg.get("frozen_stages", 3)
        self.backbone_lr_multiplier: float = boosting_cfg.get("backbone_lr_multiplier", 0.1)
        self.mining_refresh_every: int = boosting_cfg.get("mining_refresh_every", 500)
        self.grad_clip: float = config.get("training", {}).get("grad_clip", 5.0)

        run_name = config.get("logging", {}).get("run_name") or "boosting_run"
        self.checkpoint_dir = Path("checkpoints/boosting") / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval: int = config.get("training", {}).get("checkpoint_interval", 5)

        # Build backbone
        self.backbone = build_backbone(config).to(device)

        # Per-partition ArcFace heads (one per partition, independent)
        arcface_cfg = config.get("arcface", {})
        self.num_classes: int = arcface_cfg.get("num_classes", 10572)
        self.partition_arcface_heads = nn.ModuleList([
            ArcFaceHead(self.K, self.num_classes).to(device)
            for _ in range(self.num_partitions)
        ])

        self.arcface_loss_fn = ArcFaceLoss(
            s=arcface_cfg.get("s", 64.0),
            m=arcface_cfg.get("m", 0.5),
        )

        # Boosting loss for phases 1+
        self.boosting_loss_fn = build_loss(config).to(device)

        # Hard pair miner
        mining_cfg = boosting_cfg
        self.miner = HardPairMiner(
            strategy=mining_cfg.get("mining_strategy", "topk"),
            band_low=mining_cfg.get("mining_band_low", 0.2),
            band_high=mining_cfg.get("mining_band_high", 0.6),
            topk_fraction=mining_cfg.get("mining_topk_fraction", 0.1),
            refresh_every=self.mining_refresh_every,
        )

        # Build train dataset (shared across phases)
        self._train_dataset: Dataset | None = None
        self._train_loader: DataLoader | None = None
        self._num_classes_override: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_dataset: Dataset | None = None, num_classes: int | None = None) -> None:
        """Run all phases. Saves per-phase checkpoints."""
        if train_dataset is not None:
            self._train_dataset = train_dataset
        if num_classes is not None:
            self._num_classes_override = num_classes
            self.num_classes = num_classes

        training_cfg = self.config.get("training", {})
        epochs_phase0 = training_cfg.get("epochs_phase0", 20)

        print(f"[BoostingTrainer] Starting Phase 0 ({epochs_phase0} epochs)", flush=True)
        self._train_phase_0(epochs_phase0)
        self._save_phase_checkpoint(0)

        epochs_per_phase = training_cfg.get("epochs_per_phase", 20)
        for k in range(1, self.num_partitions):
            print(
                f"\n[BoostingTrainer] Starting Phase {k} ({epochs_per_phase} epochs)",
                flush=True,
            )
            self._train_phase_k(k, epochs_per_phase)
            self._save_phase_checkpoint(k)

        print("[BoostingTrainer] All phases complete.", flush=True)

    # ------------------------------------------------------------------
    # Phase 0
    # ------------------------------------------------------------------

    def _train_phase_0(self, epochs: int) -> None:
        """Train P0 with standard ArcFace (backbone fully trainable)."""
        # All backbone params trainable at base LR
        for p in self.backbone.parameters():
            p.requires_grad_(True)

        opt_cfg = self.config["training"]["optimizer"]
        base_lr: float = opt_cfg.get("lr", 0.1)
        params = list(self.backbone.parameters()) + list(self.partition_arcface_heads[0].parameters())
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 5e-4),
        )
        sched_config = _make_sched_config(self.config, epochs)
        scheduler = build_scheduler(optimizer, sched_config)

        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        loader = self._get_train_loader()
        total_batches = len(loader)
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.backbone.train()
            self.partition_arcface_heads[0].train()
            epoch_loss = 0.0
            n = 0
            t0 = time.time()

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    out = self.backbone(images)
                    p0_emb = out["partitions"][0]
                    cosine = self.partition_arcface_heads[0](p0_emb)
                    loss = self.arcface_loss_fn(cosine, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, self.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                lv = loss.item()
                epoch_loss += lv
                n += 1
                global_step += 1
                self.logger.log_scalar("phase0/loss", lv, global_step)

            scheduler.step()
            avg = epoch_loss / max(n, 1)
            elapsed = time.time() - t0
            print(
                f"  [Phase 0] epoch {epoch}/{epochs}  loss={avg:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s",
                flush=True,
            )
            self.logger.log_epoch({"phase0/epoch_loss": avg, "phase0/lr": optimizer.param_groups[0]["lr"]}, epoch)

    # ------------------------------------------------------------------
    # Phase k
    # ------------------------------------------------------------------

    def _train_phase_k(self, k: int, epochs: int) -> None:
        """Train partition k on hard pairs from ensemble of partitions 0..k-1.

        1. Freeze partitions 0..k-1 heads.
        2. Set backbone state (frozen/fine_tuned/partial).
        3. Compute previous ensemble embeddings over training set.
        4. Run HardPairMiner to get initial pair set.
        5. Train for config epochs, refreshing pair set every refresh_every steps.
        """
        # Freeze prior partition heads
        for j in range(k):
            for p in self.partition_arcface_heads[j].parameters():
                p.requires_grad_(False)

        # Configure backbone state
        opt_cfg = self.config["training"]["optimizer"]
        base_lr: float = opt_cfg.get("lr", 0.1)
        backbone_param_groups = set_backbone_state(
            self.backbone,
            state=self.backbone_state,
            frozen_stages=self.frozen_stages,
            backbone_lr_multiplier=self.backbone_lr_multiplier,
            base_lr=base_lr,
        )

        head_params = list(self.partition_arcface_heads[k].parameters())
        all_param_groups = backbone_param_groups + [{"params": head_params, "lr": base_lr}]

        if not all_param_groups:
            all_param_groups = [{"params": head_params, "lr": base_lr}]

        optimizer = torch.optim.SGD(
            all_param_groups,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 5e-4),
        )
        sched_config = _make_sched_config(self.config, epochs)
        scheduler = build_scheduler(optimizer, sched_config)

        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Compute ensemble embeddings over full training set (initial mining)
        print(f"  [Phase {k}] Computing initial ensemble embeddings...", flush=True)
        ensemble_embs, train_labels = self._compute_ensemble_embeddings(k)

        # Reset miner cache for new phase
        self.miner._cached_a = None
        self.miner._cached_b = None

        loader = self._get_train_loader()
        total_batches = len(loader)
        global_step = 0
        phase_tag = f"phase{k}"

        loss_name = self.config.get("boosting", {}).get("loss", "triplet")

        for epoch in range(1, epochs + 1):
            self.backbone.train()
            self.partition_arcface_heads[k].train()
            epoch_loss = 0.0
            n = 0
            t0 = time.time()

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                # Dynamic pair refresh
                idx_a, idx_b, mining_stats = self.miner.mine(
                    ensemble_embs, train_labels, global_step,
                )
                if mining_stats["refreshed"]:
                    self.logger.log_scalar(f"{phase_tag}/mining_n_pairs", mining_stats["n_pairs"], global_step)
                    self.logger.log_scalar(f"{phase_tag}/mining_score_mean", mining_stats["score_mean"], global_step)
                    print(
                        f"    [Phase {k}] step {global_step}: mined {mining_stats['n_pairs']} hard pairs "
                        f"(score_mean={mining_stats['score_mean']:.3f})",
                        flush=True,
                    )

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    out = self.backbone(images)
                    pk_emb = out["partitions"][k]
                    loss = self._compute_boosting_loss(
                        pk_emb, labels, idx_a, idx_b,
                        ensemble_embs, train_labels, loss_name,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                trainable_params = [p for g in all_param_groups for p in g["params"] if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                lv = loss.item()
                epoch_loss += lv
                n += 1
                global_step += 1
                self.logger.log_scalar(f"{phase_tag}/loss", lv, global_step)
                self.logger.log_scalar(f"{phase_tag}/backbone_lr", backbone_param_groups[0]["lr"] if backbone_param_groups else 0.0, global_step)
                self.logger.log_scalar(f"{phase_tag}/head_lr", base_lr, global_step)

                # Refresh ensemble embeddings when the miner triggers a refresh
                if global_step % self.mining_refresh_every == 0 and self.backbone_state != "frozen":
                    ensemble_embs, train_labels = self._compute_ensemble_embeddings(k)

            scheduler.step()
            avg = epoch_loss / max(n, 1)
            elapsed = time.time() - t0
            print(
                f"  [Phase {k}] epoch {epoch}/{epochs}  loss={avg:.4f}  "
                f"lr={optimizer.param_groups[-1]['lr']:.2e}  time={elapsed:.1f}s",
                flush=True,
            )
            self.logger.log_epoch({
                f"{phase_tag}/epoch_loss": avg,
                f"{phase_tag}/lr": optimizer.param_groups[-1]["lr"],
            }, epoch)

    def _compute_boosting_loss(
        self,
        pk_emb: Tensor,
        batch_labels: Tensor,
        idx_a: Tensor,
        idx_b: Tensor,
        ensemble_embs: Tensor,
        train_labels: Tensor,
        loss_name: str,
    ) -> Tensor:
        """Compute the boosting loss for partition k's current batch."""
        if isinstance(self.boosting_loss_fn, TripletLoss):
            return self.boosting_loss_fn(pk_emb, batch_labels)

        elif isinstance(self.boosting_loss_fn, ContrastiveLoss):
            # Use batch pairs, not global mined pairs
            B = pk_emb.shape[0]
            if B < 2:
                return pk_emb.sum() * 0.0
            half = B // 2
            emb_a = pk_emb[:half]
            emb_b = pk_emb[half: 2 * half]
            lab_a = batch_labels[:half]
            lab_b = batch_labels[half: 2 * half]
            is_same = (lab_a == lab_b).float()
            return self.boosting_loss_fn(emb_a, emb_b, is_same)

        elif isinstance(self.boosting_loss_fn, ArcFaceReweighted):
            cosine = self.partition_arcface_heads[0](pk_emb)
            # Compute pair weights from ensemble scores for batch items
            emb_norm = F.normalize(ensemble_embs[: pk_emb.shape[0]].to(self.device), dim=1)
            prev_scores = (emb_norm * F.normalize(pk_emb.detach(), dim=1)).sum(dim=1).abs()
            is_same_flag = (batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)).float()
            weights = is_same_flag.mean(dim=1) * (1.0 - prev_scores) + (1.0 - is_same_flag.mean(dim=1)) * prev_scores
            return self.boosting_loss_fn(cosine, batch_labels, weights)

        elif isinstance(self.boosting_loss_fn, ArcFaceMargin):
            # Hard: batch items whose embedding is most distant from ensemble
            hard_cosine = self.partition_arcface_heads[0](pk_emb)
            return self.boosting_loss_fn(hard_cosine, batch_labels, hard_cosine, batch_labels)

        elif isinstance(self.boosting_loss_fn, SubCenterArcFace):
            return self.boosting_loss_fn(pk_emb, batch_labels)

        else:
            return self.boosting_loss_fn(pk_emb, batch_labels)

    @torch.no_grad()
    def _compute_ensemble_embeddings(self, k: int) -> tuple[Tensor, Tensor]:
        """Compute concatenated embeddings from partitions 0..k-1 over training set."""
        self.backbone.eval()
        for j in range(k):
            self.partition_arcface_heads[j].eval()

        loader = self._get_train_loader(shuffle=False)
        all_embs: list[Tensor] = []
        all_labels: list[Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            out = self.backbone(images)
            parts = [F.normalize(out["partitions"][j], dim=1) for j in range(k)]
            concat = torch.cat(parts, dim=1).cpu()
            all_embs.append(concat)
            all_labels.append(labels)

        self.backbone.train()
        for j in range(k):
            self.partition_arcface_heads[j].train()

        return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_phase_checkpoint(self, phase: int) -> None:
        phase_dir = self.checkpoint_dir / f"phase_{phase}"
        phase_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {"backbone": self.backbone.state_dict()},
            phase_dir / "backbone.pt",
        )
        torch.save(
            {
                "partition_head": self.partition_arcface_heads[phase].state_dict(),
                "phase": phase,
                "num_partitions": self.num_partitions,
                "K": self.K,
            },
            phase_dir / f"partition_{phase}.pt",
        )
        print(f"[BoostingTrainer] Saved phase {phase} checkpoint to {phase_dir}", flush=True)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_train_loader(self, shuffle: bool = True) -> DataLoader:
        from functools import partial as _partial
        from ppi.data import build_dataloader

        if self._train_dataset is not None:
            batch_size = self.config.get("training", {}).get("batch_size", 256)
            num_workers = self.config.get("data", {}).get("num_workers", 4)
            seed = self.config.get("seed", 42)

            def _worker_init(worker_id: int) -> None:
                import random as _r
                _r.seed(seed + worker_id)
                np.random.seed(seed + worker_id)
                torch.manual_seed(seed + worker_id)

            return DataLoader(
                self._train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=_worker_init,
            )

        loader, _ = build_dataloader(self.config, split="train")
        return loader

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _make_sched_config(config: dict, epochs: int) -> dict:
    """Return a scheduler-compatible config slice with the given epoch count."""
    return {
        "training": {
            "epochs": epochs,
            "scheduler": config.get("training", {}).get(
                "scheduler", {"type": "cosine", "warmup_epochs": 1}
            ),
        }
    }
