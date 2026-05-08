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
from ppi.boosting.mining import HardPairDataset, HardPairMiner
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
        for p in self.partition_arcface_heads[0].parameters():
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

        Mined pairs drive a HardPairDataset; the inner loop draws batches of
        (img_a, img_b, lab_a, lab_b, is_same, pair_idx) from the mined set, so
        the boosting hypothesis (train P_k on pairs the previous ensemble
        fails) is realised directly.
        """
        # Freeze prior partition heads
        for j in range(k):
            for p in self.partition_arcface_heads[j].parameters():
                p.requires_grad_(False)

        # Configure backbone state. The current partition head must remain
        # trainable even when the rest of the backbone is frozen — pk_emb is
        # the only thing carrying gradient, and partition_heads[k] is what
        # produces it.
        opt_cfg = self.config["training"]["optimizer"]
        base_lr: float = opt_cfg.get("lr", 0.1)
        backbone_param_groups = set_backbone_state(
            self.backbone,
            state=self.backbone_state,
            frozen_stages=self.frozen_stages,
            backbone_lr_multiplier=self.backbone_lr_multiplier,
            base_lr=base_lr,
        )

        # Force-unfreeze the current partition projection head, and freeze the
        # prior ones explicitly (set_backbone_state may have re-enabled them
        # when state != "frozen").
        if hasattr(self.backbone, "partition_heads"):
            for j, ph in enumerate(self.backbone.partition_heads):
                want_trainable = (j == k)
                for p in ph.parameters():
                    p.requires_grad_(want_trainable)
            current_proj_params = [
                p for p in self.backbone.partition_heads[k].parameters()
                if p.requires_grad
            ]
            # Avoid double-counting if partial mode already included it
            existing_ids = {id(p) for g in backbone_param_groups for p in g["params"]}
            new_proj_params = [p for p in current_proj_params if id(p) not in existing_ids]
            if new_proj_params:
                backbone_param_groups.append({"params": new_proj_params, "lr": base_lr * self.backbone_lr_multiplier if self.backbone_state != "frozen" else base_lr})
            # Drop prior heads' parameters from any partial-mode group that
            # picked them up via attribute-walk.
            backbone_param_groups = self._strip_frozen_params(backbone_param_groups)

        head_params = list(self.partition_arcface_heads[k].parameters())
        all_param_groups: list[dict] = list(backbone_param_groups)
        all_param_groups.append({"params": head_params, "lr": base_lr})

        # Boosting-loss-owned parameters (e.g. SubCenterArcFace centroids,
        # ArcFaceReweighted has none, etc.). Without this group those
        # parameters never receive gradient updates.
        loss_params = [p for p in self.boosting_loss_fn.parameters() if p.requires_grad]
        if loss_params:
            all_param_groups.append({"params": loss_params, "lr": base_lr})

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

        global_step = 0
        phase_tag = f"phase{k}"
        loss_name = self.config.get("boosting", {}).get("loss", "triplet")

        # Initial mining
        idx_a, idx_b, mining_stats = self.miner.mine(ensemble_embs, train_labels, global_step)
        self._log_mining(phase_tag, mining_stats, global_step, k)
        pair_loader = self._build_pair_loader(idx_a, idx_b)

        for epoch in range(1, epochs + 1):
            self.backbone.train()
            self.partition_arcface_heads[k].train()
            self._set_frozen_modules_eval(k)
            epoch_loss = 0.0
            n = 0
            t0 = time.time()

            for batch in pair_loader:
                img_a, img_b, lab_a, lab_b, is_same, pidx_a, pidx_b = batch
                img_a = img_a.to(self.device, non_blocking=True)
                img_b = img_b.to(self.device, non_blocking=True)
                lab_a = lab_a.to(self.device)
                lab_b = lab_b.to(self.device)
                is_same = is_same.to(self.device)
                pidx_a = pidx_a.to(self.device)
                pidx_b = pidx_b.to(self.device)

                optimizer.zero_grad()

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    images = torch.cat([img_a, img_b], dim=0)
                    labels = torch.cat([lab_a, lab_b], dim=0)
                    out = self.backbone(images)
                    pk_emb = out["partitions"][k]
                    loss = self._compute_boosting_loss(
                        k, pk_emb, labels, is_same,
                        pidx_a, pidx_b,
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
                self.logger.log_scalar(
                    f"{phase_tag}/backbone_lr",
                    backbone_param_groups[0]["lr"] if backbone_param_groups else 0.0,
                    global_step,
                )
                self.logger.log_scalar(f"{phase_tag}/head_lr", base_lr, global_step)

            # End of epoch: refresh mined pairs & (optionally) ensemble
            # embeddings. Doing this at the epoch boundary keeps each epoch
            # covering a coherent pair set instead of being truncated mid-iter.
            if global_step >= self.miner._last_refresh_step + self.mining_refresh_every:
                if self.backbone_state != "frozen":
                    ensemble_embs, train_labels = self._compute_ensemble_embeddings(k)
                # Force refresh by clearing the miner's cache.
                self.miner._cached_a = None
                self.miner._cached_b = None
                idx_a, idx_b, mining_stats = self.miner.mine(
                    ensemble_embs, train_labels, global_step,
                )
                self._log_mining(phase_tag, mining_stats, global_step, k)
                pair_loader = self._build_pair_loader(idx_a, idx_b)

            scheduler.step()
            avg = epoch_loss / max(n, 1)
            elapsed = time.time() - t0
            backbone_lr_now = (
                backbone_param_groups[0]["lr"] if backbone_param_groups else 0.0
            )
            print(
                f"  [Phase {k}] epoch {epoch}/{epochs}  loss={avg:.4f}  "
                f"head_lr={optimizer.param_groups[-1]['lr']:.2e}  "
                f"backbone_lr={backbone_lr_now:.2e}  time={elapsed:.1f}s",
                flush=True,
            )
            self.logger.log_epoch({
                f"{phase_tag}/epoch_loss": avg,
                f"{phase_tag}/head_lr": optimizer.param_groups[-1]["lr"],
                f"{phase_tag}/backbone_lr": backbone_lr_now,
            }, epoch)

    # ------------------------------------------------------------------
    # Helpers for phase k
    # ------------------------------------------------------------------

    def _build_pair_loader(self, idx_a: Tensor, idx_b: Tensor) -> DataLoader:
        """Wrap the mined pair indices into a DataLoader over the source dataset."""
        source = self._get_source_dataset()
        pair_ds = HardPairDataset(idx_a, idx_b, source)
        batch_size = self.config.get("training", {}).get("batch_size", 256)
        # Each pair contributes 2 images, so halve the per-pair batch size
        # to keep the actual forward-pass batch size near `batch_size`.
        per_pair_bs = max(1, batch_size // 2)
        num_workers = self.config.get("data", {}).get("num_workers", 4)
        return DataLoader(
            pair_ds,
            batch_size=per_pair_bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _log_mining(self, phase_tag: str, stats: dict, global_step: int, k: int) -> None:
        if not stats.get("refreshed"):
            return
        self.logger.log_scalar(f"{phase_tag}/mining_n_pairs", stats["n_pairs"], global_step)
        self.logger.log_scalar(f"{phase_tag}/mining_score_mean", stats["score_mean"], global_step)
        print(
            f"    [Phase {k}] step {global_step}: mined {stats['n_pairs']} hard pairs "
            f"(score_mean={stats['score_mean']:.3f})",
            flush=True,
        )

    def _set_frozen_modules_eval(self, k: int) -> None:
        """Put frozen submodules in eval mode so BN running stats don't drift.

        `requires_grad_(False)` does not stop BatchNorm from updating its
        running mean/var on every forward pass — calling .eval() does.
        """
        if self.backbone_state == "frozen":
            self.backbone.eval()
            # but the current projection head must stay in train mode
            if hasattr(self.backbone, "partition_heads"):
                self.backbone.partition_heads[k].train()
            return

        if self.backbone_state == "partial":
            stage_attrs = ["conv1", "bn1", "relu", "layer1", "layer2", "layer3", "layer4", "pool", "flatten"]
            # Map frozen_stages to attribute groups (mirrors backbone_state._resnet_stage_groups)
            # 0: stem (conv1,bn1,relu); 1: layer1; 2: layer2; 3: layer3; 4: layer4; 5: pool/flatten/heads
            mapping = [
                ("conv1", "bn1", "relu"),
                ("layer1",),
                ("layer2",),
                ("layer3",),
                ("layer4",),
                ("pool", "flatten"),
            ]
            for stage_idx in range(min(self.frozen_stages, len(mapping))):
                for attr in mapping[stage_idx]:
                    mod = getattr(self.backbone, attr, None)
                    if isinstance(mod, nn.Module):
                        mod.eval()

        # Always: prior partition projection heads are frozen → eval
        if hasattr(self.backbone, "partition_heads"):
            for j, ph in enumerate(self.backbone.partition_heads):
                if j != k:
                    ph.eval()

    def rebuild_head_for_num_classes(self, k: int, num_classes: int) -> None:
        """Rebuild partition_arcface_heads[k] (and SubCenterArcFace centroids
        if applicable) when the label space changes between phases.
        """
        if num_classes == self.partition_arcface_heads[k].weight.shape[0]:
            # already correct
            self.num_classes = num_classes
            return
        self.partition_arcface_heads[k] = ArcFaceHead(self.K, num_classes).to(self.device)
        # Rebuild loss-owned classifier params (only matters for losses that
        # carry their own num_classes-shaped weight, e.g. SubCenterArcFace).
        if isinstance(self.boosting_loss_fn, SubCenterArcFace):
            arcface_cfg = self.config.get("arcface", {})
            self.boosting_loss_fn = SubCenterArcFace(
                num_classes=num_classes,
                embedding_dim=self.K,
                s=arcface_cfg.get("s", 64.0),
                m=arcface_cfg.get("m", 0.5),
                K=self.config.get("boosting", {}).get("sub_center_K", 3),
            ).to(self.device)
        self.num_classes = num_classes

    @staticmethod
    def _strip_frozen_params(param_groups: list[dict]) -> list[dict]:
        cleaned: list[dict] = []
        for g in param_groups:
            params = [p for p in g["params"] if p.requires_grad]
            if params:
                cleaned.append({**g, "params": params})
        return cleaned

    def _compute_boosting_loss(
        self,
        k: int,
        pk_emb: Tensor,
        batch_labels: Tensor,
        is_same: Tensor,
        pidx_a: Tensor,
        pidx_b: Tensor,
        ensemble_embs: Tensor,
        train_labels: Tensor,
        loss_name: str,
    ) -> Tensor:
        """Compute the boosting loss for partition k's current pair-batch.

        pk_emb is shape (2B, K) — the concat of partition-k outputs for the
        a-half and b-half of the mined pair-batch. batch_labels is (2B,);
        is_same / pidx_a / pidx_b are (B,) describing the original pairs.
        """
        B = is_same.shape[0]
        # 2B is_same flag aligned with the concat batch order [a..., b...]
        is_same_2b = torch.cat([is_same, is_same], dim=0)

        # Per-pair previous-ensemble cosine score (in [-1, 1]); used by
        # arcface_reweighted to weight examples and by arcface_margin to
        # split hard/easy.
        with torch.no_grad():
            ens_a = F.normalize(ensemble_embs[pidx_a.cpu()].to(self.device).float(), dim=1)
            ens_b = F.normalize(ensemble_embs[pidx_b.cpu()].to(self.device).float(), dim=1)
            prev_score = (ens_a * ens_b).sum(dim=1).clamp(-1.0, 1.0)

        if isinstance(self.boosting_loss_fn, TripletLoss):
            return self.boosting_loss_fn(pk_emb, batch_labels)

        elif isinstance(self.boosting_loss_fn, ContrastiveLoss):
            emb_a = pk_emb[:B]
            emb_b = pk_emb[B: 2 * B]
            return self.boosting_loss_fn(emb_a, emb_b, is_same.float())

        elif isinstance(self.boosting_loss_fn, ArcFaceReweighted):
            cosine = self.partition_arcface_heads[k](pk_emb)
            # Weight: genuine pair → 1 - prev_score (low for already-easy genuine);
            # impostor pair → (1 + prev_score)/2 (high for confused impostors).
            pair_w = torch.where(
                is_same.bool(),
                (1.0 - prev_score).clamp(0.0, 1.0),
                ((1.0 + prev_score) * 0.5).clamp(0.0, 1.0),
            )
            weights = torch.cat([pair_w, pair_w], dim=0)
            return self.boosting_loss_fn(cosine, batch_labels, weights)

        elif isinstance(self.boosting_loss_fn, ArcFaceMargin):
            head = self.partition_arcface_heads[k]
            cosine = head(pk_emb)
            # Pair difficulty: high for genuine with low prev_score, or impostor
            # with high prev_score. Split top half = hard, bottom half = easy.
            difficulty = torch.where(is_same.bool(), 1.0 - prev_score, prev_score + 1.0)
            n_hard = max(1, B // 2)
            _, hard_pair_idx = difficulty.topk(n_hard)
            hard_mask = torch.zeros(B, dtype=torch.bool, device=pk_emb.device)
            hard_mask[hard_pair_idx] = True
            hard_2b = torch.cat([hard_mask, hard_mask], dim=0)
            hard_cosine = cosine[hard_2b]
            hard_labels = batch_labels[hard_2b]
            easy_cosine = cosine[~hard_2b]
            easy_labels = batch_labels[~hard_2b]
            return self.boosting_loss_fn(hard_cosine, hard_labels, easy_cosine, easy_labels)

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

    def _get_source_dataset(self) -> Dataset:
        """Materialize the underlying training dataset (memoised)."""
        if self._train_dataset is not None:
            return self._train_dataset
        from ppi.data import build_dataloader
        loader, _ = build_dataloader(self.config, split="train")
        self._train_dataset = loader.dataset
        return self._train_dataset

    def _get_train_loader(self, shuffle: bool = True) -> DataLoader:
        ds = self._get_source_dataset()
        batch_size = self.config.get("training", {}).get("batch_size", 256)
        num_workers = self.config.get("data", {}).get("num_workers", 4)
        seed = self.config.get("seed", 42)

        def _worker_init(worker_id: int) -> None:
            import random as _r
            _r.seed(seed + worker_id)
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init,
        )

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
