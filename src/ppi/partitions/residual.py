"""Variant C: Phase-gated residual boosting partition strategy.

Each partition is trained sequentially.  Earlier partitions are frozen
before the next one trains, forcing it to capture residual information
that prior partitions don't carry.  Hard freezing replaces the soft
constraints (orthogonality loss, prefix ordering) that collapsed in
Variants A and B.

Training phases (3-partition default):
  Phase 0: backbone + f_0 + ArcFace slot-0 train.  f_1, f_2 frozen.
  Phase 1: f_1 + ArcFace slot-1 train.  Backbone, f_0, f_2 frozen.
  Phase 2: f_2 + ArcFace slot-2 train.  Backbone, f_0, f_1 frozen.
  Phase 3 (optional): all parameters fine-tuned at low LR.

Assembly-level BatchNorm is intentionally absent.  The backbone normalises
each partition to unit L2 norm before returning it, so the assembled
embedding's distribution is determined solely by the count of active
partitions.  This makes all 2^{N-1} valid subsets (any subset containing
P0) well-behaved without any per-width or per-subset BN.
"""

from __future__ import annotations

import random as _random
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ppi.partitions.base import PartitionStrategy
from ppi.training.partition_dropout import assemble_embedding


def _parse_subset_key(key: str) -> frozenset[int]:
    """Parse '[0,1]' → frozenset({0, 1})."""
    key = key.strip()
    if key in ("[]", ""):
        return frozenset()
    inner = key.lstrip("[").rstrip("]")
    return frozenset(int(x.strip()) for x in inner.split(",") if x.strip())


def _is_prefix(subset: frozenset[int]) -> bool:
    """Return True if subset == {0, 1, ..., len-1}."""
    n = len(subset)
    return n > 0 and subset == frozenset(range(n))


def _full_prefix_key(num_partitions: int) -> str:
    """Return e.g. '[0,1,2]' for num_partitions=3."""
    return "[" + ",".join(str(i) for i in range(num_partitions)) + "]"


class ResidualPartitionStrategy(PartitionStrategy, nn.Module):
    """Residual boosting via sequential phase-gated training.

    The key mechanism: partition k only trains after partitions 0..k-1
    are frozen.  The ArcFace columns for frozen slots are zeroed via
    gradient hooks so they can't drift silently.

    Implements the full training_step hook so the trainer's default
    PartitionDropout is bypassed.
    """

    handles_own_dropout: bool = True

    def __init__(self, config: dict[str, Any]) -> None:
        nn.Module.__init__(self)

        partitions_cfg = config["partitions"]
        self.num_partitions: int = partitions_cfg["num_partitions"]
        self.K: int = partitions_cfg["K"]

        # Parse phase configs
        residual_cfg = config.get("residual", {})
        self._phase_cfgs: list[dict] = residual_cfg.get("phases", [])
        if not self._phase_cfgs:
            raise ValueError("residual.phases must be a non-empty list in the config")

        # Optional fine-tune phase — subset_mix derived from num_partitions
        ft_cfg = residual_cfg.get("fine_tune", {})
        if ft_cfg.get("enabled", False):
            N = self.num_partitions
            # Full prefix at 50%; remaining 50% split evenly across shorter prefixes
            ft_mix: dict[str, float] = {_full_prefix_key(N): 0.5}
            shorter = N - 1
            if shorter > 0:
                per = 0.5 / shorter
                for width in range(1, N):
                    k = "[" + ",".join(str(i) for i in range(width)) + "]"
                    ft_mix[k] = per
            self._phase_cfgs = list(self._phase_cfgs) + [
                {
                    "name": "fine_tune",
                    "epochs": ft_cfg.get("epochs", 1),
                    "min_epochs": ft_cfg.get("epochs", 1),
                    "trainable": ["all"],
                    "subset_mix": ft_mix,
                    "lr_scale": ft_cfg.get("lr_scale", 0.1),
                }
            ]

        # Pre-parse subset keys for each phase.
        # Default subset_mix for any phase that omits it: full prefix only.
        self._phase_subsets: list[list[tuple[frozenset[int], float]]] = []
        for pc in self._phase_cfgs:
            default_key = _full_prefix_key(self.num_partitions)
            mix = pc.get("subset_mix", {default_key: 1.0})
            parsed = [(_parse_subset_key(k), float(v)) for k, v in mix.items()]
            total = sum(w for _, w in parsed)
            self._phase_subsets.append([(s, w / total) for s, w in parsed])

        # Early-stop config — plateau detection runs over epoch-level losses,
        # not per-step losses, so the window is meaningful regardless of
        # dataset size or batch size.
        es_cfg = config.get("early_stop", {})
        self._plateau_window_epochs: int = es_cfg.get("plateau_window_epochs", 3)
        self._plateau_threshold: float = es_cfg.get("plateau_threshold", 0.001)

        # Phase state
        self._current_phase: int = 0
        self._phase_epoch_count: int = 0
        self.phase_changed: bool = False

        # ArcFace slot gradient hooks
        self._arcface_hook_handles: list = []
        self._hooks_phase: int = -1

        # Epoch-level loss history for plateau detection (one entry per epoch)
        self._epoch_loss_history: list[float] = []

        # Eval state
        self._eval_width: int = self.num_partitions

    # ------------------------------------------------------------------
    # PartitionStrategy interface
    # ------------------------------------------------------------------

    def pre_training_setup(self, model: nn.Module, config: dict[str, Any]) -> None:
        """Tiny-random-init f_1..f_{N-1} tails; freeze non-phase-0 heads.

        fc.weight uses std=1e-3 rather than strict zero so that the backbone's
        F.normalize(head(x), eps=1e-12) receives a small but non-zero input.
        Strict zero-init causes fp16 gradient overflow: the F.normalize backward
        amplifies the upstream gradient by 1/eps = 1e12, which overflows fp16
        and causes the GradScaler to skip every step, permanently stalling the
        head.  With norm(head(x)) ≈ 0.02 the amplification is ~45×, which is
        stable after the scaler's normal startup halvings.
        fc.bias and bn2.bias stay zero so no directional bias is introduced.
        """
        for idx in range(1, self.num_partitions):
            head = model.partition_heads[idx]
            nn.init.normal_(head.fc.weight, std=1e-3)
            nn.init.zeros_(head.fc.bias)
            nn.init.zeros_(head.bn2.bias)

        for idx in range(1, self.num_partitions):
            model.partition_heads[idx].requires_grad_(False)

    def get_trainable_parameters(
        self,
        model: nn.Module,
        phase: int | None = None,
    ) -> list[nn.Parameter]:
        """Return the parameter list for the given phase (default: current)."""
        if phase is None:
            phase = self._current_phase

        if phase >= len(self._phase_cfgs):
            return list(model.parameters())
        if self._phase_cfgs[phase].get("trainable") == ["all"]:
            return list(model.parameters())

        if phase == 0:
            return [
                p for name, p in model.named_parameters()
                if not name.startswith("partition_heads.")
                or name.startswith("partition_heads.0.")
            ]

        head_idx = phase
        if head_idx < self.num_partitions:
            return list(model.partition_heads[head_idx].parameters())

        return list(model.parameters())

    def training_step(
        self,
        backbone_output: dict[str, Any],
        labels: Tensor,
        arcface_head: nn.Module,
        arcface_loss: nn.Module,
        partition_dropout: nn.Module,
    ) -> tuple[Tensor, dict[str, float]]:
        """Custom forward: subset sampling, assembly, loss, norm logging.

        Partitions arrive pre-normalised (unit L2 norm) from the backbone.
        Active partitions are assembled as-is; inactive slots are zeroed.
        No assembly-level BN — the per-partition norm makes it unnecessary
        and would be wrong for non-prefix subsets anyway.
        """
        if self._hooks_phase != self._current_phase:
            self._install_arcface_hooks(arcface_head, self._current_phase)
            self._hooks_phase = self._current_phase

        partitions: list[Tensor] = backbone_output["partitions"]

        # Sample a subset for this batch according to the phase mix.
        # Clamp to the last valid phase if all phases have already completed
        # (can happen when early-stop exhausts budgets before training.epochs).
        sample_phase = min(self._current_phase, len(self._phase_subsets) - 1)
        active_subset = self._sample_subset(sample_phase)

        masked = [
            p if i in active_subset else torch.zeros_like(p)
            for i, p in enumerate(partitions)
        ]

        embedding = assemble_embedding(masked)
        cosine = arcface_head(embedding)
        loss = arcface_loss(cosine, labels)

        # Per-partition norms (detached — diagnostic only)
        norm_metrics = {
            f"norm_{i}": partitions[i].detach().norm(dim=1).mean().item()
            for i in range(self.num_partitions)
        }

        metrics: dict[str, float] = {
            "arcface": loss.item(),
            "width": float(len(active_subset)),
            **norm_metrics,
        }
        return loss, metrics

    def post_epoch_hook(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict | None = None,
    ) -> None:
        """Check phase budget / plateau and advance if needed."""
        if self._current_phase >= len(self._phase_cfgs):
            return

        self._phase_epoch_count += 1

        # Track epoch-level loss for plateau detection
        if metrics is not None:
            epoch_loss = metrics.get("train/epoch_loss_total")
            if epoch_loss is not None:
                self._epoch_loss_history.append(float(epoch_loss))

        pc = self._phase_cfgs[self._current_phase]
        max_epochs: int = pc.get("epochs", 999)
        min_epochs: int = pc.get("min_epochs", 1)

        should_advance = False

        if self._phase_epoch_count >= max_epochs:
            should_advance = True
            print(
                f"[ResidualStrategy] Phase {self._current_phase} budget "
                f"({max_epochs} epochs) exhausted — advancing.",
                flush=True,
            )
        elif self._phase_epoch_count >= min_epochs and self._plateau_detected():
            should_advance = True
            print(
                f"[ResidualStrategy] Phase {self._current_phase} loss plateau "
                f"detected at epoch {self._phase_epoch_count} — advancing early.",
                flush=True,
            )

        if should_advance:
            self._advance_phase(model)

    def post_assembly(self, embedding: Tensor) -> Tensor:
        """Identity — normalisation is handled per-partition in the backbone."""
        return embedding

    def set_eval_width(
        self,
        width: int,
        partition_set: set[int] | None = None,
    ) -> None:
        """Record the active width (no BN to configure)."""
        self._eval_width = width

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_subset(self, phase: int) -> frozenset[int]:
        """Weighted random sample of a partition subset for the given phase."""
        options = self._phase_subsets[phase]
        r = _random.random()
        cumsum = 0.0
        for subset, weight in options:
            cumsum += weight
            if r < cumsum:
                return subset
        return options[-1][0]

    def _plateau_detected(self) -> bool:
        """True if epoch loss hasn't improved by > threshold over the recent window.

        Uses epoch-level loss averages so the effective window is independent
        of dataset size and batch size.
        """
        window = self._plateau_window_epochs
        if len(self._epoch_loss_history) < window:
            return False
        recent = self._epoch_loss_history[-window:]
        half = max(window // 2, 1)
        first_half_avg = sum(recent[:half]) / half
        second_half_avg = sum(recent[half:]) / max(window - half, 1)
        improvement = first_half_avg - second_half_avg
        return improvement < self._plateau_threshold

    def _advance_phase(self, model: nn.Module) -> None:
        """Freeze current phase's modules and move to the next phase."""
        prev_phase = self._current_phase
        self._current_phase += 1
        self._phase_epoch_count = 0
        self._epoch_loss_history = []
        self.phase_changed = True

        if self._current_phase >= len(self._phase_cfgs):
            print(
                f"[ResidualStrategy] All phases complete after phase {prev_phase}.",
                flush=True,
            )
            return

        print(
            f"[ResidualStrategy] → Phase {self._current_phase} "
            f"({self._phase_cfgs[self._current_phase]['name']})",
            flush=True,
        )

        model.requires_grad_(False)

        if self._current_phase < self.num_partitions:
            model.partition_heads[self._current_phase].requires_grad_(True)
        else:
            model.requires_grad_(True)

    def _install_arcface_hooks(
        self,
        arcface_head: nn.Module,
        phase: int,
    ) -> None:
        """Register gradient hooks that zero frozen ArcFace slot columns."""
        for handle in self._arcface_hook_handles:
            handle.remove()
        self._arcface_hook_handles.clear()

        if phase >= self.num_partitions:
            return

        K = self.K
        N = self.num_partitions

        frozen_ranges: list[tuple[int, int]] = [
            (slot * K, (slot + 1) * K)
            for slot in range(N)
            if slot != phase
        ]

        if not frozen_ranges:
            return

        def _hook(grad: Tensor, frozen: list[tuple[int, int]] = frozen_ranges) -> Tensor:
            g = grad.clone()
            for start, end in frozen:
                g[:, start:end] = 0.0
            return g

        self._arcface_hook_handles.append(arcface_head.weight.register_hook(_hook))
