"""Variant B: Nested/slimmable partition strategy with switchable BN and KD."""

from __future__ import annotations

import random as _random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ppi.partitions.base import PartitionStrategy


class SwitchableBatchNorm1d(nn.Module):
    """Maintains independent BatchNorm1d layers per width configuration.

    During forward, only the BN for the currently-active width is applied.
    This ensures each width sees statistics gathered exclusively from inputs
    at that width, avoiding the distribution mismatch that occurs when a
    single BN averages over wildly different zero-padding patterns.
    """

    def __init__(self, num_features: int, num_widths: int = 3) -> None:
        super().__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features) for _ in range(num_widths)]
        )
        self._active_width: int = num_widths  # default: full width (1-indexed)

    @property
    def active_width(self) -> int:
        return self._active_width

    @active_width.setter
    def active_width(self, width: int) -> None:
        if width < 1 or width > len(self.bns):
            raise ValueError(
                f"width must be in [1, {len(self.bns)}], got {width}"
            )
        self._active_width = width

    def forward(self, x: Tensor) -> Tensor:
        return self.bns[self._active_width - 1](x)


class NestedPartitionStrategy(PartitionStrategy, nn.Module):
    """Nested/slimmable partitions with switchable BN and in-place KD.

    Key differences from Variant A (orthogonal):
    - **Structural** constraint instead of loss-based: partitions form an
      ordered hierarchy (partition 0 is always the base).
    - **Switchable BatchNorm**: separate BN statistics per width (1, 2, 3)
      on the assembled 3K-dim embedding.
    - **In-place knowledge distillation**: each batch does a full-width
      (teacher) and a narrow-width (student) forward through assembly +
      ArcFace.  The narrow logits are pushed toward the full logits via
      KL-divergence.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        nn.Module.__init__(self)
        partitions_cfg = config["partitions"]
        self.num_partitions: int = partitions_cfg["num_partitions"]
        self.K: int = partitions_cfg["K"]
        embedding_dim = self.num_partitions * self.K

        # Nesting mode
        nesting_cfg = config.get("nesting", {})
        self.nesting_mode: str = nesting_cfg.get("mode", "prefix")
        if self.nesting_mode not in ("prefix", "arbitrary"):
            raise ValueError(
                f"nesting.mode must be 'prefix' or 'arbitrary', "
                f"got '{self.nesting_mode}'"
            )

        # Switchable BN — one BN per width (1, 2, 3)
        bn_cfg = config.get("switchable_bn", {})
        self.use_switchable_bn: bool = bn_cfg.get("enabled", True)
        if self.use_switchable_bn:
            self.switchable_bn = SwitchableBatchNorm1d(
                embedding_dim, num_widths=self.num_partitions,
            )

        # Knowledge distillation config
        kd_cfg = config.get("distillation", {})
        self.kd_enabled: bool = kd_cfg.get("enabled", True)
        self.kd_alpha: float = kd_cfg.get("alpha", 1.0)
        self.kd_temperature: float = kd_cfg.get("temperature", 2.0)
        # Embedding-space MSE: pushes student embedding directly toward teacher
        self.emb_mse_alpha: float = kd_cfg.get("emb_mse_alpha", 1.0)

        # Dropout distribution — same semantics as PartitionDropout:
        # [p_1part, p_2part, p_3part, p_0part]
        dropout_cfg = partitions_cfg.get("dropout", {})
        self._dropout_dist: list[float] = dropout_cfg.get(
            "distribution", [0.4, 0.3, 0.2, 0.1],
        )

        # Eval state
        self._eval_width: int = self.num_partitions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assemble_at_width(
        self,
        partition_outputs: list[Tensor],
        width: int,
    ) -> Tensor:
        """Assemble embedding using *width* prefix partitions.

        Partitions beyond the prefix are zero-filled.  The result is run
        through the width-specific BN (if enabled) and then L2-normalised.
        """
        parts: list[Tensor] = []
        for idx in range(self.num_partitions):
            if idx < width:
                parts.append(partition_outputs[idx])
            else:
                parts.append(torch.zeros_like(partition_outputs[idx]))
        cat = torch.cat(parts, dim=1)

        if self.use_switchable_bn:
            self.switchable_bn.active_width = width
            cat = self.switchable_bn(cat)

        return F.normalize(cat, dim=1, eps=1e-12)

    def _sample_narrow_width(self) -> int:
        """Sample a narrow width (< num_partitions) from the dropout dist.

        Only widths 1 .. (num_partitions - 1) are eligible for the student
        path.  We re-normalise the distribution over those widths.
        """
        # Eligible entries: indices 0..(num_partitions-2) correspond to
        # widths 1..(num_partitions-1)
        eligible = self._dropout_dist[: self.num_partitions - 1]
        total = sum(eligible)
        if total <= 0:
            return 1  # fallback
        r = _random.random() * total
        cumsum = 0.0
        for i, p in enumerate(eligible):
            cumsum += p
            if r < cumsum:
                return i + 1  # width is 1-indexed
        return 1  # fallback

    # ------------------------------------------------------------------
    # PartitionStrategy interface
    # ------------------------------------------------------------------

    def training_step(
        self,
        backbone_output: dict[str, Any],
        labels: Tensor,
        arcface_head: nn.Module,
        arcface_loss: nn.Module,
        partition_dropout: nn.Module,
    ) -> tuple[Tensor, dict[str, float]]:
        """Multi-width forward with in-place knowledge distillation."""
        partition_outputs: list[Tensor] = backbone_output["partitions"]

        # --- Teacher: full width -----------------------------------------
        full_emb = self._assemble_at_width(partition_outputs, self.num_partitions)
        full_logits = arcface_head(full_emb)
        loss_full = arcface_loss(full_logits, labels)

        metrics: dict[str, float] = {
            "arcface_full": loss_full.item(),
        }
        total_loss = loss_full

        # --- Student: narrow width ---------------------------------------
        if self.kd_enabled:
            width = self._sample_narrow_width()
            narrow_emb = self._assemble_at_width(partition_outputs, width)
            narrow_logits = arcface_head(narrow_emb)
            loss_narrow = arcface_loss(narrow_logits, labels)

            # --- KL divergence on logits: soft-label signal ---
            # The ArcFace head returns raw cosine similarities in [-1, 1].
            # We must scale by the ArcFace scale factor s (typically 64)
            # before softmax, otherwise 10k+ classes in [-1, 1] produce a
            # near-uniform distribution and KL-divergence collapses to 0.
            T = self.kd_temperature
            s = arcface_loss.s
            log_student = F.log_softmax(narrow_logits * s / T, dim=1)
            teacher_probs = F.softmax(full_logits.detach() * s / T, dim=1)
            kd_kl = (
                F.kl_div(log_student, teacher_probs, reduction="batchmean")
                * (T * T)
            )

            # --- Embedding MSE: direct representation match ---
            # Push the student embedding toward the teacher embedding in
            # L2 space.  Both are already L2-normalised, so MSE is bounded
            # in [0, 4] and directly relates to cosine similarity:
            #   MSE = 2 - 2*cos(student, teacher)
            emb_mse = F.mse_loss(narrow_emb, full_emb.detach())

            kd_loss = self.kd_alpha * kd_kl + self.emb_mse_alpha * emb_mse
            total_loss = total_loss + loss_narrow + kd_loss
            metrics["arcface_narrow"] = loss_narrow.item()
            metrics["kd_kl"] = kd_kl.item()
            metrics["kd_mse"] = emb_mse.item()
            metrics["width"] = float(width)
        else:
            # No KD — just do partition dropout like the default path
            dropped = partition_dropout(partition_outputs)
            narrow_emb = torch.cat(dropped, dim=1)
            if self.use_switchable_bn:
                # Determine active width from dropped outputs
                n_active = sum(
                    1 for d in dropped if d.abs().sum() > 0
                )
                if n_active > 0:
                    self.switchable_bn.active_width = n_active
                    narrow_emb = self.switchable_bn(narrow_emb)
            narrow_emb = F.normalize(narrow_emb, dim=1, eps=1e-12)
            narrow_logits = arcface_head(narrow_emb)
            loss_narrow = arcface_loss(narrow_logits, labels)
            total_loss = total_loss + loss_narrow
            metrics["arcface_narrow"] = loss_narrow.item()

        return total_loss, metrics

    def post_assembly(self, embedding: Tensor) -> Tensor:
        """Apply switchable BN at the current eval width."""
        if self.use_switchable_bn and not self.training:
            self.switchable_bn.active_width = self._eval_width
            # BN expects eval mode to use running stats
            embedding = F.normalize(
                self.switchable_bn(embedding), dim=1, eps=1e-12,
            )
        return embedding

    def set_eval_width(self, width: int) -> None:
        """Set which width's BN to use during evaluation."""
        self._eval_width = width
        if self.use_switchable_bn:
            self.switchable_bn.active_width = width

    def get_trainable_parameters(
        self,
        model: nn.Module,
        phase: int | None = None,
    ) -> list[nn.Parameter]:
        """All model params + switchable BN params."""
        return list(model.parameters())
