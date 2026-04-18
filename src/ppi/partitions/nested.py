"""Variant B: Nested/slimmable partition strategy with switchable BN.

The key insight vs Variant A: partitions form an ordered hierarchy enforced
by **prefix-only dropout** (always use {0}, {0,1}, or {0,1,2} — never
arbitrary subsets like {1} or {0,2}).  Each width gets its own BatchNorm
statistics via ``SwitchableBatchNorm1d``, avoiding the distribution mismatch
that would occur if a single BN averaged over wildly different zero-padding
patterns.

Knowledge distillation between widths was tried and removed — it causes
partition collapse because the shared partition outputs create a shortcut
where the network satisfies KD by making extra partitions near-zero.
"""

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
    """Nested/slimmable partitions with prefix dropout and switchable BN.

    Key differences from Variant A (orthogonal):

    - **Structural constraint** instead of loss-based: partitions form an
      ordered hierarchy (partition 0 is always the base).
    - **Prefix-only dropout**: only prefix subsets {0}, {0,1}, {0,1,2} are
      sampled — never arbitrary combinations.  Configurable via
      ``nesting.mode: prefix | arbitrary`` for ablation.
    - **Switchable BatchNorm**: separate BN statistics per width (1, 2, 3)
      on the assembled 3K-dim embedding, applied via ``post_assembly``.
    - **No auxiliary loss**: the partition ordering and per-width BN are
      purely structural.  No soft penalty that can be trivially satisfied.

    Uses the default trainer forward path (no ``training_step`` override).
    Prefix masking happens in ``process_partitions``; BN switching happens
    in ``post_assembly``.
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

        # Dropout distribution — [p_1part, p_2part, p_3part, p_0part]
        dropout_cfg = partitions_cfg.get("dropout", {})
        self._dropout_dist: list[float] = dropout_cfg.get(
            "distribution", [0.4, 0.3, 0.2, 0.1],
        )

        # In prefix mode, we handle dropout ourselves in process_partitions
        # so the trainer should skip its PartitionDropout.
        self.handles_own_dropout = (self.nesting_mode == "prefix")

        # Eval state
        self._eval_width: int = self.num_partitions
        self._eval_is_prefix: bool = True
        # Width sampled during current training step (for post_assembly)
        self._last_width: int = self.num_partitions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_width(self) -> int:
        """Sample a width from the dropout distribution.

        Distribution semantics: ``[p_1part, p_2part, p_3part, p_0part]``.
        Returns the sampled width (0 = all dropped).
        """
        r = _random.random()
        cumsum = 0.0
        for i, p in enumerate(self._dropout_dist):
            cumsum += p
            if r < cumsum:
                if i < self.num_partitions:
                    return i + 1  # widths 1, 2, 3
                else:
                    return 0  # all dropped
        return 0

    # ------------------------------------------------------------------
    # PartitionStrategy interface
    # ------------------------------------------------------------------

    def process_partitions(self, partition_outputs: list[Tensor]) -> list[Tensor]:
        """Apply prefix-only masking during training.

        In ``prefix`` mode (default), this replaces the standard
        ``PartitionDropout``: width *w* always uses partitions
        {0, ..., w-1}.  The trainer's ``PartitionDropout`` still runs
        afterwards but sees already-zeroed partitions and effectively
        becomes a no-op for the zeroed slots.

        In ``arbitrary`` mode, this is an identity — the trainer's
        ``PartitionDropout`` handles masking (for ablation).
        """
        if not self.training or self.nesting_mode == "arbitrary":
            self._last_width = self.num_partitions
            return partition_outputs

        # Prefix mode: sample a width and zero trailing partitions
        width = self._sample_width()
        self._last_width = width

        result = []
        for idx, p in enumerate(partition_outputs):
            if idx < width:
                result.append(p)
            else:
                result.append(torch.zeros_like(p))
        return result

    def post_assembly(self, embedding: Tensor) -> Tensor:
        """Apply the width-appropriate switchable BN after assembly.

        During training, uses the width sampled in ``process_partitions``
        (always a prefix config).

        During eval, BN is only applied for **prefix** partition configs
        (the configs seen during training).  Non-prefix configs (e.g.
        {0,2}) skip BN to avoid distribution mismatch — the BN running
        stats were gathered from prefix-ordered embeddings only.
        """
        if not self.use_switchable_bn:
            return embedding

        if self.training:
            width = self._last_width
        else:
            if not self._eval_is_prefix:
                # Non-prefix eval config — skip BN
                return embedding
            width = self._eval_width

        if width == 0:
            # All partitions dropped — skip BN (embedding is all zeros)
            return embedding

        self.switchable_bn.active_width = width
        # Re-normalise after BN (BN shifts the distribution, breaking
        # the unit-norm property from assemble_embedding)
        return F.normalize(self.switchable_bn(embedding), dim=1, eps=1e-12)

    def set_eval_width(self, width: int, partition_set: set[int] | None = None) -> None:
        """Set the active width and partition config for evaluation.

        Parameters
        ----------
        width:
            Number of active partitions.
        partition_set:
            The actual partition indices (e.g. ``{0, 2}``).  If provided,
            BN is only applied when this is a prefix set.  If ``None``,
            assumes a prefix config.
        """
        self._eval_width = width
        # Check if this is a prefix config: {0}, {0,1}, {0,1,2}
        if partition_set is not None:
            self._eval_is_prefix = (partition_set == set(range(width)))
        else:
            self._eval_is_prefix = True
        if self.use_switchable_bn and self._eval_is_prefix:
            self.switchable_bn.active_width = width

    def get_trainable_parameters(
        self,
        model: nn.Module,
        phase: int | None = None,
    ) -> list[nn.Parameter]:
        """Backbone parameters only; strategy params are added by the trainer."""
        return list(model.parameters())
