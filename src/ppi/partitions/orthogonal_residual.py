"""Variant D: Phase-gated residual + orthogonality vs frozen partitions.

Extends Variant C (ResidualPartitionStrategy) with two additions:

1. Per-phase orthogonality loss against detached frozen partition references.
   For phase k > 0, adds  λ_k · Σ_{j<k} mean_batch( <P_k, P_j.detach()>² ).
   Variant A's symmetric orthogonality loss collapsed because both sides
   co-trained — random vectors in ℝ^K are near-orthogonal already, so the
   constraint was trivially satisfied.  Here the reference side is fixed,
   creating a genuine repulsive force.  Skipped when ‖P_k‖ < orth_eps so
   the (numerically unstable, semantically meaningless) loss isn't applied
   to a partition whose head hasn't yet warmed out of zero-init.

2. Subset warm-up at phase boundaries.  When phase k starts, P_k ≈ 0, so
   subsets that exclude the previous prefix's partition (e.g. {0,2} when
   k=2) initially behave like degraded {P0}, producing a cold-start spike
   in the width-2 loss.  During the first ``warmup_epochs`` of phase k we
   sample only the full prefix {0..k}; afterwards the configured mix
   resumes.

Everything else (phase advancement, freezing, ArcFace slot gradient hooks,
plateau detection, parameter selection) is inherited unchanged from the
parent class.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ppi.partitions.residual import ResidualPartitionStrategy


class OrthogonalResidualPartitionStrategy(ResidualPartitionStrategy):
    """Variant D strategy.  See module docstring."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        residual_cfg = config.get("residual", {})
        self._orth_eps: float = float(residual_cfg.get("orth_eps", 0.1))

        self._lambda_orth: list[float] = [
            float(pc.get("lambda_orth", 0.0)) for pc in self._phase_cfgs
        ]
        self._warmup_epochs: list[int] = [
            int(pc.get("warmup_epochs", 0)) for pc in self._phase_cfgs
        ]

    def _orthogonality_loss(
        self,
        partitions: list[Tensor],
        phase: int,
    ) -> Tensor:
        """λ_k · Σ_{j<k} mean_batch(<P_k, P_j.detach()>²).

        Returns a scalar zero tensor when:
        - phase == 0 (no priors)
        - phase >= num_partitions (fine-tune phase, no constraint)
        - λ_k == 0
        - ‖P_k‖_mean < orth_eps (head still near zero-init)
        """
        zero = partitions[0].new_zeros(())

        if phase == 0 or phase >= self.num_partitions:
            return zero

        lam = self._lambda_orth[phase] if phase < len(self._lambda_orth) else 0.0
        if lam == 0.0:
            return zero

        p_k = partitions[phase]
        if p_k.norm(dim=1).mean().item() < self._orth_eps:
            return zero

        total = zero
        for j in range(phase):
            p_j = partitions[j].detach()
            cos_sq = (p_k * p_j).sum(dim=1).pow(2).mean()
            total = total + cos_sq

        return lam * total

    def _sample_subset(self, phase: int) -> frozenset[int]:
        """Force full prefix {0..k} during the first ``warmup_epochs`` of phase k."""
        if phase < len(self._warmup_epochs):
            warmup = self._warmup_epochs[phase]
            if warmup > 0 and self._phase_epoch_count < warmup:
                width = min(phase + 1, self.num_partitions)
                return frozenset(range(width))
        return super()._sample_subset(phase)

    def training_step(
        self,
        backbone_output,
        labels,
        arcface_head,
        arcface_loss,
        partition_dropout,
    ):
        loss, metrics = super().training_step(
            backbone_output, labels, arcface_head, arcface_loss, partition_dropout
        )
        partitions: list[Tensor] = backbone_output["partitions"]
        orth = self._orthogonality_loss(partitions, self._current_phase)
        metrics["orth"] = float(orth.detach().item())
        return loss + orth, metrics
