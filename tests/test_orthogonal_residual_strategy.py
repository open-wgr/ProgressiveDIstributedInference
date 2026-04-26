"""Tests for Variant D: OrthogonalResidualPartitionStrategy."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from ppi.backbones.resnet import PartitionedResNet
from ppi.heads.arcface import ArcFaceHead
from ppi.losses.arcface_loss import ArcFaceLoss
from ppi.partitions.base import PartitionStrategy
from ppi.partitions.orthogonal_residual import OrthogonalResidualPartitionStrategy
from ppi.training.partition_dropout import PartitionDropout


NUM_CLASSES = 10
NUM_PARTITIONS = 3
K = 8
EMBED_DIM = NUM_PARTITIONS * K


def _make_config(
    *,
    lambda_orth: tuple[float, float, float] = (0.0, 0.05, 0.05),
    warmup_epochs: tuple[int, int, int] = (0, 1, 1),
    orth_eps: float = 0.1,
    phase_epochs: tuple[int, int, int] = (12, 7, 4),
    min_epochs: tuple[int, int, int] = (8, 4, 2),
    plateau_window_epochs: int = 9999,
) -> dict:
    phases = [
        {
            "name": "phase1",
            "epochs": phase_epochs[0],
            "min_epochs": min_epochs[0],
            "trainable": ["backbone", "f_0", "arcface_slot_0"],
            "subset_mix": {"[0]": 1.0},
            "lambda_orth": lambda_orth[0],
            "warmup_epochs": warmup_epochs[0],
        },
        {
            "name": "phase2",
            "epochs": phase_epochs[1],
            "min_epochs": min_epochs[1],
            "trainable": ["f_1", "arcface_slot_1"],
            "subset_mix": {"[0,1]": 1.0},
            "lambda_orth": lambda_orth[1],
            "warmup_epochs": warmup_epochs[1],
        },
        {
            "name": "phase3",
            "epochs": phase_epochs[2],
            "min_epochs": min_epochs[2],
            "trainable": ["f_2", "arcface_slot_2"],
            "subset_mix": {
                "[0,1,2]": 0.70,
                "[0,2]":   0.15,
                "[1,2]":   0.15,
            },
            "lambda_orth": lambda_orth[2],
            "warmup_epochs": warmup_epochs[2],
        },
    ]
    return {
        "variant": "orthogonal_residual",
        "partitions": {
            "num_partitions": NUM_PARTITIONS,
            "K": K,
            "dropout": {"distribution": [0.4, 0.3, 0.2, 0.1]},
        },
        "residual": {
            "phases": phases,
            "orth_eps": orth_eps,
            "fine_tune": {"enabled": False, "epochs": 1, "lr_scale": 0.1},
        },
        "early_stop": {
            "plateau_window_epochs": plateau_window_epochs,
            "plateau_threshold": 0.001,
        },
    }


def _unit_partitions(batch_size: int = 4) -> list[torch.Tensor]:
    """Return NUM_PARTITIONS random unit-norm partitions of shape (B, K)."""
    raw = [torch.randn(batch_size, K) for _ in range(NUM_PARTITIONS)]
    return [F.normalize(p, dim=1, eps=1e-12) for p in raw]


def _dummy_backbone_output(batch_size: int = 4) -> dict:
    return {
        "features": torch.randn(batch_size, 512),
        "partitions": _unit_partitions(batch_size),
    }


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


class TestFactory:
    def test_variant_d_resolves_to_orthogonal_residual(self):
        strategy = PartitionStrategy.from_config(_make_config())
        assert isinstance(strategy, OrthogonalResidualPartitionStrategy)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            PartitionStrategy.from_config({"variant": "combined"})


# ---------------------------------------------------------------------------
# Orthogonality loss
# ---------------------------------------------------------------------------


class TestOrthogonalityLoss:
    def test_phase_zero_returns_zero(self):
        strategy = OrthogonalResidualPartitionStrategy(_make_config())
        partitions = _unit_partitions()
        loss = strategy._orthogonality_loss(partitions, phase=0)
        assert loss.item() == 0.0

    def test_fine_tune_phase_returns_zero(self):
        strategy = OrthogonalResidualPartitionStrategy(_make_config())
        partitions = _unit_partitions()
        loss = strategy._orthogonality_loss(partitions, phase=NUM_PARTITIONS)
        assert loss.item() == 0.0

    def test_zero_lambda_returns_zero(self):
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(lambda_orth=(0.0, 0.0, 0.0))
        )
        partitions = _unit_partitions()
        loss = strategy._orthogonality_loss(partitions, phase=2)
        assert loss.item() == 0.0

    def test_eps_guard_skips_zero_init_partition(self):
        """When P_k ≈ 0 (zero-init head), orth loss is skipped even if λ > 0."""
        strategy = OrthogonalResidualPartitionStrategy(_make_config())
        partitions = _unit_partitions()
        # Force P_1 to zero (simulating zero-init head output before training)
        partitions[1] = torch.zeros_like(partitions[1])
        loss = strategy._orthogonality_loss(partitions, phase=1)
        assert loss.item() == 0.0

    def test_orthogonal_partitions_zero_loss(self):
        """Orthogonal P_k vs P_0 → cos² = 0 → loss = 0."""
        strategy = OrthogonalResidualPartitionStrategy(_make_config())

        torch.manual_seed(0)
        B = 4
        # Build P_0 and P_1 in deliberately orthogonal directions per row.
        p0 = torch.zeros(B, K); p0[:, 0] = 1.0
        p1 = torch.zeros(B, K); p1[:, 1] = 1.0
        p2 = F.normalize(torch.randn(B, K), dim=1)
        partitions = [p0, p1, p2]

        loss = strategy._orthogonality_loss(partitions, phase=1)
        assert pytest.approx(loss.item(), abs=1e-7) == 0.0

    def test_aligned_partitions_lambda_loss(self):
        """P_k == P_0 → mean cos² = 1 → loss = λ_k · 1."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(lambda_orth=(0.0, 0.05, 0.05))
        )

        B = 4
        p0 = F.normalize(torch.randn(B, K), dim=1)
        partitions = [p0, p0.clone(), F.normalize(torch.randn(B, K), dim=1)]

        loss = strategy._orthogonality_loss(partitions, phase=1)
        # Sum is over j<k=1 → just j=0, mean batch ⟨p0,p0⟩²=1, then × λ=0.05
        assert pytest.approx(loss.item(), rel=1e-5) == 0.05

    def test_phase2_sums_over_two_priors(self):
        """For phase=2, loss = λ · (cos²(P2,P0) + cos²(P2,P1))."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(lambda_orth=(0.0, 0.0, 0.05))
        )

        B = 4
        # P0, P1 both equal P2 → each pairwise cos² = 1
        p2 = F.normalize(torch.randn(B, K), dim=1)
        partitions = [p2.clone(), p2.clone(), p2]

        loss = strategy._orthogonality_loss(partitions, phase=2)
        # 0.05 * (1 + 1) = 0.1
        assert pytest.approx(loss.item(), rel=1e-5) == 0.10

    def test_lambda_scales_loss_linearly(self):
        """Doubling λ doubles the orth term (everything else equal)."""
        B = 4
        torch.manual_seed(7)
        partitions_a = _unit_partitions(B)
        # Same partitions, two strategies with different λ
        strat_low = OrthogonalResidualPartitionStrategy(
            _make_config(lambda_orth=(0.0, 0.05, 0.0))
        )
        strat_hi = OrthogonalResidualPartitionStrategy(
            _make_config(lambda_orth=(0.0, 0.10, 0.0))
        )

        loss_low = strat_low._orthogonality_loss(partitions_a, phase=1)
        loss_hi = strat_hi._orthogonality_loss(partitions_a, phase=1)

        if loss_low.item() == 0.0:
            pytest.skip("Random partitions tripped eps guard; rerun with different seed")
        assert pytest.approx(loss_hi.item() / loss_low.item(), rel=1e-5) == 2.0

    def test_gradient_flows_to_p_k_only(self):
        """∂L_orth/∂P_k is non-zero; ∂L_orth/∂P_j (j<k) is zero (priors detached)."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(lambda_orth=(0.0, 0.05, 0.05))
        )

        B = 4
        p0 = F.normalize(torch.randn(B, K), dim=1).requires_grad_(True)
        p1 = F.normalize(torch.randn(B, K), dim=1).requires_grad_(True)
        p2 = F.normalize(torch.randn(B, K), dim=1).requires_grad_(True)
        partitions = [p0, p1, p2]

        loss = strategy._orthogonality_loss(partitions, phase=2)
        loss.backward()

        assert p2.grad is not None and p2.grad.abs().sum().item() > 0.0, (
            "P_k must receive gradient from orthogonality loss"
        )
        assert p0.grad is None or p0.grad.abs().sum().item() == 0.0, (
            "P_0 (frozen prior) must NOT receive gradient — should be detached"
        )
        assert p1.grad is None or p1.grad.abs().sum().item() == 0.0, (
            "P_1 (frozen prior) must NOT receive gradient — should be detached"
        )


# ---------------------------------------------------------------------------
# Subset warmup
# ---------------------------------------------------------------------------


class TestSubsetWarmup:
    def test_warmup_forces_full_prefix_in_phase2(self):
        """Phase 2 with warmup_epochs=1: must sample {0,1,2} during epoch 0."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(warmup_epochs=(0, 1, 1))
        )
        strategy._current_phase = 2
        strategy._phase_epoch_count = 0  # epoch 0 of phase 2

        random.seed(0)
        for _ in range(50):
            assert strategy._sample_subset(2) == frozenset({0, 1, 2})

    def test_after_warmup_distribution_matches_mix(self):
        """After warmup, the configured 70/15/15 mix is sampled."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(warmup_epochs=(0, 1, 1))
        )
        strategy._current_phase = 2
        strategy._phase_epoch_count = 1  # past warmup

        random.seed(42)
        counts: dict = {}
        n = 5000
        for _ in range(n):
            s = strategy._sample_subset(2)
            counts[s] = counts.get(s, 0) + 1

        assert abs(counts.get(frozenset({0, 1, 2}), 0) / n - 0.70) < 0.03
        assert abs(counts.get(frozenset({0, 2}), 0) / n - 0.15) < 0.03
        assert abs(counts.get(frozenset({1, 2}), 0) / n - 0.15) < 0.03

    def test_warmup_zero_delegates_to_super(self):
        """warmup_epochs=0 → behaviour identical to parent class on epoch 0."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(warmup_epochs=(0, 0, 0))
        )
        strategy._current_phase = 2
        strategy._phase_epoch_count = 0

        random.seed(123)
        counts: dict = {}
        n = 2000
        for _ in range(n):
            counts[strategy._sample_subset(2)] = counts.get(strategy._sample_subset(2), 0) + 1

        # Should see all three subsets, not just full prefix
        assert len(counts) >= 2, (
            f"warmup_epochs=0 must NOT force full prefix; got only {set(counts.keys())}"
        )

    def test_phase1_warmup_full_prefix_is_01(self):
        """Phase 1 warmup uses {0,1}, not {0,1,2}."""
        strategy = OrthogonalResidualPartitionStrategy(
            _make_config(warmup_epochs=(0, 1, 1))
        )
        strategy._current_phase = 1
        strategy._phase_epoch_count = 0

        for _ in range(20):
            assert strategy._sample_subset(1) == frozenset({0, 1})


# ---------------------------------------------------------------------------
# training_step integration
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def _setup(self):
        strategy = OrthogonalResidualPartitionStrategy(_make_config())
        strategy.train()
        arcface_head = ArcFaceHead(EMBED_DIM, NUM_CLASSES)
        arcface_loss = ArcFaceLoss(s=64.0, m=0.5)
        dropout = PartitionDropout(num_partitions=NUM_PARTITIONS)
        return strategy, arcface_head, arcface_loss, dropout

    def test_metrics_include_orth(self):
        strategy, head, aloss, drop = self._setup()
        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = _dummy_backbone_output()

        loss, metrics = strategy.training_step(out, labels, head, aloss, drop)

        assert "orth" in metrics
        assert "arcface" in metrics
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0

    def test_phase0_orth_metric_is_zero(self):
        """In phase 0 the orth metric must be exactly 0."""
        strategy, head, aloss, drop = self._setup()
        assert strategy._current_phase == 0
        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = _dummy_backbone_output()
        _, metrics = strategy.training_step(out, labels, head, aloss, drop)
        assert metrics["orth"] == 0.0

    def test_phase1_orth_metric_nonzero_with_aligned_partitions(self):
        """When P_1 == P_0 in phase 1, orth metric must be > 0."""
        strategy, head, aloss, drop = self._setup()
        strategy._current_phase = 1

        B = 4
        p0 = F.normalize(torch.randn(B, K), dim=1)
        out = {
            "features": torch.randn(B, 512),
            "partitions": [p0, p0.clone(), F.normalize(torch.randn(B, K), dim=1)],
        }
        labels = torch.randint(0, NUM_CLASSES, (B,))

        _, metrics = strategy.training_step(out, labels, head, aloss, drop)
        assert metrics["orth"] > 0.0

    def test_loss_is_finite(self):
        """Total loss (arcface + orth) must be finite."""
        strategy, head, aloss, drop = self._setup()
        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = _dummy_backbone_output()
        loss, _ = strategy.training_step(out, labels, head, aloss, drop)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Inheritance compatibility — λ=0 everywhere should match Variant C behaviour
# ---------------------------------------------------------------------------


class TestInheritanceCompat:
    def test_phase_advancement_inherited(self):
        config = _make_config(
            lambda_orth=(0.0, 0.0, 0.0),
            warmup_epochs=(0, 0, 0),
            phase_epochs=(1, 1, 1),
            min_epochs=(1, 1, 1),
        )
        strategy = OrthogonalResidualPartitionStrategy(config)
        backbone = PartitionedResNet("resnet18", NUM_PARTITIONS, K, input_size=32)

        for ep in range(1, 4):
            strategy.post_epoch_hook(ep, backbone)
            strategy.phase_changed = False

        assert strategy._current_phase == 3

    def test_arcface_hooks_inherited(self):
        strategy = OrthogonalResidualPartitionStrategy(_make_config())
        head = ArcFaceHead(EMBED_DIM, NUM_CLASSES)

        strategy._install_arcface_hooks(head, phase=1)
        assert len(strategy._arcface_hook_handles) == 1

    def test_pre_training_setup_zero_inits_tails(self):
        """Inherited pre_training_setup must zero f_1, f_2 and freeze them."""
        strategy = OrthogonalResidualPartitionStrategy(_make_config())
        backbone = PartitionedResNet("resnet18", NUM_PARTITIONS, K, input_size=32)
        strategy.pre_training_setup(backbone, _make_config())

        for idx in range(1, NUM_PARTITIONS):
            head = backbone.partition_heads[idx]
            assert head.fc.weight.abs().max().item() == 0.0
            for p in head.parameters():
                assert not p.requires_grad
