"""Tests for orthogonality loss and the orthogonal partition strategy."""

from __future__ import annotations

import torch

from ppi.losses.orthogonality import OrthogonalityLoss
from ppi.partitions.orthogonal import OrthogonalPartitionStrategy


class TestOrthogonalityLoss:
    def test_orthogonal_inputs_near_zero(self):
        """Orthogonal partition outputs should produce loss near zero."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0)
        K = 16
        # Create 3 orthogonal vectors (standard basis in a subspace)
        p0 = torch.zeros(4, K)
        p0[:, :5] = 1.0
        p1 = torch.zeros(4, K)
        p1[:, 5:10] = 1.0
        p2 = torch.zeros(4, K)
        p2[:, 10:15] = 1.0
        loss = loss_fn([p0, p1, p2])
        assert loss.item() < 1e-6

    def test_identical_inputs_high_loss(self):
        """Identical partition outputs should produce high loss."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0)
        K = 16
        p = torch.randn(4, K)
        loss = loss_fn([p, p.clone(), p.clone()])
        assert loss.item() > 0.1

    def test_gradient_flows_to_all_heads(self):
        """Gradients should flow to all three partition outputs."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0)
        parts = [torch.randn(4, 16, requires_grad=True) for _ in range(3)]
        loss = loss_fn(parts)
        loss.backward()
        for p in parts:
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_lambda_scaling(self):
        """Loss should scale linearly with lambda."""
        parts = [torch.randn(4, 16) for _ in range(3)]
        loss_1 = OrthogonalityLoss(lambda_orth=1.0)(parts)
        loss_01 = OrthogonalityLoss(lambda_orth=0.1)(parts)
        assert torch.allclose(loss_1 * 0.1, loss_01, atol=1e-6)

    def test_zero_lambda(self):
        """Lambda=0 should produce zero loss."""
        loss_fn = OrthogonalityLoss(lambda_orth=0.0)
        parts = [torch.randn(4, 16) for _ in range(3)]
        assert loss_fn(parts).item() == 0.0


class TestOrthogonalPartitionStrategy:
    def _make_config(self):
        return {
            "partitions": {"num_partitions": 3, "K": 8},
            "orthogonality": {"lambda": 0.1},
            "positional_encoding": {"type": "learned"},
        }

    def test_process_partitions_adds_encoding(self):
        strategy = OrthogonalPartitionStrategy(self._make_config())
        parts = [torch.zeros(2, 8) for _ in range(3)]
        processed = strategy.process_partitions(parts)
        # After adding positional encoding, outputs should differ
        assert not torch.equal(processed[0], processed[1])

    def test_auxiliary_loss_is_finite(self):
        strategy = OrthogonalPartitionStrategy(self._make_config())
        parts = [torch.randn(4, 8) for _ in range(3)]
        loss = strategy.compute_auxiliary_loss(parts)
        assert loss.isfinite()
        assert loss >= 0

    def test_from_config_factory(self):
        from ppi.partitions.base import PartitionStrategy
        config = self._make_config()
        config["variant"] = "orthogonal"
        strategy = PartitionStrategy.from_config(config)
        assert isinstance(strategy, OrthogonalPartitionStrategy)

    def test_default_strategy_no_variant(self):
        from ppi.partitions.base import DefaultStrategy, PartitionStrategy
        strategy = PartitionStrategy.from_config({"seed": 42})
        assert isinstance(strategy, DefaultStrategy)
