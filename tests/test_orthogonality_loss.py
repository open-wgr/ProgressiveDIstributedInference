"""Tests for orthogonality loss and the orthogonal partition strategy."""

from __future__ import annotations

import pytest
import torch

from ppi.losses.orthogonality import OrthogonalityLoss
from ppi.partitions.orthogonal import OrthogonalPartitionStrategy


class TestOrthogonalityLoss:
    """Tests for both 'cosine' and 'correlation' modes."""

    # --- Cosine mode (original) ---

    def test_cosine_orthogonal_inputs_near_zero(self):
        """Orthogonal partition outputs should produce loss near zero (cosine)."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0, mode="cosine")
        K = 16
        p0 = torch.zeros(4, K); p0[:, :5] = 1.0
        p1 = torch.zeros(4, K); p1[:, 5:10] = 1.0
        p2 = torch.zeros(4, K); p2[:, 10:15] = 1.0
        loss = loss_fn([p0, p1, p2])
        assert loss.item() < 1e-6

    def test_cosine_identical_inputs_high_loss(self):
        """Identical partition outputs should produce high loss (cosine)."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0, mode="cosine")
        p = torch.randn(4, 16)
        loss = loss_fn([p, p.clone(), p.clone()])
        assert loss.item() > 0.1

    # --- Correlation mode (new default) ---

    def test_correlation_identical_activations_high_loss(self):
        """Partitions with identical batch activations → high correlation loss."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0, mode="correlation")
        B, K = 32, 16
        # All partitions produce the same pattern across samples
        p = torch.randn(B, K)
        loss = loss_fn([p, p.clone(), p.clone()])
        assert loss.item() > 0.1

    def test_correlation_independent_less_than_identical(self):
        """Independent partitions should have much lower correlation loss
        than identical partitions."""
        B, K = 64, 16
        torch.manual_seed(0)
        loss_fn = OrthogonalityLoss(lambda_orth=1.0, mode="correlation")

        independent = [torch.randn(B, K) for _ in range(3)]
        loss_indep = loss_fn(independent)

        p = torch.randn(B, K)
        identical = [p, p.clone(), p.clone()]
        loss_ident = loss_fn(identical)

        assert loss_indep.item() < loss_ident.item() * 0.5

    def test_correlation_centring_matters(self):
        """Correlation mode should subtract the batch mean before computing
        the Gram matrix. Verify by constructing inputs where the batch mean
        dominates: cosine mode sees high overlap (all vectors ≈ mean direction),
        correlation mode sees the residuals which are independent."""
        B, K = 128, 16
        torch.manual_seed(42)

        # Large shared mean + small independent noise per partition
        mean = torch.randn(1, K) * 10.0
        p0 = mean + torch.randn(B, K) * 0.1
        p1 = mean + torch.randn(B, K) * 0.1
        p2 = mean + torch.randn(B, K) * 0.1

        loss_cos = OrthogonalityLoss(lambda_orth=1.0, mode="cosine")([p0, p1, p2])
        loss_corr = OrthogonalityLoss(lambda_orth=1.0, mode="correlation")([p0, p1, p2])

        # Cosine mode: all vectors ≈ same direction → high off-diagonal
        # Correlation mode: after centring, residuals are independent → low
        assert loss_corr.item() < loss_cos.item() * 0.5, (
            f"Correlation loss ({loss_corr.item():.4f}) should be much lower "
            f"than cosine loss ({loss_cos.item():.4f}) when partitions share "
            f"a dominant mean but have independent residuals"
        )

    # --- Mode-agnostic tests ---

    @pytest.mark.parametrize("mode", ["cosine", "correlation"])
    def test_gradient_flows_to_all_heads(self, mode):
        """Gradients should flow to all three partition outputs."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0, mode=mode)
        parts = [torch.randn(8, 16, requires_grad=True) for _ in range(3)]
        loss = loss_fn(parts)
        loss.backward()
        for p in parts:
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    @pytest.mark.parametrize("mode", ["cosine", "correlation"])
    def test_lambda_scaling(self, mode):
        """Loss should scale linearly with lambda."""
        torch.manual_seed(0)
        parts = [torch.randn(8, 16) for _ in range(3)]
        loss_1 = OrthogonalityLoss(lambda_orth=1.0, mode=mode)(parts)
        loss_01 = OrthogonalityLoss(lambda_orth=0.1, mode=mode)(parts)
        assert torch.allclose(loss_1 * 0.1, loss_01, atol=1e-6)

    def test_zero_lambda(self):
        """Lambda=0 should produce zero loss."""
        loss_fn = OrthogonalityLoss(lambda_orth=0.0)
        parts = [torch.randn(4, 16) for _ in range(3)]
        assert loss_fn(parts).item() == 0.0

    @pytest.mark.parametrize("mode", ["cosine", "correlation"])
    def test_single_partition_returns_zero(self, mode):
        """Single partition should return zero loss (no off-diagonal to penalize)."""
        loss_fn = OrthogonalityLoss(lambda_orth=1.0, mode=mode)
        parts = [torch.randn(4, 16)]
        loss = loss_fn(parts)
        assert loss.item() == 0.0

    def test_invalid_mode_raises(self):
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown orthogonality mode"):
            OrthogonalityLoss(mode="frobenius")

    def test_default_mode_is_correlation(self):
        """Default mode should be 'correlation'."""
        loss_fn = OrthogonalityLoss()
        assert loss_fn.mode == "correlation"


class TestOrthogonalPartitionStrategy:
    def _make_config(self, mode="correlation"):
        return {
            "partitions": {"num_partitions": 3, "K": 8},
            "orthogonality": {"lambda": 0.1, "mode": mode},
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

    @pytest.mark.parametrize("mode", ["cosine", "correlation"])
    def test_strategy_passes_mode(self, mode):
        """Strategy should forward the mode to OrthogonalityLoss."""
        strategy = OrthogonalPartitionStrategy(self._make_config(mode=mode))
        assert strategy.orth_loss.mode == mode

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
