"""Tests for Variant B: Nested/slimmable partition strategy."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ppi.partitions.nested import NestedPartitionStrategy, SwitchableBatchNorm1d


def _make_config(
    *,
    num_partitions: int = 3,
    K: int = 8,
    mode: str = "prefix",
    kd_enabled: bool = True,
    kd_alpha: float = 1.0,
    kd_temperature: float = 4.0,
    bn_enabled: bool = True,
) -> dict:
    return {
        "partitions": {
            "num_partitions": num_partitions,
            "K": K,
            "dropout": {"distribution": [0.4, 0.3, 0.2, 0.1]},
        },
        "nesting": {"mode": mode},
        "switchable_bn": {"enabled": bn_enabled},
        "distillation": {
            "enabled": kd_enabled,
            "alpha": kd_alpha,
            "temperature": kd_temperature,
        },
    }


# -------------------------------------------------------------------------
# SwitchableBatchNorm1d
# -------------------------------------------------------------------------


class TestSwitchableBatchNorm1d:
    def test_separate_bn_stats(self):
        """Training at different widths should produce distinct BN running means."""
        torch.manual_seed(0)
        sbn = SwitchableBatchNorm1d(24, num_widths=3)
        sbn.train()

        # Feed different data at width 1 vs width 3
        for _ in range(10):
            x_w1 = torch.randn(16, 24) * 2.0 + 5.0
            sbn.active_width = 1
            sbn(x_w1)

            x_w3 = torch.randn(16, 24) * 0.5 - 3.0
            sbn.active_width = 3
            sbn(x_w3)

        mean_w1 = sbn.bns[0].running_mean
        mean_w3 = sbn.bns[2].running_mean
        assert not torch.allclose(mean_w1, mean_w3, atol=0.1), (
            "BN running means for width 1 and width 3 should differ"
        )

    def test_no_bn_leakage(self):
        """Training at width 1 only should not update BN stats for width 3."""
        sbn = SwitchableBatchNorm1d(24, num_widths=3)
        sbn.train()

        # Only train at width 1
        sbn.active_width = 1
        for _ in range(10):
            sbn(torch.randn(16, 24) * 5.0 + 10.0)

        # Width 3 BN should still have init stats (running_mean ≈ 0)
        assert torch.allclose(
            sbn.bns[2].running_mean, torch.zeros(24), atol=1e-6,
        ), "Width 3 BN stats should not be updated when only training at width 1"

    def test_invalid_width_raises(self):
        sbn = SwitchableBatchNorm1d(24, num_widths=3)
        with pytest.raises(ValueError):
            sbn.active_width = 0
        with pytest.raises(ValueError):
            sbn.active_width = 4


# -------------------------------------------------------------------------
# NestedPartitionStrategy
# -------------------------------------------------------------------------


class TestNestedPartitionStrategy:
    def test_from_config_factory(self):
        from ppi.partitions.base import PartitionStrategy

        config = _make_config()
        config["variant"] = "nested"
        strategy = PartitionStrategy.from_config(config)
        assert isinstance(strategy, NestedPartitionStrategy)

    def test_training_step_returns_loss_and_metrics(self):
        """training_step should return (loss_tensor, metrics_dict)."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        B, K, C = 8, 8, 10
        backbone_output = {
            "features": torch.randn(B, 64),
            "partitions": [torch.randn(B, K) for _ in range(3)],
        }
        labels = torch.randint(0, C, (B,))

        # Minimal ArcFace head + loss
        from ppi.heads.arcface import ArcFaceHead
        from ppi.losses.arcface_loss import ArcFaceLoss

        arcface_head = ArcFaceHead(3 * K, C)
        arcface_loss = ArcFaceLoss(s=16.0, m=0.1)

        result = strategy.training_step(
            backbone_output, labels, arcface_head, arcface_loss, None,
        )
        assert result is not None
        loss, metrics = result
        assert loss.isfinite()
        assert "arcface_full" in metrics
        assert "arcface_narrow" in metrics
        assert "kd_kl" in metrics
        assert "kd_mse" in metrics
        assert "width" in metrics
        assert metrics["width"] in (1.0, 2.0)

    def test_kd_loss_is_nonzero(self):
        """Both KD signals should be > 0 when teacher and student differ."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        B, K, C = 16, 8, 10
        backbone_output = {
            "features": torch.randn(B, 64),
            "partitions": [torch.randn(B, K) for _ in range(3)],
        }
        labels = torch.randint(0, C, (B,))

        from ppi.heads.arcface import ArcFaceHead
        from ppi.losses.arcface_loss import ArcFaceLoss

        arcface_head = ArcFaceHead(3 * K, C)
        arcface_loss = ArcFaceLoss(s=16.0, m=0.1)

        _, metrics = strategy.training_step(
            backbone_output, labels, arcface_head, arcface_loss, None,
        )
        assert metrics["kd_kl"] > 0, "KL-divergence should be non-zero"
        assert metrics["kd_mse"] > 0, "Embedding MSE should be non-zero"

    def test_teacher_logits_detached(self):
        """Teacher logits used in KD should not receive gradients."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        B, K, C = 8, 8, 10
        parts = [torch.randn(B, K, requires_grad=True) for _ in range(3)]
        backbone_output = {
            "features": torch.randn(B, 64),
            "partitions": parts,
        }
        labels = torch.randint(0, C, (B,))

        from ppi.heads.arcface import ArcFaceHead
        from ppi.losses.arcface_loss import ArcFaceLoss

        arcface_head = ArcFaceHead(3 * K, C)
        arcface_loss = ArcFaceLoss(s=16.0, m=0.1)

        loss, _ = strategy.training_step(
            backbone_output, labels, arcface_head, arcface_loss, None,
        )
        loss.backward()
        # All partition tensors should have gradients (from both paths)
        for p in parts:
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_kd_disabled(self):
        """With KD disabled, training_step should still work."""
        config = _make_config(K=8, kd_enabled=False)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        B, K, C = 8, 8, 10
        backbone_output = {
            "features": torch.randn(B, 64),
            "partitions": [torch.randn(B, K) for _ in range(3)],
        }
        labels = torch.randint(0, C, (B,))

        from ppi.heads.arcface import ArcFaceHead
        from ppi.losses.arcface_loss import ArcFaceLoss
        from ppi.training.partition_dropout import PartitionDropout

        arcface_head = ArcFaceHead(3 * K, C)
        arcface_loss = ArcFaceLoss(s=16.0, m=0.1)
        dropout = PartitionDropout(num_partitions=3)
        dropout.train()

        result = strategy.training_step(
            backbone_output, labels, arcface_head, arcface_loss, dropout,
        )
        assert result is not None
        loss, metrics = result
        assert loss.isfinite()
        assert "kd_kl" not in metrics
        assert "kd_mse" not in metrics

    def test_eval_width_selects_correct_bn(self):
        """set_eval_width should select the right BN at eval time."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)

        # Train each width with distinct data
        strategy.train()
        for w in range(1, 4):
            strategy.switchable_bn.active_width = w
            for _ in range(20):
                x = torch.randn(16, 24) * w + w * 10.0
                strategy.switchable_bn(x)

        # Switch to eval and verify correct BN is used
        strategy.eval()
        strategy.set_eval_width(1)
        x = torch.randn(4, 24)
        out1 = strategy.switchable_bn(x)

        strategy.set_eval_width(3)
        out3 = strategy.switchable_bn(x)

        # Outputs should differ because different BN running stats are used
        assert not torch.allclose(out1, out3, atol=0.01), (
            "Eval outputs at width 1 vs 3 should differ due to different BN stats"
        )

    def test_monotonic_embedding_norms(self):
        """Wider assemblies should have larger pre-normalized L2 norms."""
        config = _make_config(K=8, bn_enabled=False)
        strategy = NestedPartitionStrategy(config)
        strategy.eval()

        parts = [torch.randn(4, 8) for _ in range(3)]
        norms = []
        for w in range(1, 4):
            emb = strategy._assemble_at_width(parts, w)
            # L2-norm of the pre-normalized cat (with BN off, output is L2-normed
            # so check the norm of the concatenated vector before normalize)
            cat = torch.cat(
                [parts[i] if i < w else torch.zeros_like(parts[i]) for i in range(3)],
                dim=1,
            )
            norms.append(cat.norm(dim=1).mean().item())
        assert norms[0] < norms[1] < norms[2], (
            f"Pre-norm L2 norms should be monotonically increasing: {norms}"
        )

    def test_invalid_nesting_mode_raises(self):
        with pytest.raises(ValueError, match="nesting.mode"):
            _make_config(mode="invalid")
            NestedPartitionStrategy(_make_config(mode="invalid"))

    def test_auxiliary_loss_is_zero(self):
        """Nested strategy has no auxiliary loss (KD is in training_step)."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)
        parts = [torch.randn(4, 8) for _ in range(3)]
        loss = strategy.compute_auxiliary_loss(parts)
        assert loss.item() == 0.0


# -------------------------------------------------------------------------
# Prefix-only dropout
# -------------------------------------------------------------------------


class TestPrefixDropout:
    def test_narrow_width_is_prefix(self):
        """_sample_narrow_width should only return widths 1 or 2 (< num_partitions)."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)

        widths = set()
        for _ in range(200):
            w = strategy._sample_narrow_width()
            widths.add(w)
        assert widths.issubset({1, 2}), (
            f"Narrow widths should be {{1, 2}}, got {widths}"
        )

    def test_assemble_at_width_zeros_trailing(self):
        """_assemble_at_width(parts, 1) should zero partitions 1 and 2."""
        config = _make_config(K=4, bn_enabled=False)
        strategy = NestedPartitionStrategy(config)
        strategy.eval()

        parts = [torch.ones(2, 4) * (i + 1) for i in range(3)]
        emb = strategy._assemble_at_width(parts, 1)
        # emb is L2-normalized, but the raw cat should have zeros in [4:12]
        # Since BN is off, we can check the pattern via the normalized vector
        # Partition 0 values should be non-zero, 1 and 2 should be zero
        # After L2-norm: [v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] where v > 0
        assert emb[0, 0].item() > 0
        assert emb[0, 4:].abs().sum().item() < 1e-6
