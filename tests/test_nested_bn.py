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

        # Width 3 BN should still have init stats (running_mean = 0)
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

    def test_no_training_step_override(self):
        """Strategy should use the default trainer path (no training_step)."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)
        result = strategy.training_step(
            {"partitions": [torch.randn(4, 8) for _ in range(3)]},
            torch.zeros(4, dtype=torch.long),
            None, None, None,
        )
        assert result is None

    def test_process_partitions_prefix_masking(self):
        """In prefix mode, process_partitions should zero trailing partitions."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        parts = [torch.ones(2, 4) * (i + 1) for i in range(3)]

        # Run many times and collect observed widths
        widths_seen = set()
        for _ in range(200):
            result = strategy.process_partitions(parts)
            # Count non-zero partitions
            n_active = sum(1 for r in result if r.abs().sum() > 0)
            widths_seen.add(n_active)
            # Active partitions should always be a prefix
            for idx, r in enumerate(result):
                if idx < n_active:
                    assert r.abs().sum() > 0, (
                        f"Partition {idx} should be active at width {n_active}"
                    )
                else:
                    assert r.abs().sum() == 0, (
                        f"Partition {idx} should be zeroed at width {n_active}"
                    )

        # Should see widths 0, 1, 2, 3 over 200 samples
        assert widths_seen.issuperset({1, 2, 3}), (
            f"Expected to see widths 1, 2, 3; got {widths_seen}"
        )

    def test_process_partitions_arbitrary_passthrough(self):
        """In arbitrary mode, process_partitions should be identity."""
        config = _make_config(K=4, mode="arbitrary")
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        parts = [torch.randn(2, 4) for _ in range(3)]
        result = strategy.process_partitions(parts)
        for orig, out in zip(parts, result):
            assert torch.equal(orig, out), (
                "Arbitrary mode should pass through unchanged"
            )

    def test_process_partitions_eval_passthrough(self):
        """In eval mode, process_partitions should be identity."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)
        strategy.eval()

        parts = [torch.randn(2, 4) for _ in range(3)]
        result = strategy.process_partitions(parts)
        for orig, out in zip(parts, result):
            assert torch.equal(orig, out)

    def test_post_assembly_applies_bn(self):
        """post_assembly should apply switchable BN and re-normalise."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        # Simulate a training step at width 2
        strategy._last_width = 2
        emb = torch.randn(4, 12)
        result = strategy.post_assembly(emb)

        # Should be L2-normalised
        norms = result.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

        # Should differ from input (BN transforms + re-normalise)
        assert not torch.equal(result, emb)

    def test_post_assembly_skips_width_zero(self):
        """Width 0 (all dropped) should skip BN."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        strategy._last_width = 0
        emb = torch.zeros(4, 12)
        result = strategy.post_assembly(emb)
        assert torch.equal(result, emb)

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
        strategy.set_eval_width(1, partition_set={0})
        x = torch.randn(4, 24)
        out1 = strategy.switchable_bn(x)

        strategy.set_eval_width(3, partition_set={0, 1, 2})
        out3 = strategy.switchable_bn(x)

        assert not torch.allclose(out1, out3, atol=0.01), (
            "Eval outputs at width 1 vs 3 should differ due to different BN stats"
        )

    def test_eval_non_prefix_skips_bn(self):
        """Non-prefix eval configs should skip BN in post_assembly."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)

        # Train BN so it has non-trivial stats
        strategy.train()
        strategy._last_width = 2
        for _ in range(20):
            strategy.post_assembly(torch.randn(8, 12) * 5.0 + 3.0)

        # Eval with a non-prefix config {0, 2}
        strategy.eval()
        strategy.set_eval_width(2, partition_set={0, 2})
        emb = torch.randn(4, 12)
        result = strategy.post_assembly(emb)
        # Should be unchanged — BN skipped
        assert torch.equal(result, emb), (
            "Non-prefix eval config should skip BN"
        )

    def test_eval_prefix_applies_bn(self):
        """Prefix eval configs should apply BN in post_assembly."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)

        strategy.train()
        strategy._last_width = 2
        for _ in range(20):
            strategy.post_assembly(torch.randn(8, 12) * 5.0 + 3.0)

        strategy.eval()
        strategy.set_eval_width(2, partition_set={0, 1})
        emb = torch.randn(4, 12)
        result = strategy.post_assembly(emb)
        assert not torch.equal(result, emb), (
            "Prefix eval config should apply BN"
        )

    def test_invalid_nesting_mode_raises(self):
        with pytest.raises(ValueError, match="nesting.mode"):
            NestedPartitionStrategy(_make_config(mode="invalid"))

    def test_auxiliary_loss_is_zero(self):
        """Nested strategy has no auxiliary loss."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)
        parts = [torch.randn(4, 8) for _ in range(3)]
        loss = strategy.compute_auxiliary_loss(parts)
        assert loss.item() == 0.0

    def test_bn_disabled(self):
        """With BN disabled, post_assembly should be identity."""
        config = _make_config(K=4, bn_enabled=False)
        strategy = NestedPartitionStrategy(config)
        strategy.train()
        strategy._last_width = 2
        emb = torch.randn(4, 12)
        result = strategy.post_assembly(emb)
        assert torch.equal(result, emb)


# -------------------------------------------------------------------------
# Prefix dropout distribution
# -------------------------------------------------------------------------


class TestPrefixDropout:
    def test_sampled_widths(self):
        """_sample_width should produce widths 0, 1, 2, 3."""
        config = _make_config(K=8)
        strategy = NestedPartitionStrategy(config)

        widths = set()
        for _ in range(500):
            w = strategy._sample_width()
            widths.add(w)
        assert widths == {0, 1, 2, 3}, (
            f"Expected widths {{0, 1, 2, 3}}, got {widths}"
        )

    def test_prefix_never_produces_non_prefix(self):
        """In prefix mode, active partitions must always be a contiguous prefix."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        parts = [torch.ones(2, 4) * (i + 1) for i in range(3)]
        for _ in range(200):
            result = strategy.process_partitions(parts)
            # Find which partitions are active
            active = [idx for idx, r in enumerate(result) if r.abs().sum() > 0]
            # Must be a prefix: [0], [0,1], [0,1,2], or []
            if active:
                assert active == list(range(len(active))), (
                    f"Active partitions {active} are not a contiguous prefix"
                )

    def test_assemble_at_width_zeros_trailing(self):
        """process_partitions at width 1 should zero partitions 1 and 2."""
        config = _make_config(K=4)
        strategy = NestedPartitionStrategy(config)
        strategy.train()

        parts = [torch.ones(2, 4) * (i + 1) for i in range(3)]

        # Force width 1 by setting distribution
        strategy._dropout_dist = [1.0, 0.0, 0.0, 0.0]
        result = strategy.process_partitions(parts)

        assert result[0].abs().sum() > 0, "Partition 0 should be active"
        assert result[1].abs().sum() == 0, "Partition 1 should be zeroed"
        assert result[2].abs().sum() == 0, "Partition 2 should be zeroed"
