"""Tests for partition dropout module."""

from __future__ import annotations

import random
from collections import Counter

import pytest
import torch
from scipy import stats

from ppi.training.partition_dropout import PartitionDropout


class TestPartitionDropout:
    def test_distribution_validation(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            PartitionDropout(distribution=[0.5, 0.5, 0.5, 0.5])

    def test_eval_mode_passthrough(self):
        """In eval mode, all partitions pass through unchanged."""
        dropout = PartitionDropout(num_partitions=3)
        dropout.eval()
        parts = [torch.randn(4, 8) for _ in range(3)]
        result = dropout(parts)
        for orig, out in zip(parts, result):
            assert torch.equal(orig, out)

    def test_dropped_partitions_are_zero(self):
        """Dropped partitions must be exactly zero."""
        # Use a distribution that always drops to 1 partition
        dropout = PartitionDropout(
            num_partitions=3,
            distribution=[1.0, 0.0, 0.0, 0.0],
        )
        dropout.train()
        parts = [torch.ones(4, 8) for _ in range(3)]

        for _ in range(20):
            result = dropout(parts)
            active = sum(1 for r in result if r.any())
            assert active == 1
            for r in result:
                assert (r == 0.0).all() or torch.equal(r, torch.ones(4, 8))

    def test_non_dropped_unmodified(self):
        """Active partitions should be bitwise identical to originals."""
        dropout = PartitionDropout(
            num_partitions=3,
            distribution=[1.0, 0.0, 0.0, 0.0],
        )
        dropout.train()
        parts = [torch.randn(4, 8) for _ in range(3)]

        result = dropout(parts)
        for orig, out in zip(parts, result):
            if out.any():
                assert torch.equal(orig, out)

    def test_zero_partition_case(self):
        """100% chance of 0 partitions → all outputs zero."""
        dropout = PartitionDropout(
            num_partitions=3,
            distribution=[0.0, 0.0, 0.0, 1.0],
        )
        dropout.train()
        parts = [torch.ones(2, 8) for _ in range(3)]
        result = dropout(parts)
        for r in result:
            assert (r == 0.0).all()

    def test_all_partition_case(self):
        """100% chance of 3 partitions → all outputs unchanged."""
        dropout = PartitionDropout(
            num_partitions=3,
            distribution=[0.0, 0.0, 1.0, 0.0],
        )
        dropout.train()
        parts = [torch.randn(2, 8) for _ in range(3)]
        result = dropout(parts)
        for orig, out in zip(parts, result):
            assert torch.equal(orig, out)

    def test_statistical_distribution(self):
        """Chi-squared test: sampled widths match configured distribution."""
        dist = [0.40, 0.30, 0.20, 0.10]
        dropout = PartitionDropout(num_partitions=3, distribution=dist)
        dropout.train()

        n_samples = 10_000
        width_counts = Counter()
        parts = [torch.ones(1, 4) for _ in range(3)]

        random.seed(42)
        for _ in range(n_samples):
            result = dropout(parts)
            active = sum(1 for r in result if r.any())
            width_counts[active] += 1

        # Expected: [1-part: 40%, 2-part: 30%, 3-part: 20%, 0-part: 10%]
        observed = [width_counts[1], width_counts[2], width_counts[3], width_counts[0]]
        expected = [d * n_samples for d in dist]

        chi2, p_value = stats.chisquare(observed, expected)
        # Fail if p < 0.001 (very unlikely under correct implementation)
        assert p_value > 0.001, (
            f"Distribution mismatch: chi2={chi2:.1f}, p={p_value:.4f}, "
            f"observed={observed}, expected={expected}"
        )

    def test_batch_consistent_mask(self):
        """All samples in the batch should have the same mask."""
        dropout = PartitionDropout(
            num_partitions=3,
            distribution=[1.0, 0.0, 0.0, 0.0],
        )
        dropout.train()
        # Each sample has different values
        parts = [torch.arange(8).float().unsqueeze(0).expand(4, -1) + i
                 for i in range(3)]

        for _ in range(20):
            result = dropout(parts)
            # Check which partitions are active
            mask = [r.any().item() for r in result]
            # All samples should see the same mask (same partition active)
            for r in result:
                if mask[0]:
                    # If active for sample 0, should be active for all
                    assert r.any(dim=1).all() or not r.any()
