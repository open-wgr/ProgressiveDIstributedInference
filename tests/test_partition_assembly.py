"""Tests for embedding assembly from partition subsets."""

from __future__ import annotations

import torch

from ppi.training.partition_dropout import assemble_embedding


class TestAssembleEmbedding:
    def test_all_partitions(self):
        """Assembling all 3 partitions produces a 3K vector with correct slots."""
        K = 8
        p0 = torch.ones(2, K) * 1.0
        p1 = torch.ones(2, K) * 2.0
        p2 = torch.ones(2, K) * 3.0
        result = assemble_embedding([p0, p1, p2])
        assert result.shape == (2, 3 * K)
        # After L2 normalisation, check relative magnitudes are preserved
        # Slot 0 values < slot 1 values < slot 2 values
        assert result[0, 0] < result[0, K]
        assert result[0, K] < result[0, 2 * K]

    def test_subset_zero_pads(self):
        """Missing partitions should be zero after assembly."""
        K = 8
        p0 = torch.ones(2, K)
        p1 = torch.zeros(2, K)  # "missing"
        p2 = torch.ones(2, K)
        result = assemble_embedding([p0, p1, p2])
        assert result.shape == (2, 3 * K)
        # Middle slot should be zero (it was zero before normalisation,
        # and L2-norm scales uniformly)
        assert (result[:, K:2*K] == 0.0).all()

    def test_empty_all_zeros(self):
        """All-zero partitions produce all-zero output (no NaN)."""
        K = 8
        zeros = [torch.zeros(2, K) for _ in range(3)]
        result = assemble_embedding(zeros)
        assert result.shape == (2, 3 * K)
        assert (result == 0.0).all()
        assert torch.isfinite(result).all()

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        K = 8
        parts = [torch.randn(4, K) for _ in range(3)]
        r1 = assemble_embedding(parts)
        r2 = assemble_embedding(parts)
        assert torch.equal(r1, r2)

    def test_unit_norm(self):
        """Non-zero assembled embeddings should have unit L2 norm."""
        K = 16
        parts = [torch.randn(4, K) for _ in range(3)]
        result = assemble_embedding(parts)
        norms = result.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_single_partition_active(self):
        """Only one partition active — others zero, result still unit norm."""
        K = 8
        p0 = torch.randn(2, K)
        p1 = torch.zeros(2, K)
        p2 = torch.zeros(2, K)
        result = assemble_embedding([p0, p1, p2])
        norms = result.norm(dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)
        # Only the first K elements should be non-zero
        assert (result[:, K:] == 0.0).all()
