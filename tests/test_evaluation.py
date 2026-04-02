"""Tests for evaluation metrics and evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from ppi.evaluation.metrics import compute_pair_accuracy, compute_rank1, compute_tar_at_far


class TestTarAtFar:
    def test_perfect_separation(self):
        """Well-separated distributions should yield TAR near 1.0."""
        genuine = np.array([0.9, 0.95, 0.85, 0.92, 0.88])
        impostor = np.array([0.1, 0.05, 0.15, 0.08, 0.12])
        tar = compute_tar_at_far(genuine, impostor, far_target=1e-2)
        assert tar >= 0.8

    def test_overlapping_distributions(self):
        """Overlapping distributions should yield TAR between 0 and 1."""
        rng = np.random.RandomState(42)
        genuine = rng.normal(0.6, 0.15, 1000)
        impostor = rng.normal(0.3, 0.15, 1000)
        tar = compute_tar_at_far(genuine, impostor, far_target=1e-2)
        assert 0.0 <= tar <= 1.0

    def test_identical_distributions(self):
        """Identical distributions: TAR at low FAR should be near FAR itself."""
        rng = np.random.RandomState(42)
        scores = rng.normal(0.5, 0.1, 500)
        tar = compute_tar_at_far(scores, scores.copy(), far_target=0.1)
        assert 0.0 <= tar <= 1.0


class TestRank1:
    def test_perfect_gallery(self):
        """One-hot embeddings with matching labels → rank-1 = 1.0."""
        embs = np.eye(5)
        labels = np.arange(5)
        assert compute_rank1(embs, embs, labels, labels) == 1.0

    def test_wrong_gallery(self):
        """All-same embeddings → rank-1 based on first match, not perfect."""
        query = np.ones((3, 4))
        gallery = np.ones((3, 4))
        q_labels = np.array([0, 1, 2])
        g_labels = np.array([0, 0, 0])
        r1 = compute_rank1(query, gallery, q_labels, g_labels)
        # Only label 0 matches
        assert 0.0 <= r1 <= 1.0


class TestPairAccuracy:
    def test_perfect_pairs(self):
        """Perfectly separated pairs → accuracy near 1.0."""
        n = 100
        e1_same = np.random.RandomState(42).randn(n, 32) + 5
        e2_same = e1_same + np.random.RandomState(43).randn(n, 32) * 0.01
        e1_diff = np.random.RandomState(44).randn(n, 32)
        e2_diff = np.random.RandomState(45).randn(n, 32)

        embs1 = np.vstack([e1_same, e1_diff])
        embs2 = np.vstack([e2_same, e2_diff])
        issame = np.array([True] * n + [False] * n)

        acc, std = compute_pair_accuracy(embs1, embs2, issame, n_folds=5)
        assert acc > 0.9
        assert 0.0 <= std <= 0.5

    def test_deterministic(self):
        """Same inputs produce same results."""
        rng = np.random.RandomState(42)
        embs1 = rng.randn(50, 16)
        embs2 = rng.randn(50, 16)
        issame = np.array([True] * 25 + [False] * 25)

        acc1, _ = compute_pair_accuracy(embs1, embs2, issame, n_folds=5)
        acc2, _ = compute_pair_accuracy(embs1, embs2, issame, n_folds=5)
        assert acc1 == acc2


class TestEvaluatorConfigs:
    def test_all_7_configs_generated(self):
        """_all_partition_configs should return 7 configs for 3 partitions."""
        from ppi.evaluation.evaluator import _all_partition_configs
        configs = _all_partition_configs(3)
        assert len(configs) == 7
        # 3 singles + 3 pairs + 1 full
        singles = [c for c in configs if len(c) == 1]
        pairs = [c for c in configs if len(c) == 2]
        triples = [c for c in configs if len(c) == 3]
        assert len(singles) == 3
        assert len(pairs) == 3
        assert len(triples) == 1
