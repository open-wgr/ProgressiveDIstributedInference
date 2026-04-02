"""Tests for ArcFace loss function."""

from __future__ import annotations

import torch

from ppi.heads.arcface import ArcFaceHead
from ppi.losses.arcface_loss import ArcFaceLoss


class TestArcFaceLoss:
    def test_finite_positive_loss(self):
        """Random embedding + random target produces finite positive loss."""
        head = ArcFaceHead(64, 10)
        loss_fn = ArcFaceLoss(s=64, m=0.5)
        x = torch.randn(8, 64)
        labels = torch.randint(0, 10, (8,))
        cosine = head(x)
        loss = loss_fn(cosine, labels)
        assert loss.isfinite()
        assert loss > 0

    def test_zero_padded_embedding(self):
        """Zero-padded embedding (simulating missing partitions) → finite loss."""
        head = ArcFaceHead(24, 10)  # 3 partitions × K=8
        loss_fn = ArcFaceLoss()
        # Zero out 2 of 3 partition slots
        x = torch.randn(4, 24)
        x[:, 8:] = 0.0  # only first partition active
        cosine = head(x)
        loss = loss_fn(cosine, torch.randint(0, 10, (4,)))
        assert loss.isfinite()

    def test_all_zero_embedding(self):
        """All-zero embedding (0 partitions) → finite loss, no NaN."""
        head = ArcFaceHead(24, 10)
        loss_fn = ArcFaceLoss()
        x = torch.zeros(4, 24)
        cosine = head(x)
        loss = loss_fn(cosine, torch.randint(0, 10, (4,)))
        assert loss.isfinite()

    def test_gradient_flow_nonzero_slots(self):
        """Gradients flow through non-zero partition slots."""
        head = ArcFaceHead(24, 10)
        loss_fn = ArcFaceLoss()
        x = torch.randn(4, 24, requires_grad=True)
        # Zero out the last 8 dims
        mask = torch.ones(24)
        mask[16:] = 0.0
        x_masked = x * mask
        cosine = head(x_masked)
        loss = loss_fn(cosine, torch.randint(0, 10, (4,)))
        loss.backward()
        # Grad should be non-zero for active slots
        assert x.grad is not None
        assert x.grad[:, :16].abs().sum() > 0

    def test_gradient_zero_for_zero_slots(self):
        """Gradients are zero for zero-padded slots."""
        head = ArcFaceHead(24, 10)
        loss_fn = ArcFaceLoss()
        x = torch.randn(4, 24, requires_grad=True)
        mask = torch.ones(24)
        mask[16:] = 0.0
        x_masked = x * mask
        cosine = head(x_masked)
        loss = loss_fn(cosine, torch.randint(0, 10, (4,)))
        loss.backward()
        # Grad should be zero for masked slots
        assert (x.grad[:, 16:] == 0.0).all()

    def test_margin_increases_loss(self):
        """With margin m>0, loss should be higher than m=0 for same input."""
        head = ArcFaceHead(32, 10)
        loss_no_margin = ArcFaceLoss(s=64, m=0.0)
        loss_with_margin = ArcFaceLoss(s=64, m=0.5)

        torch.manual_seed(42)
        x = torch.randn(8, 32)
        labels = torch.randint(0, 10, (8,))
        cosine = head(x)

        l0 = loss_no_margin(cosine, labels)
        lm = loss_with_margin(cosine, labels)
        assert lm > l0
