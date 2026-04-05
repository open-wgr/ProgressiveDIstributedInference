"""ArcFace loss with margin and scaling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss (ArcFace).

    Expects **cosine similarities** as input (from ``ArcFaceHead``),
    not raw logits.
    """

    def __init__(self, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Threshold for numerical stability fallback
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine: Tensor, labels: Tensor) -> Tensor:
        """Compute ArcFace loss.

        Parameters
        ----------
        cosine : Tensor[B, C]
            Cosine similarities from ``ArcFaceHead``.
        labels : Tensor[B]
            Ground-truth class indices.
        """
        # Hard clamp to [-1, 1] — essential for numerical stability.
        # F.linear on normalised vectors should produce values in this range,
        # but floating-point accumulation can push values slightly outside,
        # leading to NaN in the sqrt below.
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        # Numerical stability: when cos(theta) < cos(pi - m), use fallback
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for target class
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s

        return F.cross_entropy(logits, labels)
