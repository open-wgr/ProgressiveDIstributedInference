"""ArcFace classification head producing cosine similarities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ArcFaceHead(nn.Module):
    """Cosine-similarity classification head for ArcFace.

    Forward returns ``F.linear(x_norm, w_norm)`` — raw cosine similarities
    of shape ``(B, num_classes)``.  The ArcFace margin and scaling are
    applied by the loss function, not here.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x must be pre-normalised (e.g. by assemble_embedding or _assemble_at_width)
        w_norm = F.normalize(self.weight, dim=1, eps=1e-12)
        return F.linear(x, w_norm)
