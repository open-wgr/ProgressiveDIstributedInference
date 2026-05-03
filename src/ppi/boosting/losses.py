"""Boosting loss functions for Direction 2 partition training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ArcFaceReweighted(nn.Module):
    """ArcFace with per-pair weights from previous ensemble confidence.

    Weight = 1 - prev_score for genuine pairs, prev_score for impostors.
    Acknowledged geometric assumption violation (ArcFace is a classification
    loss, not a metric loss); included for ablation completeness.
    """

    def __init__(self, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        cosine: Tensor,
        labels: Tensor,
        pair_weights: Tensor,
    ) -> Tensor:
        cosine = cosine.float().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        weights = pair_weights.float().clamp(0.0, 1.0)
        return (per_sample_loss * weights).mean()


class ArcFaceMargin(nn.Module):
    """Two-loss: margin-based on hard pairs, ArcFace as regulariser on easy pairs.

    total_loss = hard_loss + easy_loss_weight * arcface_loss_on_easy_pairs
    """

    def __init__(
        self,
        s: float = 64.0,
        m: float = 0.5,
        easy_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.easy_loss_weight = easy_loss_weight
        self._arcface = _ArcFaceCore(s=s, m=m)

    def forward(
        self,
        hard_cosine: Tensor,
        hard_labels: Tensor,
        easy_cosine: Tensor,
        easy_labels: Tensor,
    ) -> Tensor:
        hard_loss = self._arcface(hard_cosine, hard_labels)
        if easy_cosine.shape[0] == 0:
            return hard_loss
        easy_loss = self._arcface(easy_cosine, easy_labels)
        return hard_loss + self.easy_loss_weight * easy_loss


class TripletLoss(nn.Module):
    """Triplet loss with anchors, positives, and negatives from hard pairs.

    Default loss. Optimises pairwise relative ordering directly, matching
    the verification evaluation objective more closely than classification losses.

    mining_strategy: "batch_hard" (hardest positive/negative per anchor in batch)
                  or "semi_hard" (hardest negative further than positive).
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining_strategy: str = "batch_hard",
    ) -> None:
        super().__init__()
        if mining_strategy not in ("batch_hard", "semi_hard"):
            raise ValueError(f"Unknown triplet mining strategy '{mining_strategy}'.")
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        emb = F.normalize(embeddings.float(), dim=1)
        # Pairwise cosine distance matrix (1 - cosine_sim)
        sim = emb @ emb.T
        dist = 1.0 - sim

        B = emb.shape[0]
        labels = labels.view(-1)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        diff = ~same
        eye = torch.eye(B, dtype=torch.bool, device=emb.device)
        same_no_diag = same & ~eye

        if self.mining_strategy == "batch_hard":
            # Hardest positive: largest distance among same-class pairs
            pos_dist = dist.clone()
            pos_dist[~same_no_diag] = -1.0
            ap = pos_dist.max(dim=1).values  # (B,)

            # Hardest negative: smallest distance among different-class pairs
            neg_dist = dist.clone()
            neg_dist[~diff] = 1e9
            an = neg_dist.min(dim=1).values  # (B,)

        else:  # semi_hard
            ap_all = dist.clone()
            ap_all[~same_no_diag] = 0.0
            ap = (ap_all * same_no_diag.float()).sum(dim=1) / (same_no_diag.float().sum(dim=1) + 1e-12)

            # Semi-hard negative: negative further than positive but within margin
            neg_dist = dist.clone()
            neg_dist[~diff] = 1e9
            semi_mask = diff & (dist > ap.unsqueeze(1)) & (dist < ap.unsqueeze(1) + self.margin)
            semi_neg = dist.clone()
            semi_neg[~semi_mask] = 1e9
            an = semi_neg.min(dim=1).values
            # Fall back to hard negative if no semi-hard found
            hard_neg = neg_dist.min(dim=1).values
            fallback = an >= 1e8
            an = torch.where(fallback, hard_neg, an)

        # Only include anchors that have both valid positives and negatives
        valid = (ap >= 0) & (an < 1e8)
        if not valid.any():
            return embeddings.sum() * 0.0

        loss = F.relu(ap[valid] - an[valid] + self.margin)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss on hard genuine/impostor pairs."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        is_same: Tensor,
    ) -> Tensor:
        emb_a = F.normalize(emb_a.float(), dim=1)
        emb_b = F.normalize(emb_b.float(), dim=1)
        dist = (emb_a - emb_b).pow(2).sum(dim=1).sqrt()
        is_same = is_same.float()
        loss = is_same * dist.pow(2) + (1.0 - is_same) * F.relu(self.margin - dist).pow(2)
        return loss.mean()


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace: K sub-centers per class to handle intra-class variation."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        s: float = 64.0,
        m: float = 0.5,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.K = K
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes * K, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self._arcface_core = _ArcFaceCore(s=s, m=m)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        emb = F.normalize(embeddings.float(), dim=1)
        w_norm = F.normalize(self.weight.float(), dim=1)
        # (B, num_classes * K)
        cosine_all = F.linear(emb, w_norm)
        # (B, num_classes, K) → max over sub-centers
        cosine = cosine_all.view(-1, self.num_classes, self.K).max(dim=2).values
        return self._arcface_core(cosine, labels)


def build_loss(config: dict) -> nn.Module:
    """Factory. Reads config['boosting']['loss'] and instantiates."""
    boosting_cfg = config.get("boosting", {})
    loss_name: str = boosting_cfg.get("loss", "triplet")
    arcface_cfg = config.get("arcface", {})
    s: float = arcface_cfg.get("s", 64.0)
    m: float = arcface_cfg.get("m", 0.5)

    if loss_name == "arcface_reweighted":
        return ArcFaceReweighted(s=s, m=m)
    elif loss_name == "arcface_margin":
        return ArcFaceMargin(s=s, m=m, easy_loss_weight=boosting_cfg.get("easy_loss_weight", 0.3))
    elif loss_name == "triplet":
        return TripletLoss(
            margin=boosting_cfg.get("triplet_margin", 0.3),
            mining_strategy=boosting_cfg.get("triplet_mining", "batch_hard"),
        )
    elif loss_name == "contrastive":
        return ContrastiveLoss(margin=boosting_cfg.get("contrastive_margin", 1.0))
    elif loss_name == "sub_center_arcface":
        num_classes: int = arcface_cfg.get("num_classes", 10572)
        embedding_dim: int = config.get("partitions", {}).get("K", 128)
        return SubCenterArcFace(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            s=s,
            m=m,
            K=boosting_cfg.get("sub_center_K", 3),
        )
    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            "Choose from: arcface_reweighted, arcface_margin, triplet, "
            "contrastive, sub_center_arcface"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ArcFaceCore(nn.Module):
    """Shared ArcFace margin computation used by ArcFaceMargin and SubCenterArcFace."""

    def __init__(self, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine: Tensor, labels: Tensor) -> Tensor:
        cosine = cosine.float().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return F.cross_entropy(logits, labels)
