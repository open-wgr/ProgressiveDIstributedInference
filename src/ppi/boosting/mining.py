"""Hard-pair mining from the previous ensemble's failure distribution."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Subset


class HardPairMiner:
    """Dynamic hard-pair mining from the previous ensemble's failure distribution.

    Supports two strategies:
      - band:  pairs whose previous-ensemble cosine score falls in [band_low, band_high]
      - topk:  top-k pairs ranked by previous-ensemble verification loss

    Mining is dynamic: the failure distribution shifts as the current partition
    trains (and the backbone shifts if not frozen), so the pair set is recomputed
    every refresh_every steps.
    """

    def __init__(
        self,
        strategy: str,
        band_low: float = 0.2,
        band_high: float = 0.6,
        topk_fraction: float = 0.1,
        refresh_every: int = 500,
    ) -> None:
        if strategy not in ("band", "topk"):
            raise ValueError(f"Unknown mining strategy '{strategy}'. Use 'band' or 'topk'.")
        self.strategy = strategy
        self.band_low = band_low
        self.band_high = band_high
        self.topk_fraction = topk_fraction
        self.refresh_every = refresh_every

        self._cached_a: Tensor | None = None
        self._cached_b: Tensor | None = None
        self._last_refresh_step: int = -1

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
        global_step: int,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Return (pair_indices_A, pair_indices_B, stats).

        Mining is skipped (returns cached pair set) if global_step % refresh_every != 0
        and a cached pair set exists. Stats["refreshed"] is True when recomputed.
        """
        needs_refresh = (
            self._cached_a is None
            or global_step % self.refresh_every == 0
        )

        if not needs_refresh:
            assert self._cached_a is not None
            assert self._cached_b is not None
            stats: dict[str, Any] = {
                "n_pairs": self._cached_a.shape[0],
                "score_mean": float("nan"),
                "score_std": float("nan"),
                "refreshed": False,
            }
            return self._cached_a, self._cached_b, stats

        # Reproducible randomness at refresh time
        gen = torch.Generator()
        gen.manual_seed(global_step)

        idx_a, idx_b, scores = self._build_candidate_pairs(embeddings, labels, gen)

        if self.strategy == "band":
            mask = (scores >= self.band_low) & (scores <= self.band_high)
            sel_a = idx_a[mask]
            sel_b = idx_b[mask]
        else:  # topk
            # Higher loss = more confused = harder; loss = -log p_genuine ≈ 1 - score for genuines
            # Use distance from the ideal: genuines should be near 1, impostors near -1
            is_same = labels[idx_a] == labels[idx_b]
            difficulty = torch.where(is_same, 1.0 - scores, scores + 1.0)
            k = max(1, int(self.topk_fraction * scores.shape[0]))
            _, top_idx = difficulty.topk(k)
            sel_a = idx_a[top_idx]
            sel_b = idx_b[top_idx]
            scores = scores[top_idx]

        if sel_a.shape[0] == 0:
            # Fallback: return all pairs if band/topk produces empty set
            sel_a = idx_a
            sel_b = idx_b

        self._cached_a = sel_a
        self._cached_b = sel_b
        self._last_refresh_step = global_step

        stats = {
            "n_pairs": sel_a.shape[0],
            "score_mean": float(scores.mean().item()) if scores.shape[0] > 0 else float("nan"),
            "score_std": float(scores.std().item()) if scores.shape[0] > 1 else 0.0,
            "refreshed": True,
        }
        return sel_a, sel_b, stats

    def _build_candidate_pairs(
        self,
        embeddings: Tensor,
        labels: Tensor,
        gen: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build a random sample of candidate pairs and compute their cosine scores."""
        N = embeddings.shape[0]
        # Sample up to 50k pairs to keep memory manageable
        max_pairs = min(50_000, N * (N - 1) // 2)
        n_sample = min(max_pairs, N * 10)

        idx_a = torch.randint(0, N, (n_sample,), generator=gen)
        idx_b = torch.randint(0, N, (n_sample,), generator=gen)
        # Remove self-pairs
        different = idx_a != idx_b
        idx_a = idx_a[different]
        idx_b = idx_b[different]

        emb_a = F.normalize(embeddings[idx_a], dim=1)
        emb_b = F.normalize(embeddings[idx_b], dim=1)
        scores = (emb_a * emb_b).sum(dim=1)
        return idx_a, idx_b, scores

    def build_hard_pair_dataset(
        self,
        pair_indices_a: Tensor,
        pair_indices_b: Tensor,
        source_dataset: Dataset,
    ) -> "HardPairDataset":
        """Wrap indexed source pairs into a Dataset for the DataLoader."""
        return HardPairDataset(pair_indices_a, pair_indices_b, source_dataset)


class HardPairDataset(Dataset):
    """Dataset of (image_a, image_b, is_same_label) pairs from mined indices."""

    def __init__(
        self,
        pair_indices_a: Tensor,
        pair_indices_b: Tensor,
        source_dataset: Dataset,
    ) -> None:
        self.idx_a = pair_indices_a.tolist()
        self.idx_b = pair_indices_b.tolist()
        self.source = source_dataset

    def __len__(self) -> int:
        return len(self.idx_a)

    def __getitem__(self, i: int) -> tuple[Any, Any, int, int, int]:
        img_a, label_a = self.source[self.idx_a[i]]
        img_b, label_b = self.source[self.idx_b[i]]
        is_same = int(label_a == label_b)
        return img_a, img_b, label_a, label_b, is_same
