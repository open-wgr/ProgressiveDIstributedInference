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
        genuine_fraction: float = 0.5,
    ) -> None:
        if strategy not in ("band", "topk"):
            raise ValueError(f"Unknown mining strategy '{strategy}'. Use 'band' or 'topk'.")
        self.strategy = strategy
        self.band_low = band_low
        self.band_high = band_high
        self.topk_fraction = topk_fraction
        self.refresh_every = refresh_every
        # Genuine and impostor difficulty are not on a comparable scale
        # (genuine ≈ 1-score ≈ 0, impostor ≈ score+1 ≈ 1), so a single global
        # ranking selects impostors exclusively. Both candidate generation and
        # selection are therefore stratified by pair type, with this target
        # share of genuine pairs. Training on impostors alone gives the loss
        # nothing to pull together and collapses the embedding to a uniform
        # spread on the sphere (= chance verification accuracy).
        self.genuine_fraction = genuine_fraction

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

        is_same = labels[idx_a] == labels[idx_b]

        if self.strategy == "band":
            # Rank within each pair type: a single band on raw score admits
            # almost only impostors, since genuine and impostor scores occupy
            # different parts of the range.
            in_band = (scores >= self.band_low) & (scores <= self.band_high)
            sel_mask = torch.zeros_like(in_band)
            for want_same in (True, False):
                type_mask = in_band & (is_same == want_same)
                sel_mask |= type_mask
            # Guarantee genuine representation even if the band excludes them.
            if not (sel_mask & is_same).any() and is_same.any():
                # Fall back to the hardest genuine pairs (lowest score).
                gen_scores = torch.where(is_same, scores, torch.full_like(scores, 1e9))
                n_add = max(1, int((sel_mask & ~is_same).sum().item() * self.genuine_fraction))
                n_add = min(n_add, int(is_same.sum().item()))
                add_idx = gen_scores.topk(n_add, largest=False).indices
                sel_mask[add_idx] = True
            sel_a, sel_b = idx_a[sel_mask], idx_b[sel_mask]
            scores = scores[sel_mask]
        else:  # topk
            # Difficulty = distance from the ideal score (genuine -> 1,
            # impostor -> -1). These scales are NOT comparable across types,
            # so select the hardest of each type separately rather than
            # ranking them against each other.
            difficulty = torch.where(is_same, 1.0 - scores, scores + 1.0)
            k_total = max(2, int(self.topk_fraction * scores.shape[0]))
            k_gen = int(k_total * self.genuine_fraction)
            k_imp = k_total - k_gen

            chosen: list[Tensor] = []
            for want_same, k_want in ((True, k_gen), (False, k_imp)):
                pool = torch.nonzero(is_same == want_same).squeeze(1)
                if pool.numel() == 0 or k_want <= 0:
                    continue
                k_eff = min(k_want, pool.numel())
                top = difficulty[pool].topk(k_eff).indices
                chosen.append(pool[top])

            if chosen:
                sel = torch.cat(chosen)
                sel_a, sel_b = idx_a[sel], idx_b[sel]
                scores = scores[sel]
            else:
                sel_a, sel_b = idx_a, idx_b

        if sel_a.shape[0] == 0:
            # Fallback: return all pairs if band/topk produces empty set
            sel_a = idx_a
            sel_b = idx_b

        self._cached_a = sel_a
        self._cached_b = sel_b
        self._last_refresh_step = global_step

        sel_genuine = (labels[sel_a] == labels[sel_b])
        genuine_frac = (
            float(sel_genuine.float().mean().item()) if sel_a.shape[0] > 0 else float("nan")
        )
        stats = {
            "n_pairs": sel_a.shape[0],
            "score_mean": float(scores.mean().item()) if scores.shape[0] > 0 else float("nan"),
            "score_std": float(scores.std().item()) if scores.shape[0] > 1 else 0.0,
            # A mined set with no genuine pairs gives the loss nothing to pull
            # together; the embedding collapses to a uniform spread on the
            # sphere and verification lands at chance. Always log this.
            "genuine_fraction": genuine_frac,
            "refreshed": True,
        }
        return sel_a, sel_b, stats

    def _build_candidate_pairs(
        self,
        embeddings: Tensor,
        labels: Tensor,
        gen: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build a class-balanced candidate pair set and score it.

        Uniformly random pairs are ~1/num_classes genuine (1% on CIFAR-100,
        0.01% on CASIA), so genuine pairs must be sampled explicitly rather
        than hoped for.
        """
        N = embeddings.shape[0]
        n_sample = min(50_000, max(N * 10, 2))
        n_genuine = int(n_sample * self.genuine_fraction)
        n_impostor = n_sample - n_genuine

        # --- Impostor candidates: random pairs, rejecting same-class ---
        ia = torch.randint(0, N, (n_impostor,), generator=gen)
        ib = torch.randint(0, N, (n_impostor,), generator=gen)
        keep = (ia != ib) & (labels[ia] != labels[ib])
        imp_a, imp_b = ia[keep], ib[keep]

        # --- Genuine candidates: pick a class with >=2 members, then two of them ---
        gen_a, gen_b = self._sample_genuine_pairs(labels, n_genuine, gen)

        idx_a = torch.cat([gen_a, imp_a])
        idx_b = torch.cat([gen_b, imp_b])

        emb_a = F.normalize(embeddings[idx_a], dim=1)
        emb_b = F.normalize(embeddings[idx_b], dim=1)
        scores = (emb_a * emb_b).sum(dim=1)
        return idx_a, idx_b, scores

    @staticmethod
    def _sample_genuine_pairs(
        labels: Tensor,
        n: int,
        gen: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Sample n distinct same-class index pairs (vectorised)."""
        if n <= 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty

        order = torch.argsort(labels)
        counts = torch.unique_consecutive(labels[order], return_counts=True)[1]
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])

        eligible = torch.nonzero(counts >= 2).squeeze(1)
        if eligible.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty

        pick = eligible[torch.randint(0, eligible.numel(), (n,), generator=gen)]
        cnt = counts[pick]
        off = offsets[pick]
        r1 = (torch.rand(n, generator=gen) * cnt).long().clamp_max_(cnt - 1)
        # Draw the partner from the remaining cnt-1 slots, then shift past r1
        # so the two indices are always distinct.
        r2 = (torch.rand(n, generator=gen) * (cnt - 1)).long().clamp_max_(cnt - 2)
        r2 = r2 + (r2 >= r1).long()
        return order[off + r1], order[off + r2]

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

    def __getitem__(self, i: int) -> tuple[Any, Any, int, int, int, int, int]:
        idx_a = self.idx_a[i]
        idx_b = self.idx_b[i]
        img_a, label_a = self.source[idx_a]
        img_b, label_b = self.source[idx_b]
        is_same = int(label_a == label_b)
        return img_a, img_b, label_a, label_b, is_same, idx_a, idx_b
