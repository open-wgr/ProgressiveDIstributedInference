"""Datasets for combiner training over cached partition embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


class CachedPartitionDataset(Dataset):
    """Subset-sampling dataset for combiner training.

    At each __getitem__ call, draws a uniformly-random non-empty subset of
    partitions from all 2^N - 1 possibilities.  Absent partitions are zero-padded;
    a float presence mask is returned alongside the embedding.  This forces the
    combiner to handle every possible subset during training, which is the correct
    inductive bias for a subset-agnostic model.

    Reproducibility: the internal generator is seeded once at construction time.
    The DataLoader should use num_workers=0 (or persistent_workers with a fixed
    seed) to keep draws identical across runs with the same seed.
    """

    def __init__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        num_partitions: int,
        seed: int = 42,
    ) -> None:
        self.embeddings = embeddings   # (N, num_partitions, K), raw backbone output
        self.labels = labels           # (N,), int64
        self.num_partitions = num_partitions
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (masked_embedding, mask, label).

        masked_embedding: (num_partitions * K,) — per-partition L2-normed, zeros for absent
        mask:             (num_partitions,)      — float 0/1
        label:            scalar int
        """
        raw = self.embeddings[idx]   # (num_partitions, K)
        label = self.labels[idx]

        # Draw a non-empty subset uniformly from {1, ..., 2^N - 1}.
        # Each integer encodes a bitmask over partition indices.
        num_subsets = (1 << self.num_partitions) - 1
        subset_bits = int(torch.randint(1, num_subsets + 1, (1,), generator=self._rng).item())

        mask = torch.zeros(self.num_partitions, dtype=torch.float32)
        parts = torch.zeros_like(raw)   # (num_partitions, K)
        for i in range(self.num_partitions):
            if subset_bits & (1 << i):
                mask[i] = 1.0
                parts[i] = F.normalize(raw[i], dim=0, eps=1e-12)

        return parts.reshape(-1), mask, label


class FullTripleDataset(Dataset):
    """Control dataset: always returns the full N-partition triple with no masking.

    Used for robustness control runs comparing against CachedPartitionDataset.
    The comparison isolates whether explicit subset exposure during training is
    necessary for the combiner to extract joint identity signal.
    """

    def __init__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        num_partitions: int,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels
        self.num_partitions = num_partitions

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (full_embedding, all_ones_mask, label)."""
        raw = self.embeddings[idx]   # (num_partitions, K)
        label = self.labels[idx]

        mask = torch.ones(self.num_partitions, dtype=torch.float32)
        parts = torch.stack(
            [F.normalize(raw[i], dim=0, eps=1e-12) for i in range(self.num_partitions)],
            dim=0,
        )
        return parts.reshape(-1), mask, label
