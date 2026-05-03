"""Face dataset wrappers using ImageFolder layout."""

from __future__ import annotations

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class FaceDataset(Dataset):
    """Base face-recognition dataset backed by an ``ImageFolder`` directory
    tree (one sub-directory per identity)."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        input_size: int = 112,
    ) -> None:
        if train:
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])

        self._dataset = ImageFolder(root=root, transform=transform)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._dataset[index]

    @property
    def num_classes(self) -> int:
        return len(self._dataset.classes)


class CASIAWebFace(FaceDataset):
    """CASIA-WebFace dataset (ImageFolder layout)."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        input_size: int = 112,
    ) -> None:
        super().__init__(root=root, train=train, input_size=input_size)


class CASIASubset(Dataset):
    """Identity-stratified subset of CASIA-WebFace for gating runs.

    Draws a fixed random-seed sample of num_identities identities, keeping
    all images for each selected identity. This mirrors the full dataset
    structure (one label per identity) while reducing compute by ~80% for
    a 2k/10k identity subset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        input_size: int = 112,
        num_identities: int = 2000,
        seed: int = 42,
    ) -> None:
        self._full = CASIAWebFace(root=root, train=train, input_size=input_size)
        full_dataset = self._full._dataset

        all_targets = np.array(full_dataset.targets)
        all_classes = np.unique(all_targets)
        total_classes = len(all_classes)

        if num_identities >= total_classes:
            self._subset = self._full
            self._num_classes = total_classes
            return

        rng = np.random.RandomState(seed)
        chosen_classes = rng.choice(all_classes, size=num_identities, replace=False)
        chosen_set = set(chosen_classes.tolist())

        indices = [i for i, t in enumerate(all_targets) if t in chosen_set]

        # Remap labels to 0..num_identities-1
        label_map = {old: new for new, old in enumerate(sorted(chosen_set))}
        self._indices = indices
        self._labels = [label_map[int(all_targets[i])] for i in indices]
        self._full_dataset = full_dataset
        self._num_classes = num_identities
        self._subset = None

    def __len__(self) -> int:
        if self._subset is not None:
            return len(self._subset)
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        if self._subset is not None:
            return self._subset[idx]
        img, _ = self._full_dataset[self._indices[idx]]
        return img, self._labels[idx]

    @property
    def num_classes(self) -> int:
        return self._num_classes
