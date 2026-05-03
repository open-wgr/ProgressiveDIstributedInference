"""CIFAR-100 adaptor for boosting smoke tests.

Maps the CIFAR-100 superclass/subclass hierarchy to the PPI boosting interface.
P0 is trained on superclass prediction (20 classes), P1/P2 are boosted on the
hard pairs from the previous ensemble — same-subclass verification pairs that
the ensemble fails on.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

# CIFAR-100 superclass → list of subclass indices
# Source: https://www.cs.toronto.edu/~kriz/cifar.html
_SUPERCLASS_MAP: list[list[int]] = [
    [4, 30, 55, 72, 95],   # aquatic mammals
    [1, 32, 67, 73, 91],   # fish
    [54, 62, 70, 82, 92],  # flowers
    [9, 10, 16, 28, 61],   # food containers
    [0, 51, 53, 57, 83],   # fruit and vegetables
    [22, 39, 40, 86, 87],  # household electrical devices
    [5, 20, 25, 84, 94],   # household furniture
    [6, 7, 14, 18, 24],    # insects
    [3, 42, 43, 88, 97],   # large carnivores
    [12, 17, 37, 68, 76],  # large man-made outdoor things
    [23, 33, 49, 60, 71],  # large natural outdoor scenes
    [15, 19, 21, 31, 38],  # large omnivores and herbivores
    [34, 63, 64, 66, 75],  # medium-sized mammals
    [26, 45, 77, 79, 99],  # non-insect invertebrates
    [2, 11, 35, 46, 98],   # people
    [27, 29, 44, 78, 93],  # reptiles
    [36, 50, 65, 74, 80],  # small mammals
    [47, 52, 56, 59, 96],  # trees
    [8, 13, 48, 58, 90],   # vehicles 1
    [41, 69, 81, 85, 89],  # vehicles 2
]

# Invert: subclass_idx → superclass_idx
_SUBCLASS_TO_SUPERCLASS: dict[int, int] = {}
for _super_idx, _sub_list in enumerate(_SUPERCLASS_MAP):
    for _sub in _sub_list:
        _SUBCLASS_TO_SUPERCLASS[_sub] = _super_idx

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CIFAR100BoostingAdaptor:
    """Adapts CIFAR-100 superclass/subclass hierarchy to the PPI boosting interface.

    Partition assignment:
      P0: trained on superclass prediction (20 classes)
      P1: boosted on pairs where P0 fails at subclass level (100 classes)
      P2: boosted on pairs where P0+P1 fail

    Verification framing: given two images, are they the same subclass?
    This matches the LFW same-person/different-person structure.
    """

    def __init__(self, root: str, input_size: int = 32) -> None:
        self.root = root
        self.input_size = input_size

        train_transform = transforms.Compose([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

        self._train_raw = CIFAR100(root=root, train=True, download=True, transform=train_transform)
        self._val_raw = CIFAR100(root=root, train=False, download=True, transform=val_transform)

        self._train_superclass = _SuperclassDataset(self._train_raw)
        self._train_subclass = self._train_raw  # full 100-class labels for P1/P2

    def get_train_dataset(self, phase: int = 0) -> Dataset:
        """Return training dataset appropriate for the given phase.

        Phase 0 → 20 superclass labels.
        Phase 1+ → 100 subclass labels (hard-pair mining handles the hard-case selection).
        """
        if phase == 0:
            return self._train_superclass
        return self._train_subclass

    def get_val_pairs(
        self,
        n_pairs: int = 10_000,
        seed: int = 42,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (images_a, images_b, is_same_subclass) for verification eval."""
        rng = np.random.RandomState(seed)
        targets = np.array(self._val_raw.targets)
        indices = np.arange(len(targets))

        pairs_a, pairs_b, labels = [], [], []
        half = n_pairs // 2

        # Genuine pairs (same subclass)
        added = 0
        while added < half:
            cls = rng.randint(0, 100)
            cls_idx = indices[targets == cls]
            if len(cls_idx) < 2:
                continue
            i, j = rng.choice(cls_idx, size=2, replace=False)
            pairs_a.append(i)
            pairs_b.append(j)
            labels.append(1)
            added += 1

        # Impostor pairs (different subclass)
        added = 0
        while added < half:
            cls_a = rng.randint(0, 100)
            cls_b = rng.randint(0, 100)
            if cls_a == cls_b:
                continue
            idx_a = rng.choice(indices[targets == cls_a])
            idx_b = rng.choice(indices[targets == cls_b])
            pairs_a.append(int(idx_a))
            pairs_b.append(int(idx_b))
            labels.append(0)
            added += 1

        def _collect(idxs: list[int]) -> Tensor:
            imgs = [self._val_raw[i][0] for i in idxs]
            return torch.stack(imgs)

        return _collect(pairs_a), _collect(pairs_b), torch.tensor(labels, dtype=torch.long)

    def superclass_label(self, subclass_label: int) -> int:
        return _SUBCLASS_TO_SUPERCLASS[subclass_label]

    @property
    def num_classes_phase0(self) -> int:
        return 20

    @property
    def num_classes_phase1plus(self) -> int:
        return 100


class _SuperclassDataset(Dataset):
    """Wraps a CIFAR-100 dataset, replacing fine labels with coarse superclass labels."""

    def __init__(self, base: Dataset) -> None:
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img, fine_label = self._base[idx]
        return img, _SUBCLASS_TO_SUPERCLASS[fine_label]

    @property
    def num_classes(self) -> int:
        return 20
