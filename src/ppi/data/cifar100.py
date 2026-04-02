"""CIFAR-100 dataset wrapper with standard augmentation."""

from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CIFAR100Dataset(Dataset):
    """Thin wrapper around ``torchvision.datasets.CIFAR100`` with standard
    train/val augmentation pipelines."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        input_size: int = 32,
    ) -> None:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])

        self._dataset = CIFAR100(
            root=root,
            train=train,
            transform=transform,
            download=True,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._dataset[index]

    @property
    def num_classes(self) -> int:
        return 100
