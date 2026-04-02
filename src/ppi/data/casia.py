"""Face dataset wrappers using ImageFolder layout."""

from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset
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
