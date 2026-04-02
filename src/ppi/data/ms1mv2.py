"""MS1MV2 face dataset (ImageFolder layout)."""

from __future__ import annotations

from ppi.data.casia import FaceDataset


class MS1MV2(FaceDataset):
    """MS1MV2 dataset (ImageFolder layout)."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        input_size: int = 112,
    ) -> None:
        super().__init__(root=root, train=train, input_size=input_size)
