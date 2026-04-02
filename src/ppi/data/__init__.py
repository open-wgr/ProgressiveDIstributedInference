"""Dataset loaders and data utilities."""

from __future__ import annotations

import random
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ppi.data.casia import CASIAWebFace, FaceDataset
from ppi.data.cifar100 import CIFAR100Dataset
from ppi.data.ms1mv2 import MS1MV2

__all__ = [
    "CIFAR100Dataset",
    "CASIAWebFace",
    "FaceDataset",
    "MS1MV2",
    "build_dataloader",
]

_DATASET_REGISTRY: dict[str, type[Dataset]] = {
    "cifar100": CIFAR100Dataset,
    "casia": CASIAWebFace,
    "ms1mv2": MS1MV2,
}


def _worker_init_fn(worker_id: int, seed: int) -> None:
    """Seed every data-loader worker deterministically."""
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloader(
    config: dict[str, Any],
    split: str = "train",
) -> tuple[DataLoader, int]:
    """Build a DataLoader from a configuration dictionary.

    Parameters
    ----------
    config:
        Full experiment config. Must contain at least ``config["data"]``
        with keys ``dataset``, ``root``, and optionally ``num_workers``,
        ``input_size``.  ``config["seed"]`` is used for worker seeding and
        ``config["training"]["batch_size"]`` for the batch size.
    split:
        ``"train"`` or ``"val"``.

    Returns
    -------
    tuple[DataLoader, int]
        The data loader and the number of classes in the dataset.
    """
    data_cfg = config["data"]
    dataset_name: str = data_cfg["dataset"]

    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(_DATASET_REGISTRY)}"
        )

    is_train = split == "train"
    dataset_cls = _DATASET_REGISTRY[dataset_name]

    dataset = dataset_cls(
        root=data_cfg["root"],
        train=is_train,
        input_size=data_cfg.get("input_size", 32 if dataset_name == "cifar100" else 112),
    )

    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)
    seed = config.get("seed", 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=partial(_worker_init_fn, seed=seed),
    )

    num_classes: int = dataset.num_classes  # type: ignore[attr-defined]
    return loader, num_classes
