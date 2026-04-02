"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

CIFAR100_ROOT = Path(__file__).resolve().parent.parent / "data" / "cifar100"


def pytest_addoption(parser):
    parser.addoption(
        "--run-cifar100",
        action="store_true",
        default=False,
        help="Download CIFAR-100 (~169 MB) if needed and run real-data tests",
    )


def pytest_configure(config):
    if config.getoption("--run-cifar100"):
        # Trigger download so the dataset is on disk for all tests
        from torchvision.datasets import CIFAR100

        CIFAR100(root=str(CIFAR100_ROOT), train=True, download=True)
        CIFAR100(root=str(CIFAR100_ROOT), train=False, download=True)


@pytest.fixture
def tiny_config() -> dict:
    """Minimal valid config for unit tests."""
    return {
        "seed": 42,
        "backbone": {"name": "resnet18", "pretrained": False},
        "partitions": {
            "num_partitions": 3,
            "K": 8,
            "dropout": {
                "enabled": True,
                "distribution": [0.40, 0.30, 0.20, 0.10],
            },
        },
        "arcface": {"s": 64, "m": 0.5, "num_classes": 10},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "optimizer": {
                "type": "sgd",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 5.0e-4,
            },
            "scheduler": {"type": "cosine", "warmup_epochs": 1},
            "val_interval": 1,
            "checkpoint_interval": 1,
        },
        "data": {
            "dataset": "cifar100",
            "root": "/tmp/test_data",
            "num_workers": 0,
            "input_size": 32,
        },
        "logging": {
            "output_dir": "/tmp/test_runs",
            "tensorboard": False,
            "wandb": False,
        },
    }


@pytest.fixture
def dummy_batch():
    """Small batch of dummy face images and labels."""
    images = torch.randn(4, 3, 112, 112)
    labels = torch.randint(0, 100, (4,))
    return images, labels


@pytest.fixture
def dummy_batch_cifar():
    """Small batch of dummy CIFAR-sized images and labels."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 100, (4,))
    return images, labels
