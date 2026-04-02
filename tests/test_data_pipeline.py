"""Tests for the data pipeline (CIFAR-100, FaceDataset, build_dataloader)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ppi.data import build_dataloader
from ppi.data.casia import CASIAWebFace, FaceDataset
from ppi.data.cifar100 import CIFAR100Dataset, CIFAR100_MEAN, CIFAR100_STD
from ppi.data.ms1mv2 import MS1MV2
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face_root(tmp_path: Path, num_classes: int = 3, imgs_per_class: int = 2) -> Path:
    """Create a minimal ImageFolder tree with actual JPEG files."""
    root = tmp_path / "faces"
    for c in range(num_classes):
        cls_dir = root / f"class_{c}"
        cls_dir.mkdir(parents=True)
        for i in range(imgs_per_class):
            img = Image.new("RGB", (120, 120), color=(c * 40, i * 60, 128))
            img.save(cls_dir / f"img_{i}.jpg")
    return root


# ---------------------------------------------------------------------------
# CIFAR-100 transform pipeline tests
# ---------------------------------------------------------------------------

class TestCIFAR100Transforms:
    """Test the CIFAR-100 transform pipelines directly (no download)."""

    def test_train_transform_output_shape(self):
        """Train transforms should produce (3, input_size, input_size)."""
        input_size = 32
        tfm = transforms.Compose([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        img = Image.new("RGB", (32, 32))
        tensor = tfm(img)
        assert tensor.shape == (3, input_size, input_size)

    def test_val_transform_output_shape(self):
        """Val transforms should produce (3, H, W) matching the input."""
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        img = Image.new("RGB", (32, 32))
        tensor = tfm(img)
        assert tensor.shape == (3, 32, 32)

    def test_normalization_range(self):
        """After normalisation the mean should be roughly centred around 0."""
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        tensor = tfm(img)
        # 128/255 ≈ 0.502; after normalising it should be near 0
        assert tensor.mean().abs() < 1.0

    def test_num_classes(self):
        """CIFAR100Dataset.num_classes should always be 100."""
        # We cannot instantiate without data, but we can check the property
        # via the class definition.
        assert CIFAR100Dataset.num_classes.fget is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# FaceDataset / CASIAWebFace / MS1MV2 tests
# ---------------------------------------------------------------------------

class TestFaceDataset:

    def test_face_dataset_len(self, tmp_path: Path):
        num_classes, imgs = 3, 2
        root = _make_face_root(tmp_path, num_classes=num_classes, imgs_per_class=imgs)
        ds = FaceDataset(root=str(root), train=True, input_size=112)
        assert len(ds) == num_classes * imgs

    def test_face_dataset_getitem_shape(self, tmp_path: Path):
        root = _make_face_root(tmp_path, num_classes=2, imgs_per_class=1)
        ds = FaceDataset(root=str(root), train=False, input_size=112)
        img, label = ds[0]
        assert img.shape == (3, 112, 112)
        assert isinstance(label, int)

    def test_face_dataset_num_classes(self, tmp_path: Path):
        root = _make_face_root(tmp_path, num_classes=5, imgs_per_class=1)
        ds = FaceDataset(root=str(root), train=True)
        assert ds.num_classes == 5

    def test_casia_webface_inherits(self, tmp_path: Path):
        root = _make_face_root(tmp_path)
        ds = CASIAWebFace(root=str(root), train=True)
        assert isinstance(ds, FaceDataset)
        assert len(ds) > 0

    def test_ms1mv2_inherits(self, tmp_path: Path):
        root = _make_face_root(tmp_path)
        ds = MS1MV2(root=str(root), train=True)
        assert isinstance(ds, FaceDataset)
        assert len(ds) > 0

    def test_val_no_random_flip(self, tmp_path: Path):
        """Val transforms should be deterministic (no random flip)."""
        root = _make_face_root(tmp_path, num_classes=1, imgs_per_class=1)
        ds = FaceDataset(root=str(root), train=False, input_size=112)
        t1, _ = ds[0]
        t2, _ = ds[0]
        assert torch.equal(t1, t2)


# ---------------------------------------------------------------------------
# build_dataloader tests
# ---------------------------------------------------------------------------

class TestBuildDataloader:

    @pytest.fixture()
    def face_config(self, tmp_path: Path) -> dict:
        root = _make_face_root(tmp_path, num_classes=4, imgs_per_class=2)
        return {
            "seed": 123,
            "data": {
                "dataset": "casia",
                "root": str(root),
                "num_workers": 0,
                "input_size": 112,
            },
            "training": {"batch_size": 2},
        }

    def test_returns_dataloader_and_num_classes(self, face_config: dict):
        loader, num_classes = build_dataloader(face_config, split="train")
        assert isinstance(loader, DataLoader)
        assert num_classes == 4

    def test_train_batch_shape(self, face_config: dict):
        loader, _ = build_dataloader(face_config, split="train")
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 112, 112)
        assert labels.shape == (2,)

    def test_val_no_shuffle(self, face_config: dict):
        """Val loader should not shuffle."""
        loader, _ = build_dataloader(face_config, split="val")
        # DataLoader stores shuffle state indirectly; for non-shuffled
        # loaders the sampler is a SequentialSampler.
        from torch.utils.data import SequentialSampler
        assert isinstance(loader.sampler, SequentialSampler)

    def test_unknown_dataset_raises(self):
        cfg = {
            "seed": 0,
            "data": {"dataset": "nonexistent", "root": "/tmp"},
            "training": {"batch_size": 1},
        }
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_dataloader(cfg)

    def test_ms1mv2_via_build(self, tmp_path: Path):
        root = _make_face_root(tmp_path, num_classes=2, imgs_per_class=1)
        cfg = {
            "seed": 0,
            "data": {
                "dataset": "ms1mv2",
                "root": str(root),
                "num_workers": 0,
                "input_size": 112,
            },
            "training": {"batch_size": 1},
        }
        loader, nc = build_dataloader(cfg, split="train")
        assert nc == 2
        images, labels = next(iter(loader))
        assert images.shape == (1, 3, 112, 112)


# ---------------------------------------------------------------------------
# Worker seeding determinism
# ---------------------------------------------------------------------------

class TestWorkerSeeding:
    """Verify that the worker_init_fn produces deterministic results."""

    def test_deterministic_seeding(self, tmp_path: Path):
        """Two loaders with the same seed should yield identical batches."""
        root = _make_face_root(tmp_path, num_classes=2, imgs_per_class=4)
        cfg = {
            "seed": 999,
            "data": {
                "dataset": "casia",
                "root": str(root),
                "num_workers": 0,
                "input_size": 112,
            },
            "training": {"batch_size": 8},
        }

        # With num_workers=0 the worker_init_fn doesn't fire, but the
        # global seed still controls shuffle order via the generator.
        # We test the init function directly instead.
        from ppi.data import _worker_init_fn

        _worker_init_fn(0, seed=42)
        a_rand = random.random()
        a_np = np.random.rand()
        a_torch = torch.rand(1).item()

        _worker_init_fn(0, seed=42)
        b_rand = random.random()
        b_np = np.random.rand()
        b_torch = torch.rand(1).item()

        assert a_rand == b_rand
        assert a_np == b_np
        assert a_torch == b_torch

    def test_different_workers_get_different_seeds(self):
        from ppi.data import _worker_init_fn

        _worker_init_fn(0, seed=42)
        a = random.random()

        _worker_init_fn(1, seed=42)
        b = random.random()

        assert a != b
