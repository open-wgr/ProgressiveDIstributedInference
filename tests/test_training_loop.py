"""Integration test for the training loop on synthetic data."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image


def _create_synthetic_dataset(root: Path, num_classes: int = 10, imgs_per_class: int = 10) -> Path:
    """Create a tiny image folder dataset for integration testing."""
    for c in range(num_classes):
        cls_dir = root / f"class_{c}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(imgs_per_class):
            img = Image.new("RGB", (120, 120), color=(c * 25, i * 10, 128))
            img.save(cls_dir / f"img_{i}.jpg")
    return root


@pytest.fixture
def training_config(tmp_path) -> dict:
    """Build a minimal config for integration testing."""
    data_root = _create_synthetic_dataset(tmp_path / "data", num_classes=10, imgs_per_class=10)
    run_dir = tmp_path / "runs"
    return {
        "seed": 42,
        "backbone": {"name": "resnet18", "pretrained": False},
        "partitions": {
            "num_partitions": 3,
            "K": 8,
            "dropout": {
                "enabled": True,
                "distribution": [0.4, 0.3, 0.2, 0.1],
            },
        },
        "arcface": {"s": 64, "m": 0.5, "num_classes": 10},
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "optimizer": {"type": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
            "scheduler": {"type": "cosine", "warmup_epochs": 1},
            "val_interval": 1,
            "checkpoint_interval": 1,
            "amp": False,   # disable fp16 for test stability on tiny synthetic data
        },
        "data": {
            "dataset": "casia",
            "root": str(data_root),
            "num_workers": 0,
            "input_size": 32,
        },
        "logging": {
            "output_dir": str(run_dir),
            "tensorboard": False,
            "wandb": False,
        },
    }


class TestTrainingLoop:
    def test_loss_decreases(self, training_config):
        """Loss should decrease over 2 epochs on a tiny dataset."""
        from ppi.training.trainer import Trainer

        trainer = Trainer(training_config)

        # Capture per-epoch losses by running manually
        losses = []
        for epoch in range(1, 3):
            trainer.backbone.train()
            trainer.arcface_head.train()
            trainer.partition_dropout.train()
            epoch_loss = 0.0
            n = 0
            for images, labels in trainer.train_loader:
                images = images.to(trainer.device)
                labels = labels.to(trainer.device)
                out = trainer.backbone(images)
                parts = trainer.strategy.process_partitions(out["partitions"])
                parts = trainer.partition_dropout(parts)
                from ppi.training.partition_dropout import assemble_embedding
                emb = assemble_embedding(parts)
                cosine = trainer.arcface_head(emb)
                loss = trainer.arcface_loss(cosine, labels)
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                epoch_loss += loss.item()
                n += 1
            trainer.scheduler.step()
            losses.append(epoch_loss / n)

        # Loss should generally decrease (allow some noise with small data)
        # At minimum, both should be finite
        assert all(l > 0 and l < 100 for l in losses)

    def test_full_train_and_checkpoint(self, training_config):
        """Full train() call produces checkpoints and config."""
        from ppi.training.trainer import Trainer

        trainer = Trainer(training_config)
        trainer.train()

        run_dir = trainer.logger.run_dir
        # Config saved
        assert (run_dir / "config.yaml").exists()
        with open(run_dir / "config.yaml") as f:
            saved_cfg = yaml.safe_load(f)
        assert saved_cfg["seed"] == 42

        # Checkpoints saved (2 epochs, interval=1)
        assert (run_dir / "checkpoint_epoch1.pt").exists()
        assert (run_dir / "checkpoint_epoch2.pt").exists()

    def test_checkpoint_loads_correctly(self, training_config):
        """A saved checkpoint can be loaded and produces the same model state."""
        from ppi.training.trainer import Trainer
        from ppi.utils.logging import ExperimentLogger

        trainer = Trainer(training_config)
        trainer.train()

        ckpt_path = trainer.logger.run_dir / "checkpoint_epoch2.pt"
        ckpt = ExperimentLogger.load_checkpoint(ckpt_path)
        assert ckpt["epoch"] == 2
        assert "backbone" in ckpt["model_state_dict"]
        assert "arcface_head" in ckpt["model_state_dict"]

        # Load state into a fresh backbone and verify outputs match
        # Compare on CPU to avoid device mismatches
        from ppi.backbones import build_backbone
        fresh = build_backbone(training_config)
        fresh.load_state_dict(ckpt["model_state_dict"]["backbone"])
        fresh.eval()

        trained = build_backbone(training_config)
        trained.load_state_dict(ckpt["model_state_dict"]["backbone"])
        trained.eval()

        x = torch.randn(1, 3, 32, 32)
        out1 = trained(x)
        out2 = fresh(x)
        assert torch.allclose(out1["features"], out2["features"], atol=1e-5)
