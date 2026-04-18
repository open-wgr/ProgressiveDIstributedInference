"""Tests for backbone, partition heads, and ArcFace head."""

from __future__ import annotations

import pytest
import torch

from ppi.backbones import build_backbone
from ppi.backbones.resnet import PartitionedResNet
from ppi.heads.arcface import ArcFaceHead
from ppi.heads.partition_head import PartitionHead


class TestPartitionHead:
    def test_output_shape(self):
        head = PartitionHead(in_features=512, out_features=64)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 64)

    def test_bn_fc_bn_structure(self):
        head = PartitionHead(in_features=256, out_features=32)
        assert isinstance(head.bn1, torch.nn.BatchNorm1d)
        assert isinstance(head.fc, torch.nn.Linear)
        assert isinstance(head.bn2, torch.nn.BatchNorm1d)


class TestArcFaceHead:
    def test_output_shape(self):
        head = ArcFaceHead(in_features=384, num_classes=100)
        x = torch.randn(4, 384)
        out = head(x)
        assert out.shape == (4, 100)

    def test_output_is_cosine_similarity(self):
        head = ArcFaceHead(in_features=64, num_classes=10)
        # ArcFaceHead expects pre-normalised input (from assemble_embedding)
        x = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
        out = head(x)
        # Cosine similarities should be in [-1, 1]
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_zero_input(self):
        head = ArcFaceHead(in_features=64, num_classes=10)
        x = torch.zeros(2, 64)
        out = head(x)
        assert torch.isfinite(out).all()


class TestPartitionedResNet:
    def test_resnet50_112(self):
        model = PartitionedResNet("resnet50", num_partitions=3, K=128, input_size=112)
        x = torch.randn(2, 3, 112, 112)
        out = model(x)
        assert out["features"].shape == (2, 2048)
        assert len(out["partitions"]) == 3
        for p in out["partitions"]:
            assert p.shape == (2, 128)

    def test_resnet18_32(self):
        model = PartitionedResNet("resnet18", num_partitions=3, K=64, input_size=32)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out["features"].shape == (2, 512)
        assert len(out["partitions"]) == 3
        for p in out["partitions"]:
            assert p.shape == (2, 64)

    def test_backbone_dim_property(self):
        model = PartitionedResNet("resnet18", num_partitions=3, K=64)
        assert model.backbone_dim == 512
        model50 = PartitionedResNet("resnet50", num_partitions=3, K=128)
        assert model50.backbone_dim == 2048

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            PartitionedResNet("resnet101", num_partitions=3, K=128)

    def test_param_count_resnet18(self):
        model = PartitionedResNet("resnet18", num_partitions=3, K=64)
        total = sum(p.numel() for p in model.parameters())
        # ResNet-18 ~11M params + partition heads
        assert 10_000_000 < total < 15_000_000

    def test_param_count_resnet50(self):
        model = PartitionedResNet("resnet50", num_partitions=3, K=128)
        total = sum(p.numel() for p in model.parameters())
        # ResNet-50 ~23M params + partition heads
        assert 20_000_000 < total < 30_000_000


class TestBuildBackbone:
    def test_build_resnet18(self, tiny_config):
        model = build_backbone(tiny_config)
        assert isinstance(model, PartitionedResNet)
        assert model.backbone_dim == 512

    def test_build_unknown_raises(self, tiny_config):
        tiny_config["backbone"]["name"] = "unknown"
        with pytest.raises(ValueError):
            build_backbone(tiny_config)
