"""Direction 2 boosting tests.

Covers the contracts established by the D2 bug sweep (see
``D2_bugfix_tracker.md``): hard-pair mining, loss numerics, backbone-state
parameter groups, the phase-k trainer, checkpoint format, combiner
strategies, scheduler edge cases, and the class-balanced sampler.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ppi.boosting.backbone_state import set_backbone_state
from ppi.boosting.combination import CosineConcat, ConfidenceWeighted, LearnedCombiner
from ppi.boosting.losses import (
    ArcFaceMargin,
    ArcFaceReweighted,
    ContrastiveLoss,
    SubCenterArcFace,
    TripletLoss,
    build_loss,
)
from ppi.boosting.mining import HardPairDataset, HardPairMiner
from ppi.boosting.trainer import BoostingTrainer
from ppi.training.schedulers import build_scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubLogger:
    """No-op logger compatible with BoostingTrainer's expected interface."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.epochs: list[tuple[dict, int]] = []
        self._wandb_run = None
        self.run_dir = Path("/tmp")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), step))

    def log_epoch(self, metrics: dict, epoch: int) -> None:
        self.epochs.append((dict(metrics), epoch))

    def close(self) -> None:
        pass


class _ToyDataset(Dataset):
    """Tiny synthetic image dataset with a controllable class layout."""

    def __init__(self, n_classes: int = 4, per_class: int = 8, image_size: int = 16) -> None:
        self.n_classes = n_classes
        self.per_class = per_class
        self.image_size = image_size
        self._labels = [c for c in range(n_classes) for _ in range(per_class)]
        # Deterministic per-class noise so embeddings cluster a little
        gen = torch.Generator().manual_seed(0)
        self._images = torch.stack([
            torch.randn(3, image_size, image_size, generator=gen) + lab * 0.05
            for lab in self._labels
        ])

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return self._images[idx], int(self._labels[idx])

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def targets(self) -> list[int]:
        return self._labels


def _toy_config(num_partitions: int = 3, K: int = 8, num_classes: int = 4, loss: str = "triplet") -> dict:
    return {
        "seed": 0,
        "num_partitions": num_partitions,
        "backbone": {"name": "resnet18", "pretrained": False},
        "partitions": {"num_partitions": num_partitions, "K": K},
        "arcface": {"s": 64.0, "m": 0.5, "num_classes": num_classes},
        "boosting": {
            "backbone_state": "partial",
            "frozen_stages": 3,
            "backbone_lr_multiplier": 0.1,
            "mining_strategy": "topk",
            "mining_topk_fraction": 0.5,
            "mining_refresh_every": 1_000_000,  # effectively never within tests
            "loss": loss,
            "triplet_margin": 0.3,
            "triplet_mining": "batch_hard",
            "contrastive_margin": 1.0,
            "easy_loss_weight": 0.3,
            "sub_center_K": 2,
        },
        "training": {
            "epochs_phase0": 1,
            "epochs_per_phase": 1,
            "batch_size": 4,
            "grad_clip": 5.0,
            "optimizer": {"type": "sgd", "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "scheduler": {"type": "cosine", "warmup_epochs": 0},
        },
        "data": {"dataset": "cifar100", "root": "/tmp/_unused", "num_workers": 0, "input_size": 16},
        "logging": {"output_dir": "/tmp/_unused", "tensorboard": False, "wandb": False, "run_name": "test"},
    }


def _build_trainer(loss: str = "triplet", num_partitions: int = 3, K: int = 8, num_classes: int = 4) -> BoostingTrainer:
    cfg = _toy_config(num_partitions=num_partitions, K=K, num_classes=num_classes, loss=loss)
    trainer = BoostingTrainer(cfg, device=torch.device("cpu"), logger=_StubLogger())
    trainer._train_dataset = _ToyDataset(n_classes=num_classes, per_class=4, image_size=cfg["data"]["input_size"])
    trainer.num_classes = num_classes
    return trainer


# ---------------------------------------------------------------------------
# Loss numerics — BUG-7, BUG-11
# ---------------------------------------------------------------------------


class TestLossNumerics:
    def test_contrastive_zero_distance_finite_grad(self):
        loss_fn = ContrastiveLoss(margin=1.0)
        a = torch.randn(4, 8, requires_grad=True)
        b = a.clone().detach().requires_grad_(True)
        loss = loss_fn(a, b, torch.ones(4))
        loss.backward()
        assert torch.isfinite(loss).item()
        assert torch.isfinite(a.grad).all().item()

    def test_triplet_semi_hard_no_positives_returns_finite_zero(self):
        loss_fn = TripletLoss(margin=0.3, mining_strategy="semi_hard")
        emb = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])  # all unique → no positives
        loss = loss_fn(emb, labels)
        assert torch.isfinite(loss).item()
        assert loss.item() == 0.0

    def test_triplet_semi_hard_with_positives_runs(self):
        loss_fn = TripletLoss(margin=0.3, mining_strategy="semi_hard")
        emb = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(emb, labels)
        loss.backward()
        assert torch.isfinite(loss).item()
        assert torch.isfinite(emb.grad).all().item()

    def test_triplet_batch_hard_filters_orphans(self):
        loss_fn = TripletLoss(margin=0.3, mining_strategy="batch_hard")
        emb = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(emb, labels)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Hard-pair mining — BUG-1
# ---------------------------------------------------------------------------


class TestHardPairMiner:
    def test_refresh_then_cache(self):
        miner = HardPairMiner(strategy="topk", topk_fraction=0.5, refresh_every=10)
        emb = torch.randn(32, 16)
        labels = torch.randint(0, 4, (32,))
        a0, b0, stats0 = miner.mine(emb, labels, global_step=0)
        assert stats0["refreshed"] is True
        a1, b1, stats1 = miner.mine(emb, labels, global_step=1)
        assert stats1["refreshed"] is False
        assert torch.equal(a0, a1) and torch.equal(b0, b1)
        # Reaching the refresh boundary forces recomputation.
        a2, b2, stats2 = miner.mine(emb, labels, global_step=10)
        assert stats2["refreshed"] is True

    def test_band_strategy_filters_in_range(self):
        miner = HardPairMiner(strategy="band", band_low=-2.0, band_high=2.0, refresh_every=1)
        emb = torch.randn(16, 8)
        labels = torch.randint(0, 3, (16,))
        a, b, stats = miner.mine(emb, labels, global_step=0)
        assert stats["n_pairs"] > 0
        assert a.shape == b.shape

    def test_hard_pair_dataset_returns_seven_tuple(self):
        ds = _ToyDataset(n_classes=3, per_class=4)
        idx_a = torch.tensor([0, 4, 8])
        idx_b = torch.tensor([1, 5, 9])
        pair_ds = HardPairDataset(idx_a, idx_b, ds)
        item = pair_ds[0]
        assert len(item) == 7
        img_a, img_b, lab_a, lab_b, is_same, pidx_a, pidx_b = item
        assert img_a.shape == img_b.shape
        # Pair (0, 1) — both are class 0 in the toy layout
        assert is_same == 1
        assert pidx_a == 0 and pidx_b == 1


# ---------------------------------------------------------------------------
# Backbone state — BUG-2 contract underpinnings
# ---------------------------------------------------------------------------


class TestBackboneState:
    def _make_backbone(self):
        from ppi.backbones import build_backbone
        cfg = _toy_config()
        return build_backbone(cfg)

    def test_frozen_disables_all_grads(self):
        backbone = self._make_backbone()
        groups = set_backbone_state(backbone, state="frozen")
        assert groups == []
        assert all(not p.requires_grad for p in backbone.parameters())

    def test_fine_tuned_yields_one_group_with_lower_lr(self):
        backbone = self._make_backbone()
        groups = set_backbone_state(backbone, state="fine_tuned", backbone_lr_multiplier=0.1, base_lr=0.5)
        assert len(groups) == 1
        assert groups[0]["lr"] == pytest.approx(0.05)
        assert all(p.requires_grad for p in backbone.parameters())

    def test_partial_freezes_only_early_stages(self):
        backbone = self._make_backbone()
        groups = set_backbone_state(backbone, state="partial", frozen_stages=3, backbone_lr_multiplier=0.1, base_lr=0.5)
        # Stem + layer1 + layer2 frozen → those params have requires_grad=False
        for p in backbone.conv1.parameters():
            assert not p.requires_grad
        for p in backbone.layer1.parameters():
            assert not p.requires_grad
        # Trainable groups exist (layer3, layer4, tail)
        assert len(groups) >= 1


# ---------------------------------------------------------------------------
# Phase-k trainer — BUG-1, 2, 3, 5, 6, 8 dimensional contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_name", ["triplet", "contrastive", "arcface_reweighted", "arcface_margin", "sub_center_arcface"])
@pytest.mark.parametrize("k", [1, 2])
def test_train_phase_k_end_to_end_no_crash(loss_name: str, k: int):
    """Phase k must produce non-NaN loss and a gradient on partition_heads[k]."""
    trainer = _build_trainer(loss=loss_name, num_partitions=3, K=8, num_classes=4)
    # Capture the initial weights of partition_heads[k] to confirm an update.
    before = {n: p.detach().clone() for n, p in trainer.backbone.partition_heads[k].named_parameters()}

    # Run phase 0 briefly (one epoch) so the prior ensemble is well-defined.
    trainer._train_phase_0(epochs=1)
    if k == 2:
        # Phase 1 must have run for k=2 to be a meaningful test.
        trainer._train_phase_k(1, epochs=1)
    trainer._train_phase_k(k, epochs=1)

    # partition_heads[k] should have moved from its initial weights.
    moved = any(
        not torch.equal(before[n], p.detach())
        for n, p in trainer.backbone.partition_heads[k].named_parameters()
    )
    assert moved, f"partition_heads[{k}] did not update under loss={loss_name}"


def test_train_phase_k_with_frozen_backbone_still_trains_current_head():
    """BUG-2: frozen backbone + non-arcface loss must still produce gradient on current head."""
    cfg = _toy_config(loss="triplet")
    cfg["boosting"]["backbone_state"] = "frozen"
    trainer = BoostingTrainer(cfg, device=torch.device("cpu"), logger=_StubLogger())
    trainer._train_dataset = _ToyDataset(n_classes=4, per_class=4, image_size=16)
    trainer.num_classes = 4
    trainer._train_phase_0(epochs=1)
    before = {n: p.detach().clone() for n, p in trainer.backbone.partition_heads[1].named_parameters()}
    trainer._train_phase_k(1, epochs=1)
    moved = any(
        not torch.equal(before[n], p.detach())
        for n, p in trainer.backbone.partition_heads[1].named_parameters()
    )
    assert moved


def test_sub_center_arcface_centroids_get_updated():
    """BUG-3: SubCenterArcFace.weight is now in the optimizer."""
    trainer = _build_trainer(loss="sub_center_arcface", num_partitions=2, K=8, num_classes=4)
    before = trainer.boosting_loss_fn.weight.detach().clone()
    trainer._train_phase_0(epochs=1)
    trainer._train_phase_k(1, epochs=1)
    after = trainer.boosting_loss_fn.weight.detach()
    assert not torch.equal(before, after)


# ---------------------------------------------------------------------------
# Checkpoint format — BUG-9, BUG-17
# ---------------------------------------------------------------------------


def test_checkpoint_format_nested_and_partition_proj(tmp_path: Path):
    trainer = _build_trainer(loss="triplet", num_partitions=2, K=8, num_classes=4)
    trainer.checkpoint_dir = tmp_path
    trainer._save_phase_checkpoint(0)

    backbone_pt = torch.load(tmp_path / "phase_0" / "backbone.pt", weights_only=False)
    assert "model_state_dict" in backbone_pt
    assert "backbone" in backbone_pt["model_state_dict"]

    head_pt = torch.load(tmp_path / "phase_0" / "partition_0.pt", weights_only=False)
    assert "partition_head" in head_pt
    assert "partition_proj" in head_pt
    assert head_pt["partition_proj"] is not None


# ---------------------------------------------------------------------------
# Combiners — BUG-10, BUG-13, BUG-14
# ---------------------------------------------------------------------------


class TestCombiners:
    def test_cosine_concat_accepts_mask_kwarg(self):
        combiner = CosineConcat()
        parts = [torch.randn(4, 8), None, torch.randn(4, 8)]
        out = combiner.combine(parts, mask=None)
        assert out.shape == (4, 24)

    def test_confidence_weighted_is_module_with_scalar_head_params(self):
        cw = ConfidenceWeighted("scalar_head", num_partitions=3, partition_dim=8)
        assert isinstance(cw, nn.Module)
        assert sum(p.numel() for p in cw.parameters()) > 0

    def test_confidence_weighted_embedding_norm_uses_raw(self):
        cw = ConfidenceWeighted("embedding_norm", num_partitions=2, partition_dim=4)
        # Already-unit-norm partition embeddings; raw embeddings provide signal.
        unit = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        parts = [unit, unit]
        raw_strong = [torch.randn(2, 4) * 10.0, torch.randn(2, 4) * 0.1]
        raw_weak = [torch.randn(2, 4) * 0.1, torch.randn(2, 4) * 10.0]
        out_strong = cw.combine(parts, raw_embeddings=raw_strong)
        out_weak = cw.combine(parts, raw_embeddings=raw_weak)
        assert not torch.allclose(out_strong, out_weak), "raw norms must change the weighted output"

    def test_learned_combiner_shape_inference_picks_largest_p(self, tmp_path: Path):
        # Build a fake combiner state for (P=4, K=64) → in_dim = 4*64 + 4 = 260
        from ppi.combiner.mlp import PartitionCombiner
        ckpt_path = tmp_path / "fake_combiner.pt"
        comb = PartitionCombiner(num_partitions=4, partition_dim=64, hidden_dim=32, output_dim=64)
        torch.save({"combiner_state_dict": comb.state_dict()}, ckpt_path)
        # No num_partitions / partition_dim metadata → shape inference must
        # find P=4 not P=2.
        learned = LearnedCombiner(str(ckpt_path), device=torch.device("cpu"))
        # Inspect the first Linear in the loaded combiner. For (P=4, K=64)
        # the in_dim is 4*64 + 4 = 260 — confirms the load succeeded with
        # the right factorisation, not the bogus P=2 / K=129 one.
        first_linear = learned.combiner.net[0]
        assert first_linear.in_features == 260

    def test_learned_combiner_combine_accepts_mask_none(self, tmp_path: Path):
        from ppi.combiner.mlp import PartitionCombiner
        ckpt_path = tmp_path / "fake_combiner2.pt"
        comb = PartitionCombiner(num_partitions=2, partition_dim=8, hidden_dim=16, output_dim=8)
        torch.save({"combiner_state_dict": comb.state_dict(), "num_partitions": 2, "partition_dim": 8}, ckpt_path)
        learned = LearnedCombiner(str(ckpt_path), device=torch.device("cpu"))
        parts = [torch.randn(3, 8), torch.randn(3, 8)]
        out = learned.combine(parts)  # mask omitted
        assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# Scheduler edge case — BUG-15
# ---------------------------------------------------------------------------


def test_build_scheduler_clamps_when_epochs_equals_warmup():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    cfg = {"training": {"epochs": 1, "scheduler": {"warmup_epochs": 1}}}
    sched = build_scheduler(opt, cfg)
    # Should not raise; T_max clamped to ≥ 1.
    sched.step()
    sched.step()


# ---------------------------------------------------------------------------
# Class-balanced sampler — BUG-12
# ---------------------------------------------------------------------------


def test_balanced_sampler_equalises_class_draws():
    from ppi.data import build_class_balanced_sampler

    # Imbalanced dataset: 90% class 0, 10% class 1
    class _Imbal(Dataset):
        def __init__(self):
            self.targets = [0] * 90 + [1] * 10
        def __len__(self): return len(self.targets)
        def __getitem__(self, i): return torch.zeros(1), self.targets[i]

    ds = _Imbal()
    sampler = build_class_balanced_sampler(ds, seed=0)
    drawn = [ds.targets[i] for i in list(sampler)]
    # Expect roughly equal class counts (within statistical noise).
    n0, n1 = drawn.count(0), drawn.count(1)
    ratio = n1 / max(n0, 1)
    assert 0.6 < ratio < 1.6, f"class-1 / class-0 draws = {ratio:.2f}; expected ≈ 1.0"
