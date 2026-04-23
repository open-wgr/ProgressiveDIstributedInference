"""Tests for Variant C: ResidualPartitionStrategy."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ppi.backbones.resnet import PartitionedResNet
from ppi.heads.arcface import ArcFaceHead
from ppi.losses.arcface_loss import ArcFaceLoss
from ppi.partitions.residual import ResidualPartitionStrategy, _parse_subset_key, _is_prefix
from ppi.training.partition_dropout import PartitionDropout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 10
NUM_PARTITIONS = 3
K = 8
EMBED_DIM = NUM_PARTITIONS * K


def _make_config(
    *,
    phase_epochs: list[int] | None = None,
    min_epochs: list[int] | None = None,
    bn_enabled: bool = True,
    fine_tune: bool = False,
    plateau_window_epochs: int = 9999,
) -> dict:
    if phase_epochs is None:
        phase_epochs = [12, 7, 4]
    if min_epochs is None:
        min_epochs = [8, 4, 2]

    phases = [
        {
            "name": "phase1",
            "epochs": phase_epochs[0],
            "min_epochs": min_epochs[0],
            "trainable": ["backbone", "f_0", "arcface_slot_0"],
            "subset_mix": {"[0]": 1.0},
        },
        {
            "name": "phase2",
            "epochs": phase_epochs[1],
            "min_epochs": min_epochs[1],
            "trainable": ["f_1", "arcface_slot_1"],
            "subset_mix": {"[0,1]": 0.6, "[0]": 0.4},
        },
        {
            "name": "phase3",
            "epochs": phase_epochs[2],
            "min_epochs": min_epochs[2],
            "trainable": ["f_2", "arcface_slot_2"],
            "subset_mix": {
                "[0,1,2]": 0.50,
                "[0,1]": 0.25,
                "[0]": 0.15,
                "[0,2]": 0.05,
                "[1,2]": 0.05,
            },
        },
    ]
    return {
        "partitions": {
            "num_partitions": NUM_PARTITIONS,
            "K": K,
            "dropout": {"distribution": [0.4, 0.3, 0.2, 0.1]},
        },
        "residual": {
            "phases": phases,
            "fine_tune": {"enabled": fine_tune, "epochs": 1, "lr_scale": 0.1},
        },
        "switchable_bn": {"enabled": bn_enabled},
        "early_stop": {"plateau_window_epochs": plateau_window_epochs, "plateau_threshold": 0.001},
    }


def _make_backbone() -> PartitionedResNet:
    return PartitionedResNet(
        backbone_name="resnet18",
        num_partitions=NUM_PARTITIONS,
        K=K,
        pretrained=False,
        input_size=32,
    )


def _make_arcface() -> ArcFaceHead:
    return ArcFaceHead(EMBED_DIM, NUM_CLASSES)


def _dummy_backbone_output(batch_size: int = 4) -> dict:
    return {
        "features": torch.randn(batch_size, 512),
        "partitions": [torch.randn(batch_size, K) for _ in range(NUM_PARTITIONS)],
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestParseSubsetKey:
    def test_single(self):
        assert _parse_subset_key("[0]") == frozenset({0})

    def test_pair(self):
        assert _parse_subset_key("[0,1]") == frozenset({0, 1})

    def test_triple(self):
        assert _parse_subset_key("[0,1,2]") == frozenset({0, 1, 2})

    def test_nonprefix(self):
        assert _parse_subset_key("[0,2]") == frozenset({0, 2})

    def test_is_prefix_true(self):
        assert _is_prefix(frozenset({0, 1, 2}))
        assert _is_prefix(frozenset({0}))
        assert _is_prefix(frozenset({0, 1}))

    def test_is_prefix_false(self):
        assert not _is_prefix(frozenset({0, 2}))
        assert not _is_prefix(frozenset({1, 2}))
        assert not _is_prefix(frozenset())


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestZeroInitTails:
    def test_fc_weight_and_bias_zeroed(self):
        """pre_training_setup must zero f_1 and f_2 fc weights and biases."""
        torch.manual_seed(0)
        backbone = _make_backbone()
        strategy = ResidualPartitionStrategy(_make_config())

        strategy.pre_training_setup(backbone, _make_config())

        for idx in range(1, NUM_PARTITIONS):
            head = backbone.partition_heads[idx]
            assert head.fc.weight.abs().max().item() == 0.0, (
                f"partition_heads[{idx}].fc.weight should be zero after init"
            )
            assert head.fc.bias.abs().max().item() == 0.0, (
                f"partition_heads[{idx}].fc.bias should be zero after init"
            )
            assert head.bn2.bias.abs().max().item() == 0.0, (
                f"partition_heads[{idx}].bn2.bias should be zero after init"
            )

    def test_head0_not_zeroed(self):
        """partition_heads[0] should NOT be zero-initialised."""
        torch.manual_seed(0)
        backbone = _make_backbone()
        strategy = ResidualPartitionStrategy(_make_config())
        strategy.pre_training_setup(backbone, _make_config())

        head0 = backbone.partition_heads[0]
        assert head0.fc.weight.abs().max().item() > 0.0

    def test_later_heads_frozen_after_setup(self):
        """Heads 1 and 2 must have requires_grad=False after pre_training_setup."""
        backbone = _make_backbone()
        strategy = ResidualPartitionStrategy(_make_config())
        strategy.pre_training_setup(backbone, _make_config())

        for idx in range(1, NUM_PARTITIONS):
            for p in backbone.partition_heads[idx].parameters():
                assert not p.requires_grad, (
                    f"partition_heads[{idx}] param should be frozen after setup"
                )


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    def test_phase_changed_flag_set_on_budget_exhaustion(self):
        """phase_changed must flip True when an epoch budget is exhausted."""
        config = _make_config(phase_epochs=[2, 7, 4], min_epochs=[2, 4, 2])
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()

        assert strategy._current_phase == 0
        assert not strategy.phase_changed

        # Phase 0 budget = 2 epochs
        for epoch in range(1, 3):
            strategy.post_epoch_hook(epoch, backbone)

        assert strategy.phase_changed, "phase_changed should be True after budget exhausted"
        assert strategy._current_phase == 1

    def test_phase_epoch_counter_resets(self):
        """Phase-local epoch counter must reset to 0 when advancing."""
        config = _make_config(phase_epochs=[1, 7, 4], min_epochs=[1, 4, 2])
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()

        strategy.post_epoch_hook(1, backbone)  # exhausts phase 0
        assert strategy._phase_epoch_count == 0

    def test_all_phases_advance_sequentially(self):
        """All three phases should advance in order when budgets are minimal."""
        config = _make_config(
            phase_epochs=[1, 1, 1], min_epochs=[1, 1, 1],
        )
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()

        for epoch in range(1, 4):
            strategy.post_epoch_hook(epoch, backbone)
            strategy.phase_changed = False  # simulate trainer clearing the flag

        assert strategy._current_phase == 3

    def test_backbone_frozen_after_phase0(self):
        """Backbone requires_grad must be False once phase 1 begins."""
        config = _make_config(phase_epochs=[1, 7, 4], min_epochs=[1, 4, 2])
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()
        strategy.pre_training_setup(backbone, config)

        strategy.post_epoch_hook(1, backbone)  # advances to phase 1

        for name, p in backbone.named_parameters():
            if name.startswith("partition_heads.1."):
                assert p.requires_grad, f"{name} should be trainable in phase 1"
            elif name.startswith("partition_heads."):
                assert not p.requires_grad, f"{name} should be frozen in phase 1"

    def test_no_advance_before_min_epochs(self):
        """Phase should not advance before min_epochs even if plateau detected."""
        config = _make_config(
            phase_epochs=[10, 7, 4],
            min_epochs=[5, 4, 2],
            plateau_window_epochs=1,  # tiny window → plateau detected immediately
        )
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()

        # Feed identical epoch losses so plateau is detectable after 1 epoch
        flat_loss = {"train/epoch_loss_total": 1.0}

        # Only 2 epochs — below min_epochs=5; should not advance despite plateau
        for ep in range(1, 3):
            strategy.post_epoch_hook(ep, backbone, metrics=flat_loss)

        assert strategy._current_phase == 0, "Should not advance before min_epochs"


# ---------------------------------------------------------------------------
# ArcFace gradient hooks
# ---------------------------------------------------------------------------


class TestArcFaceSlotHooks:
    def _run_backward(self, strategy, arcface_head, phase):
        """Install hooks for phase and run a backward pass; return weight grad."""
        strategy._install_arcface_hooks(arcface_head, phase)

        batch_size = 4
        embedding = torch.randn(batch_size, EMBED_DIM, requires_grad=True)
        labels = torch.randint(0, NUM_CLASSES, (batch_size,))
        arcface_loss = ArcFaceLoss(s=64.0, m=0.5)

        cosine = arcface_head(embedding)
        loss = arcface_loss(cosine, labels)
        loss.backward()
        return arcface_head.weight.grad.clone()

    def test_phase0_only_slot0_receives_grad(self):
        """In phase 0, only columns [0:K] should have non-zero gradient."""
        strategy = ResidualPartitionStrategy(_make_config())
        arcface_head = _make_arcface()

        grad = self._run_backward(strategy, arcface_head, phase=0)

        # Slot 0 (cols 0:K) must have gradients
        assert grad[:, :K].abs().max().item() > 0.0, "Slot 0 should receive gradient"
        # Slots 1 and 2 (cols K:3K) must be zeroed
        assert grad[:, K:].abs().max().item() == 0.0, "Slots 1&2 must be frozen"

    def test_phase1_only_slot1_receives_grad(self):
        """In phase 1, only columns [K:2K] should have non-zero gradient."""
        strategy = ResidualPartitionStrategy(_make_config())
        arcface_head = _make_arcface()

        grad = self._run_backward(strategy, arcface_head, phase=1)

        assert grad[:, :K].abs().max().item() == 0.0, "Slot 0 must be frozen"
        assert grad[:, K:2 * K].abs().max().item() > 0.0, "Slot 1 should receive gradient"
        assert grad[:, 2 * K:].abs().max().item() == 0.0, "Slot 2 must be frozen"

    def test_phase2_only_slot2_receives_grad(self):
        """In phase 2, only columns [2K:3K] should have non-zero gradient."""
        strategy = ResidualPartitionStrategy(_make_config())
        arcface_head = _make_arcface()

        grad = self._run_backward(strategy, arcface_head, phase=2)

        assert grad[:, :2 * K].abs().max().item() == 0.0, "Slots 0&1 must be frozen"
        assert grad[:, 2 * K:].abs().max().item() > 0.0, "Slot 2 should receive gradient"

    def test_hooks_replaced_on_phase_change(self):
        """Old hooks must be removed when phase changes."""
        strategy = ResidualPartitionStrategy(_make_config())
        arcface_head = _make_arcface()

        strategy._install_arcface_hooks(arcface_head, phase=0)
        initial_hook_count = len(strategy._arcface_hook_handles)
        assert initial_hook_count == 1

        strategy._install_arcface_hooks(arcface_head, phase=1)
        assert len(strategy._arcface_hook_handles) == 1, "Should still be exactly 1 hook"

    def test_fine_tune_phase_no_hooks(self):
        """Fine-tune phase (phase index >= num_partitions) installs no hooks."""
        strategy = ResidualPartitionStrategy(_make_config())
        arcface_head = _make_arcface()

        strategy._install_arcface_hooks(arcface_head, phase=NUM_PARTITIONS)
        assert len(strategy._arcface_hook_handles) == 0


# ---------------------------------------------------------------------------
# Subset sampling
# ---------------------------------------------------------------------------


class TestSubsetSampling:
    def test_phase0_only_samples_subset0(self):
        """Phase 0 mix is {0}: 1.0 — should always sample {0}."""
        strategy = ResidualPartitionStrategy(_make_config())
        torch.manual_seed(0)
        import random
        random.seed(0)

        for _ in range(100):
            subset = strategy._sample_subset(0)
            assert subset == frozenset({0}), f"Phase 0 should only sample {{0}}, got {subset}"

    def test_phase2_distribution_roughly_correct(self):
        """Phase 2 full mix — run 5000 samples and check frequencies within tolerance."""
        strategy = ResidualPartitionStrategy(_make_config())
        import random
        random.seed(42)

        counts: dict = {}
        n = 5000
        for _ in range(n):
            subset = strategy._sample_subset(2)
            counts[subset] = counts.get(subset, 0) + 1

        full = frozenset({0, 1, 2})
        assert abs(counts.get(full, 0) / n - 0.50) < 0.03, "Full subset should be ~50%"
        prefix2 = frozenset({0, 1})
        assert abs(counts.get(prefix2, 0) / n - 0.25) < 0.03, "{0,1} should be ~25%"


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_training_step_returns_loss_and_metrics(self):
        """training_step must return a (Tensor, dict) tuple."""
        strategy = ResidualPartitionStrategy(_make_config())
        strategy.train()
        arcface_head = _make_arcface()
        arcface_loss = ArcFaceLoss(s=64.0, m=0.5)
        dropout = PartitionDropout(num_partitions=NUM_PARTITIONS)

        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = _dummy_backbone_output(batch_size=4)

        result = strategy.training_step(out, labels, arcface_head, arcface_loss, dropout)
        assert result is not None
        loss, metrics = result

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert "arcface" in metrics
        assert "width" in metrics
        assert "norm_0" in metrics
        assert "norm_1" in metrics
        assert "norm_2" in metrics

    def test_training_step_loss_is_finite(self):
        """Loss must be finite — no NaN from zero-init or BN."""
        strategy = ResidualPartitionStrategy(_make_config())
        backbone = _make_backbone()
        strategy.pre_training_setup(backbone, _make_config())
        strategy.train()
        arcface_head = _make_arcface()
        arcface_loss = ArcFaceLoss(s=64.0, m=0.5)
        dropout = PartitionDropout(num_partitions=NUM_PARTITIONS)

        labels = torch.randint(0, NUM_CLASSES, (4,))
        x = torch.randn(4, 3, 32, 32)
        out = backbone(x)

        loss, _ = strategy.training_step(out, labels, arcface_head, arcface_loss, dropout)
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"

    def test_hooks_installed_on_first_step(self):
        """ArcFace slot hooks should be installed during the first training_step."""
        strategy = ResidualPartitionStrategy(_make_config())
        strategy.train()
        arcface_head = _make_arcface()
        arcface_loss = ArcFaceLoss(s=64.0, m=0.5)
        dropout = PartitionDropout(num_partitions=NUM_PARTITIONS)

        assert strategy._hooks_phase == -1  # not installed yet

        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = _dummy_backbone_output()
        strategy.training_step(out, labels, arcface_head, arcface_loss, dropout)

        assert strategy._hooks_phase == 0
        assert len(strategy._arcface_hook_handles) == 1


# ---------------------------------------------------------------------------
# Switchable BN — eval gating
# ---------------------------------------------------------------------------


class TestSwitchableBNGating:
    def test_bn_applied_for_prefix_eval_config(self):
        """post_assembly should apply BN for a prefix config during eval."""
        strategy = ResidualPartitionStrategy(_make_config(bn_enabled=True))
        strategy.eval()
        strategy.set_eval_width(2, partition_set={0, 1})

        embedding = torch.randn(4, EMBED_DIM)
        strategy.switchable_bn.train()  # ensure BN has running stats
        # Give the BN some stats first
        for _ in range(5):
            strategy.switchable_bn.active_width = 2
            strategy.switchable_bn(torch.randn(16, EMBED_DIM))
        strategy.switchable_bn.eval()

        out = strategy.post_assembly(embedding)
        # BN + normalise changes the embedding
        assert not torch.allclose(out, embedding / (embedding.norm(dim=1, keepdim=True) + 1e-12))

    def test_bn_skipped_for_nonprefix_eval_config(self):
        """post_assembly must NOT apply BN for non-prefix config {0, 2}."""
        strategy = ResidualPartitionStrategy(_make_config(bn_enabled=True))
        strategy.eval()
        strategy.set_eval_width(2, partition_set={0, 2})

        embedding = torch.randn(4, EMBED_DIM)
        # pre-normalize to check identity
        import torch.nn.functional as F
        normed = F.normalize(embedding, dim=1, eps=1e-12)

        out = strategy.post_assembly(normed)
        # Should be identity (no BN applied)
        assert torch.allclose(out, normed, atol=1e-6), (
            "post_assembly should be identity for non-prefix eval configs"
        )

    def test_bn_disabled_config(self):
        """When switchable_bn.enabled=False, post_assembly is always identity."""
        strategy = ResidualPartitionStrategy(_make_config(bn_enabled=False))
        strategy.eval()
        strategy.set_eval_width(3, partition_set={0, 1, 2})

        embedding = torch.randn(4, EMBED_DIM)
        out = strategy.post_assembly(embedding)
        assert torch.allclose(out, embedding)


# ---------------------------------------------------------------------------
# get_trainable_parameters
# ---------------------------------------------------------------------------


class TestGetTrainableParameters:
    def test_phase0_includes_backbone(self):
        """Phase 0 must include backbone parameters."""
        strategy = ResidualPartitionStrategy(_make_config())
        backbone = _make_backbone()

        params = strategy.get_trainable_parameters(backbone, phase=0)
        param_set = {id(p) for p in params}

        backbone_core_param = next(backbone.layer1.parameters())
        assert id(backbone_core_param) in param_set, "Phase 0 must include backbone params"

    def test_phase1_excludes_backbone(self):
        """Phase 1 must only include partition_heads[1] parameters."""
        strategy = ResidualPartitionStrategy(_make_config())
        backbone = _make_backbone()

        params = strategy.get_trainable_parameters(backbone, phase=1)
        param_set = {id(p) for p in params}

        backbone_param = next(backbone.layer1.parameters())
        assert id(backbone_param) not in param_set, "Phase 1 must NOT include backbone"

        head1_param = next(backbone.partition_heads[1].parameters())
        assert id(head1_param) in param_set, "Phase 1 must include partition_heads[1]"

    def test_defaults_to_current_phase(self):
        """Calling without phase arg should use _current_phase."""
        strategy = ResidualPartitionStrategy(_make_config())
        strategy._current_phase = 2
        backbone = _make_backbone()

        params_explicit = strategy.get_trainable_parameters(backbone, phase=2)
        params_default = strategy.get_trainable_parameters(backbone)

        assert {id(p) for p in params_explicit} == {id(p) for p in params_default}
