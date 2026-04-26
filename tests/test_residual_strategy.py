"""Tests for Variant C: ResidualPartitionStrategy."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # Partitions are unit-norm (as the backbone now produces them)
    raw = [torch.randn(batch_size, K) for _ in range(NUM_PARTITIONS)]
    return {
        "features": torch.randn(batch_size, 512),
        "partitions": [F.normalize(p, dim=1, eps=1e-12) for p in raw],
    }


# ---------------------------------------------------------------------------
# Backbone output contract
# ---------------------------------------------------------------------------


class TestBackboneOutputContract:
    def test_partitions_are_unit_norm(self):
        """Backbone must return L2-normalised partitions (norm ≈ 1 per sample)."""
        backbone = _make_backbone()
        backbone.eval()
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            out = backbone(x)
        for i, p in enumerate(out["partitions"]):
            norms = p.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
                f"partition_heads[{i}] output must have unit L2 norm"
            )

    def test_gradient_checkpointing_same_output(self):
        """Gradient checkpointing must not change forward-pass outputs."""
        torch.manual_seed(0)
        normal = PartitionedResNet("resnet18", NUM_PARTITIONS, K, input_size=32,
                                   gradient_checkpointing=False)
        ckpt = PartitionedResNet("resnet18", NUM_PARTITIONS, K, input_size=32,
                                 gradient_checkpointing=True)
        ckpt.load_state_dict(normal.state_dict())

        x = torch.randn(4, 3, 32, 32)
        normal.train()
        ckpt.train()
        with torch.no_grad():
            out_n = normal(x)
            out_c = ckpt(x)

        for i in range(NUM_PARTITIONS):
            assert torch.allclose(out_n["partitions"][i], out_c["partitions"][i], atol=1e-5), (
                f"partition {i} differs between normal and checkpointed forward"
            )


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
# Evaluator subset generation
# ---------------------------------------------------------------------------


class TestEvalSubsetGeneration:
    def test_all_configs_contain_p0(self):
        """Every generated eval config must include partition 0."""
        from ppi.evaluation.evaluator import _all_partition_configs
        for n in range(2, 6):
            configs = _all_partition_configs(n)
            for cfg in configs:
                assert 0 in cfg, f"Config {cfg} missing P0 (N={n})"

    def test_correct_count(self):
        """Should return exactly 2^{N-1} configs."""
        from ppi.evaluation.evaluator import _all_partition_configs
        for n in range(1, 6):
            configs = _all_partition_configs(n)
            assert len(configs) == 2 ** (n - 1), (
                f"N={n}: expected {2**(n-1)} configs, got {len(configs)}"
            )

    def test_no_duplicates(self):
        """No two configs should be identical."""
        from ppi.evaluation.evaluator import _all_partition_configs
        configs = _all_partition_configs(4)
        frozen = [frozenset(c) for c in configs]
        assert len(frozen) == len(set(frozen))


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
        config = _make_config(phase_epochs=[1, 1, 1], min_epochs=[1, 1, 1])
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()

        for epoch in range(1, 4):
            strategy.post_epoch_hook(epoch, backbone)
            strategy.phase_changed = False

        assert strategy._current_phase == 3

    def test_backbone_frozen_after_phase0(self):
        """Backbone requires_grad must be False once phase 1 begins."""
        config = _make_config(phase_epochs=[1, 7, 4], min_epochs=[1, 4, 2])
        strategy = ResidualPartitionStrategy(config)
        backbone = _make_backbone()
        strategy.pre_training_setup(backbone, config)

        strategy.post_epoch_hook(1, backbone)

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

        assert grad[:, :K].abs().max().item() > 0.0, "Slot 0 should receive gradient"
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
        assert len(strategy._arcface_hook_handles) == 1

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

        loss, metrics = strategy.training_step(out, labels, arcface_head, arcface_loss, dropout)

        assert isinstance(loss, torch.Tensor) and loss.ndim == 0
        assert "arcface" in metrics
        assert "width" in metrics
        assert all(f"norm_{i}" in metrics for i in range(NUM_PARTITIONS))

    def test_training_step_loss_is_finite(self):
        """Loss must be finite — no NaN from zero-init partitions."""
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

    def test_post_assembly_is_identity(self):
        """post_assembly must return its input unchanged."""
        strategy = ResidualPartitionStrategy(_make_config())
        embedding = torch.randn(4, EMBED_DIM)
        out = strategy.post_assembly(embedding)
        assert out is embedding, "post_assembly should return the input tensor unchanged"

    def test_hooks_installed_on_first_step(self):
        """ArcFace slot hooks should be installed during the first training_step."""
        strategy = ResidualPartitionStrategy(_make_config())
        strategy.train()
        arcface_head = _make_arcface()
        arcface_loss = ArcFaceLoss(s=64.0, m=0.5)
        dropout = PartitionDropout(num_partitions=NUM_PARTITIONS)

        assert strategy._hooks_phase == -1

        labels = torch.randint(0, NUM_CLASSES, (4,))
        out = _dummy_backbone_output()
        strategy.training_step(out, labels, arcface_head, arcface_loss, dropout)

        assert strategy._hooks_phase == 0
        assert len(strategy._arcface_hook_handles) == 1


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
