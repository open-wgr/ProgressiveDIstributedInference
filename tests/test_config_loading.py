"""Tests for config loading, merging, overrides, and validation."""

from __future__ import annotations

import warnings

import pytest
import yaml

from ppi.utils.config import (
    apply_overrides,
    load_config,
    load_full_config,
    merge_configs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with base, stage, and variant files."""
    base = {
        "seed": 42,
        "backbone": {"name": "resnet50", "pretrained": False},
        "partitions": {"num_partitions": 3, "K": 128},
        "arcface": {"s": 64, "m": 0.5},
        "training": {"epochs": 24, "batch_size": 256, "optimizer": {"lr": 0.1}},
        "data": {"dataset": "ms1mv2", "root": "/data/ms1mv2/"},
        "logging": {"output_dir": "runs/", "tensorboard": True},
    }
    _write_yaml(tmp_path / "base.yaml", base)

    stage = {
        "_base_": "base.yaml",
        "data": {"dataset": "cifar100", "root": "/data/cifar100/", "input_size": 32},
        "backbone": {"name": "resnet18"},
        "partitions": {"K": 64},
        "training": {"epochs": 50, "batch_size": 128},
        "arcface": {"num_classes": 100},
    }
    _write_yaml(tmp_path / "stage0.yaml", stage)

    variant = {
        "variant": "orthogonal",
        "orthogonality": {"lambda": 0.1},
    }
    _write_yaml(tmp_path / "variant_a.yaml", variant)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_base(self, config_dir):
        cfg = load_config(config_dir / "base.yaml")
        assert cfg["seed"] == 42
        assert cfg["backbone"]["name"] == "resnet50"

    def test_loads_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_config(empty)
        assert cfg == {}


class TestMergeConfigs:
    def test_override_leaf(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        merged = merge_configs(base, override)
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3  # sibling preserved

    def test_list_replaced_not_appended(self):
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        merged = merge_configs(base, override)
        assert merged["items"] == [4, 5]

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        merged = merge_configs(base, override)
        assert merged == {"a": 1, "b": 2}


class TestLoadFullConfig:
    def test_stage_resolves_base(self, config_dir):
        cfg = load_full_config(config_dir / "stage0.yaml")
        # Overridden values
        assert cfg["data"]["dataset"] == "cifar100"
        assert cfg["backbone"]["name"] == "resnet18"
        assert cfg["partitions"]["K"] == 64
        # Inherited values
        assert cfg["seed"] == 42
        assert cfg["arcface"]["s"] == 64
        assert cfg["logging"]["tensorboard"] is True
        # _base_ key removed
        assert "_base_" not in cfg

    def test_variant_overlay(self, config_dir):
        cfg = load_full_config(
            config_dir / "stage0.yaml",
            variant_path=config_dir / "variant_a.yaml",
        )
        assert cfg["variant"] == "orthogonal"
        assert cfg["orthogonality"]["lambda"] == 0.1
        # Stage + base values still present
        assert cfg["data"]["dataset"] == "cifar100"
        assert cfg["seed"] == 42

    def test_missing_required_key_raises(self, tmp_path):
        incomplete = {"backbone": {"name": "resnet50"}}
        _write_yaml(tmp_path / "bad.yaml", incomplete)
        with pytest.raises(ValueError, match="Missing required config key"):
            load_full_config(tmp_path / "bad.yaml")

    def test_unknown_top_level_key_warns(self, config_dir):
        # Add an unknown key to the stage config
        stage = load_config(config_dir / "stage0.yaml")
        stage["foobar"] = 123
        _write_yaml(config_dir / "stage0_bad.yaml", stage)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_full_config(config_dir / "stage0_bad.yaml")
            unknown_warnings = [x for x in w if "Unknown top-level config key" in str(x.message)]
            assert len(unknown_warnings) == 1
            assert "foobar" in str(unknown_warnings[0].message)


class TestApplyOverrides:
    def test_dot_notation(self):
        cfg = {"training": {"optimizer": {"lr": 0.1}}}
        result = apply_overrides(cfg, ["training.optimizer.lr=0.01"])
        assert result["training"]["optimizer"]["lr"] == 0.01

    def test_creates_intermediate_dicts(self):
        cfg = {}
        result = apply_overrides(cfg, ["a.b.c=hello"])
        assert result["a"]["b"]["c"] == "hello"

    def test_type_casting(self):
        cfg = {}
        result = apply_overrides(cfg, [
            "int_val=42",
            "float_val=3.14",
            "bool_true=true",
            "bool_false=False",
            "str_val=hello",
        ])
        assert result["int_val"] == 42
        assert isinstance(result["int_val"], int)
        assert result["float_val"] == 3.14
        assert isinstance(result["float_val"], float)
        assert result["bool_true"] is True
        assert result["bool_false"] is False
        assert result["str_val"] == "hello"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid override format"):
            apply_overrides({}, ["no_equals_sign"])

    def test_does_not_mutate_original(self):
        cfg = {"a": 1}
        result = apply_overrides(cfg, ["a=2"])
        assert cfg["a"] == 1
        assert result["a"] == 2


class TestRealConfigs:
    """Smoke tests against the actual config files in the repo."""

    def test_load_base(self):
        cfg = load_config("configs/base.yaml")
        assert cfg["seed"] == 42
        assert cfg["backbone"]["name"] == "resnet50"
        assert cfg["partitions"]["K"] == 128

    def test_load_stage0(self):
        cfg = load_full_config("configs/stage0_cifar100.yaml")
        assert cfg["data"]["dataset"] == "cifar100"
        assert cfg["backbone"]["name"] == "resnet18"
        assert cfg["partitions"]["K"] == 64
        assert cfg["seed"] == 42  # inherited from base

    def test_load_stage0_with_variant_a(self):
        cfg = load_full_config(
            "configs/stage0_cifar100.yaml",
            variant_path="configs/variant_a.yaml",
        )
        assert cfg["variant"] == "orthogonal"
        assert cfg["data"]["dataset"] == "cifar100"

    def test_load_stage1(self):
        cfg = load_full_config("configs/stage1_casia.yaml")
        assert cfg["data"]["dataset"] == "casia"
        assert cfg["backbone"]["name"] == "resnet50"  # inherited

    def test_load_stage2(self):
        cfg = load_full_config("configs/stage2_ms1mv2.yaml")
        assert cfg["data"]["dataset"] == "ms1mv2"
