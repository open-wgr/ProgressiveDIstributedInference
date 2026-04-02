"""YAML config loading with base-stage-variant merge and CLI overrides."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = [
    "seed",
    ("backbone", "name"),
    ("partitions", "num_partitions"),
    ("partitions", "K"),
    ("training", "epochs"),
    ("data", "dataset"),
]

KNOWN_TOP_LEVEL_KEYS = {
    "seed",
    "backbone",
    "partitions",
    "arcface",
    "training",
    "data",
    "logging",
    "variant",
    "orthogonality",
    "positional_encoding",
    "evaluation",
    "_base_",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file."""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep-merge override into base. Lists are replaced, dicts are merged."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_base(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """If config has a _base_ key, load and merge the base config underneath."""
    if "_base_" not in config:
        return config
    base_path = config_dir / config["_base_"]
    base_config = load_config(base_path)
    # Recursively resolve the base's own _base_ if present
    base_config = _resolve_base(base_config, base_path.parent)
    merged = merge_configs(base_config, config)
    del merged["_base_"]
    return merged


def load_full_config(
    stage_path: str | Path,
    variant_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a complete config: resolve base inheritance, merge variant overlay.

    Args:
        stage_path: Path to the stage config (e.g. stage0_cifar100.yaml).
        variant_path: Optional path to a variant config overlay.

    Returns:
        Fully merged config dict.
    """
    stage_path = Path(stage_path)
    config = load_config(stage_path)
    config = _resolve_base(config, stage_path.parent)

    if variant_path is not None:
        variant_path = Path(variant_path)
        variant_config = load_config(variant_path)
        # Variant configs should not have _base_ — they overlay on top of resolved stage
        if "_base_" in variant_config:
            variant_config = _resolve_base(variant_config, variant_path.parent)
        config = merge_configs(config, variant_config)

    _warn_unknown_keys(config)
    _validate_required_keys(config)
    return config


def _validate_required_keys(config: dict[str, Any]) -> None:
    """Raise ValueError if any required keys are missing."""
    for key in REQUIRED_KEYS:
        if isinstance(key, tuple):
            obj = config
            path_str = ".".join(key)
            for part in key:
                if not isinstance(obj, dict) or part not in obj:
                    raise ValueError(f"Missing required config key: {path_str}")
                obj = obj[part]
        else:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")


def _warn_unknown_keys(config: dict[str, Any]) -> None:
    """Emit warnings for unrecognised top-level keys."""
    for key in config:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            warnings.warn(
                f"Unknown top-level config key: '{key}'",
                UserWarning,
                stacklevel=3,
            )


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation CLI overrides to a config dict.

    Each override has the form 'key.subkey=value'. Values are auto-cast
    to int, float, or bool where possible; otherwise kept as strings.
    """
    config = _deep_copy_dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format (expected key=value): {override}")
        key_path, raw_value = override.split("=", 1)
        parts = key_path.split(".")
        value = _cast_value(raw_value)

        obj = config
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part], dict):
                obj[part] = {}
            obj = obj[part]
        obj[parts[-1]] = value
    return config


def _cast_value(raw: str) -> Any:
    """Best-effort cast of a string to int, float, bool, or leave as str."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dict/list structure without importing copy."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result
