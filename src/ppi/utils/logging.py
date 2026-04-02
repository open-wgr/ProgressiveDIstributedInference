"""Experiment logging: TensorBoard, checkpoints, config saving."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


class ExperimentLogger:
    """Manages a single experiment run directory with logging and checkpoints."""

    def __init__(self, config: dict[str, Any]) -> None:
        log_cfg = config.get("logging", {})
        variant = config.get("variant", "default")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_cfg.get("output_dir", "runs")) / f"{variant}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # TensorBoard
        self._writer = None
        if log_cfg.get("tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def save_checkpoint(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        path = self.run_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "metrics": metrics or {},
            },
            path,
        )
        return path

    @staticmethod
    def load_checkpoint(path: str | Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
