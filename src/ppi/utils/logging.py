"""Experiment logging: TensorBoard, Weights & Biases, checkpoints, config saving."""

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

        # Weights & Biases
        self._wandb_run = None
        if log_cfg.get("wandb", False):
            self._init_wandb(config, log_cfg, variant, timestamp)

    def _init_wandb(
        self,
        config: dict[str, Any],
        log_cfg: dict[str, Any],
        variant: str,
        timestamp: str,
    ) -> None:
        """Initialise a Weights & Biases run."""
        try:
            import wandb
        except ImportError:
            print(
                "[Logger] WARNING: wandb=true in config but wandb is not installed. "
                "Install with: pip install wandb"
            )
            return

        project = log_cfg.get("wandb_project", "ppi")
        name = log_cfg.get("wandb_name", f"{variant}_{timestamp}")
        tags = log_cfg.get("wandb_tags", [])
        if variant != "default":
            tags = list(tags) + [variant]

        # Build a flat summary of key hyperparameters for wandb config
        wandb_config = {
            "seed": config.get("seed"),
            "backbone": config.get("backbone", {}).get("name"),
            "num_partitions": config.get("partitions", {}).get("num_partitions"),
            "K": config.get("partitions", {}).get("K"),
            "embedding_dim": (
                config.get("partitions", {}).get("num_partitions", 3)
                * config.get("partitions", {}).get("K", 128)
            ),
            "dropout_dist": config.get("partitions", {}).get("dropout", {}).get("distribution"),
            "arcface_s": config.get("arcface", {}).get("s"),
            "arcface_m": config.get("arcface", {}).get("m"),
            "epochs": config.get("training", {}).get("epochs"),
            "batch_size": config.get("training", {}).get("batch_size"),
            "lr": config.get("training", {}).get("optimizer", {}).get("lr"),
            "optimizer": config.get("training", {}).get("optimizer", {}).get("type"),
            "weight_decay": config.get("training", {}).get("optimizer", {}).get("weight_decay"),
            "scheduler": config.get("training", {}).get("scheduler", {}).get("type"),
            "warmup_epochs": config.get("training", {}).get("scheduler", {}).get("warmup_epochs"),
            "dataset": config.get("data", {}).get("dataset"),
            "variant": variant,
            "lambda_orth": config.get("orthogonality", {}).get("lambda"),
            "orth_mode": config.get("orthogonality", {}).get("mode"),
        }
        # Remove None values
        wandb_config = {k: v for k, v in wandb_config.items() if v is not None}

        self._wandb_run = wandb.init(
            project=project,
            name=name,
            config=wandb_config,
            tags=tags,
            dir=str(self.run_dir),
            reinit=True,
        )
        print(f"[Logger] W&B run: {self._wandb_run.url}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to all active backends."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_epoch(self, metrics: dict[str, float], epoch: int) -> None:
        """Log a dict of epoch-level metrics to all active backends.

        Use this for end-of-epoch summaries to avoid per-step overhead on wandb.
        """
        for tag, value in metrics.items():
            if self._writer is not None:
                self._writer.add_scalar(tag, value, epoch)
        if self._wandb_run is not None:
            import wandb
            wandb.log({**metrics, "epoch": epoch})

    def save_checkpoint(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        metrics: dict[str, float] | None = None,
        scheduler_state: dict | None = None,
        global_step: int | None = None,
        scaler_state: dict | None = None,
    ) -> Path:
        path = self.run_dir / f"checkpoint_epoch{epoch}.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics or {},
        }
        if scheduler_state is not None:
            payload["scheduler_state_dict"] = scheduler_state
        if global_step is not None:
            payload["global_step"] = global_step
        if scaler_state is not None:
            payload["scaler_state_dict"] = scaler_state
        torch.save(payload, path)

        # Log checkpoint as wandb artifact
        if self._wandb_run is not None:
            try:
                import wandb
                artifact = wandb.Artifact(
                    f"checkpoint-epoch{epoch}",
                    type="model",
                    metadata=metrics or {},
                )
                artifact.add_file(str(path))
                self._wandb_run.log_artifact(artifact)
            except Exception as e:
                print(f"[Logger] WARNING: Failed to upload checkpoint to W&B: {e}")

        return path

    @staticmethod
    def load_checkpoint(path: str | Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
