"""Entry point for PPI training.

Examples:
    # Basic Stage 0 run
    python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml

    # Sweep orthogonality lambda
    python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml --lambda 0.5
    python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml --lambda 1.0

    # Longer training with higher LR
    python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml --lr 0.2 --epochs 100

    # Custom partition dim sweep
    python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml --K 32
    python scripts/train.py --config configs/stage0_cifar100.yaml --variant configs/variant_a.yaml --K 128

    # Escape hatch for anything not exposed as a flag
    python scripts/train.py --config configs/stage0_cifar100.yaml --override arcface.s=32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without `pip install -e .` by adding src/ to the path
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def main():
    parser = argparse.ArgumentParser(
        description="PPI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config files
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--variant", default=None, help="Variant config overlay path")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup epochs")

    # Partition hyperparameters
    parser.add_argument("--K", type=int, default=None, help="Per-partition embedding dimension")
    parser.add_argument("--num-partitions", type=int, default=None, help="Number of partitions")

    # ArcFace
    parser.add_argument("--arcface-s", type=float, default=None, help="ArcFace scale factor")
    parser.add_argument("--arcface-m", type=float, default=None, help="ArcFace angular margin")

    # Variant A: orthogonality
    parser.add_argument("--lambda", dest="lambda_orth", type=float, default=None,
                        help="Orthogonality regularisation weight")

    # Dropout distribution
    parser.add_argument("--dropout-dist", type=float, nargs=4, default=None,
                        metavar=("P1", "P2", "P3", "P0"),
                        help="Partition dropout distribution [1-part, 2-part, 3-part, 0-part]")

    # Infrastructure
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")

    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (default: 'ppi')")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name (default: auto-generated from variant + timestamp)")

    parser.add_argument("--override", action="append", default=[],
                        help="Arbitrary key=value config overrides (dot notation)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume training from. "
                             "Loads model, optimizer, scheduler, epoch, and "
                             "global_step. Note: starts a fresh W&B run; pass "
                             "--wandb-name to group with the original run.")

    args = parser.parse_args()

    from ppi.utils.config import apply_overrides, load_full_config

    config = load_full_config(args.config, variant_path=args.variant)

    # Apply named CLI args to config
    _apply_cli_args(config, args)

    # Apply freeform overrides last (highest priority)
    if args.override:
        config = apply_overrides(config, args.override)

    from ppi.training.trainer import Trainer

    trainer = Trainer(config, resume_from=args.resume)
    trainer.train()


def _apply_cli_args(config: dict, args: argparse.Namespace) -> None:
    """Map named CLI arguments into the config dict (in-place)."""
    if args.lr is not None:
        config.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.warmup is not None:
        config.setdefault("training", {}).setdefault("scheduler", {})["warmup_epochs"] = args.warmup
    if args.K is not None:
        config.setdefault("partitions", {})["K"] = args.K
    if args.num_partitions is not None:
        config.setdefault("partitions", {})["num_partitions"] = args.num_partitions
    if args.arcface_s is not None:
        config.setdefault("arcface", {})["s"] = args.arcface_s
    if args.arcface_m is not None:
        config.setdefault("arcface", {})["m"] = args.arcface_m
    if args.lambda_orth is not None:
        config.setdefault("orthogonality", {})["lambda"] = args.lambda_orth
    if args.dropout_dist is not None:
        config.setdefault("partitions", {}).setdefault("dropout", {})["distribution"] = args.dropout_dist
    if args.seed is not None:
        config["seed"] = args.seed
    if args.workers is not None:
        config.setdefault("data", {})["num_workers"] = args.workers
    if args.wandb:
        config.setdefault("logging", {})["wandb"] = True
    if args.wandb_project is not None:
        config.setdefault("logging", {})["wandb_project"] = args.wandb_project
    if args.wandb_name is not None:
        config.setdefault("logging", {})["wandb_name"] = args.wandb_name


if __name__ == "__main__":
    main()
