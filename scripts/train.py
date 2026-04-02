"""Entry point: python scripts/train.py --config configs/stage0_cifar100.yaml [--variant configs/variant_a.yaml]"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="PPI Training")
    parser.add_argument("--config", required=True, help="Stage config path")
    parser.add_argument("--variant", default=None, help="Variant config overlay path")
    parser.add_argument("--override", action="append", default=[], help="Key=value overrides")
    args = parser.parse_args()

    from ppi.utils.config import apply_overrides, load_full_config

    config = load_full_config(args.config, variant_path=args.variant)
    if args.override:
        config = apply_overrides(config, args.override)

    from ppi.training.trainer import Trainer

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
