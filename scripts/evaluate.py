"""Entry point: python scripts/evaluate.py --checkpoint X --config Y [--partitions 0,1,2]"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="PPI Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--partitions", default=None, help="Comma-separated partition indices")
    args = parser.parse_args()

    from ppi.utils.config import load_full_config

    config = load_full_config(args.config)

    partition_configs = None
    if args.partitions:
        partition_configs = [set(int(x) for x in args.partitions.split(","))]

    from ppi.evaluation.evaluator import Evaluator

    evaluator = Evaluator(config, args.checkpoint)
    results = evaluator.evaluate(partition_configs=partition_configs)
    for cfg_name, metrics in results.items():
        print(f"\n{cfg_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
