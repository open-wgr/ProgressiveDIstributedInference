"""Entry point for PPI evaluation.

Examples:
    # Classification evaluation (CIFAR-100 / nearest-centroid)
    python scripts/evaluate.py --checkpoint runs/run_001/checkpoint_epoch50.pt --config configs/stage0_cifar100.yaml

    # LFW pair verification (all 7 partition configs)
    python scripts/evaluate.py --checkpoint runs/run_001/checkpoint_epoch28.pt --config configs/stage1_casia.yaml --benchmark lfw

    # LFW with specific partition subset
    python scripts/evaluate.py --checkpoint runs/run_001/checkpoint_epoch28.pt --config configs/stage1_casia.yaml --benchmark lfw --partitions 0,1,2

    # Classification on a single partition
    python scripts/evaluate.py --checkpoint runs/run_001/checkpoint_epoch50.pt --config configs/stage0_cifar100.yaml --partitions 0
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
        description="PPI Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--variant", default=None, help="Variant config overlay path")
    parser.add_argument(
        "--partitions", default=None,
        help="Comma-separated partition indices (e.g. '0,1,2'). "
             "If omitted, evaluates all 7 non-degenerate configs.",
    )
    parser.add_argument(
        "--benchmark", default=None, choices=["lfw"],
        help="Run a pair-verification benchmark instead of classification eval",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force a specific device (e.g. 'cpu', 'cuda', 'cuda:1'). "
             "Defaults to cuda if available. Use 'cpu' to run eval without "
             "contending with a training run for the GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Shorthand for --device cpu.",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else args.device

    from ppi.utils.config import load_full_config

    config = load_full_config(args.config, variant_path=args.variant)

    partition_configs = None
    if args.partitions:
        partition_configs = [set(int(x) for x in args.partitions.split(","))]

    from ppi.evaluation.evaluator import Evaluator

    evaluator = Evaluator(config, args.checkpoint, device=device)

    if args.benchmark == "lfw":
        results = evaluator.evaluate_lfw(partition_configs=partition_configs)
        print("\n=== LFW Pair Verification ===")
        for cfg_name, metrics in results.items():
            acc = metrics["pair_accuracy"]
            std = metrics["pair_std"]
            tar = metrics["tar_at_far_1e-3"]
            print(f"  {cfg_name}:  accuracy={acc:.4f} +/- {std:.4f}  TAR@FAR=1e-3={tar:.4f}")
    else:
        results = evaluator.evaluate(partition_configs=partition_configs)
        print("\n=== Classification Evaluation ===")
        for cfg_name, metrics in results.items():
            print(f"\n{cfg_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
