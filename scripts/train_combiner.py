"""Train and evaluate a learned combiner over frozen partition checkpoints.

Execution flow:
  1. Load direction_1.yaml; apply CLI overrides.
  2. Auto-discover backbone config from checkpoint directory.
  3. EmbeddingCache.build() — no-op if cache already exists.
  4. Load embeddings and labels via EmbeddingCache.load().
  5. Construct CachedPartitionDataset (or FullTripleDataset if --full-triple-only).
  6. Instantiate PartitionCombiner with resolved K_out.
  7. train_combiner() — returns trained combiner.
  8. CombinerEvaluator.evaluate_lfw() — all subsets, combiner vs baseline.
  9. If CFP-FP or AgeDB-30 paths are configured, run those too.
  10. evaluate_graceful_degradation() — log inter-subset cosine similarity.
  11. Print and log results tables.

Example usage:
  python scripts/train_combiner.py --config configs/direction_1.yaml \\
      --checkpoint variant_c --k-out K
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Ensure src/ is on the path when run directly
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ppi.combiner.cache import EmbeddingCache
from ppi.combiner.dataset import CachedPartitionDataset, FullTripleDataset
from ppi.combiner.mlp import PartitionCombiner
from ppi.combiner.trainer import train_combiner
from ppi.evaluation.combiner_eval import CombinerEvaluator
from ppi.heads.arcface import ArcFaceHead
from ppi.losses.arcface_loss import ArcFaceLoss
from ppi.utils.config import load_full_config
from ppi.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a learned combiner over frozen partitions.")
    p.add_argument("--config", required=True, help="Path to direction_1.yaml")
    p.add_argument(
        "--checkpoint", required=True,
        choices=["variant_c", "variant_d"],
        help="Which frozen checkpoint to use",
    )
    p.add_argument(
        "--k-out", default="K", choices=["K", "3K"],
        help="Output dimensionality of the combiner (default: K)",
    )
    p.add_argument(
        "--full-triple-only", action="store_true",
        help="Use FullTripleDataset (control) instead of CachedPartitionDataset",
    )
    p.add_argument("--cache-dir", default=None, help="Override cache directory from config")
    p.add_argument("--wandb-project", default=None, help="Override wandb project from config")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Device (default: auto-detect)")
    p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_direction_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_backbone_config(checkpoint_path: str) -> dict:
    """Load the original training config stored alongside the checkpoint.

    Expects a config.yaml in the same directory as the checkpoint file.
    """
    ckpt_dir = Path(checkpoint_path).parent
    config_file = ckpt_dir / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Backbone config not found at {config_file}. "
            "The original training config.yaml must be co-located with the checkpoint."
        )
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    print(f"[train_combiner] Loaded backbone config from {config_file}")
    return cfg


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results_table(
    results: dict[str, dict[str, float]],
    title: str,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"{'Subset':<12}  {'Method':<12}  {'Accuracy':>10}  {'Std':>8}  {'TAR@FAR1e-3':>12}"
    print(header)
    print("-" * 60)

    # Group by subset (strip _combiner / _baseline suffix)
    seen_subsets: list[str] = []
    for key in results:
        subset = key.rsplit("_", 1)[0]
        if subset not in seen_subsets:
            seen_subsets.append(subset)

    for subset in seen_subsets:
        for method in ("combiner", "baseline"):
            key = f"{subset}_{method}"
            if key not in results:
                continue
            m = results[key]
            acc = m.get("pair_accuracy", float("nan"))
            std = m.get("pair_std", float("nan"))
            tar = m.get("tar_at_far_1e-3", float("nan"))
            print(f"{subset:<12}  {method:<12}  {acc:>10.4f}  {std:>8.4f}  {tar:>12.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # --- Config ---
    d1_cfg = _load_direction_config(args.config)

    if args.seed is not None:
        d1_cfg["seed"] = args.seed
    seed: int = d1_cfg.get("seed", 42)

    torch.manual_seed(seed)

    checkpoint_path: str = d1_cfg["checkpoints"][args.checkpoint]
    backbone_cfg = _load_backbone_config(checkpoint_path)

    # CLI overrides
    if args.cache_dir:
        d1_cfg.setdefault("cache", {})["dir"] = args.cache_dir
    if args.wandb_project:
        d1_cfg.setdefault("logging", {})["wandb_project"] = args.wandb_project

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_combiner] device={device}")

    # Resolve architecture dims from backbone config
    num_partitions: int = backbone_cfg["partitions"]["num_partitions"]
    K: int = backbone_cfg["partitions"]["K"]
    output_dim: int = K if args.k_out == "K" else 3 * K
    print(f"[train_combiner] num_partitions={num_partitions}, K={K}, k_out={output_dim}")

    # --- Embedding cache ---
    base_cache_dir = Path(d1_cfg.get("cache", {}).get("dir", "cache/direction_1"))
    variant_cache_dir = base_cache_dir / args.checkpoint
    cache_batch_size: int = d1_cfg.get("cache", {}).get("batch_size", 256)

    cache = EmbeddingCache(
        config=backbone_cfg,
        checkpoint_path=checkpoint_path,
        cache_dir=str(variant_cache_dir),
        batch_size=cache_batch_size,
        device=str(device),
    )
    cache.build()
    embeddings, labels = EmbeddingCache.load(str(variant_cache_dir))
    print(f"[train_combiner] Loaded cache: embeddings={tuple(embeddings.shape)}, labels={tuple(labels.shape)}")

    # --- Dataset ---
    num_classes: int = int(labels.max().item()) + 1
    print(f"[train_combiner] num_classes={num_classes}")

    if args.full_triple_only:
        dataset = FullTripleDataset(embeddings, labels, num_partitions)
        dataset_tag = "full_triple"
    else:
        dataset = CachedPartitionDataset(embeddings, labels, num_partitions, seed=seed)
        dataset_tag = "subset_sampled"
    print(f"[train_combiner] Dataset: {dataset_tag}, {len(dataset)} samples")

    # --- Logger ---
    run_name = (
        d1_cfg.get("logging", {}).get("run_name")
        or f"direction1_{args.checkpoint}_{args.k_out}_{dataset_tag}"
    )
    log_config = {
        "variant": run_name,
        "seed": seed,
        "backbone": backbone_cfg.get("backbone", {}).get("name", "unknown"),
        "partitions": backbone_cfg.get("partitions", {}),
        "arcface": d1_cfg.get("arcface", {}),
        "training": d1_cfg.get("training", {}),
        "data": backbone_cfg.get("data", {}),
        "logging": {
            **d1_cfg.get("logging", {}),
            "output_dir": d1_cfg.get("logging", {}).get("output_dir", "runs/direction_1"),
        },
    }
    logger = ExperimentLogger(log_config)
    print(f"[train_combiner] Run dir: {logger.run_dir}")

    # --- Combiner + ArcFace ---
    combiner_cfg = d1_cfg.get("combiner", {})
    combiner = PartitionCombiner(
        num_partitions=num_partitions,
        partition_dim=K,
        hidden_dim=combiner_cfg.get("hidden_dim", 512),
        output_dim=output_dim,
        dropout=combiner_cfg.get("dropout", 0.1),
    )
    arcface_s: float = d1_cfg.get("arcface", {}).get("s", 64.0)
    arcface_m: float = d1_cfg.get("arcface", {}).get("m", 0.5)
    arcface_head = ArcFaceHead(in_features=output_dim, num_classes=num_classes)
    arcface_loss = ArcFaceLoss(s=arcface_s, m=arcface_m)

    # --- Training ---
    print("[train_combiner] Starting combiner training ...")
    trained_combiner = train_combiner(
        combiner=combiner,
        arcface_head=arcface_head,
        arcface_loss=arcface_loss,
        train_dataset=dataset,
        config=d1_cfg,
        logger=logger,
        device=device,
    )

    # --- Evaluation ---
    # Re-build frozen backbone for evaluation
    from ppi.backbones import build_backbone
    from ppi.utils.logging import ExperimentLogger as EL
    backbone = build_backbone(backbone_cfg).to(device)
    ckpt = EL.load_checkpoint(checkpoint_path)
    backbone.load_state_dict(ckpt["model_state_dict"]["backbone"])
    backbone.eval()

    # Merge eval config: use backbone_cfg's data.input_size + d1_cfg's evaluation paths
    eval_config = {
        **backbone_cfg,
        "evaluation": d1_cfg.get("evaluation", {}),
    }

    evaluator = CombinerEvaluator(
        backbone=backbone,
        combiner=trained_combiner,
        config=eval_config,
        device=device,
    )

    table_title = f"Checkpoint={args.checkpoint}  K_out={args.k_out}  Dataset={dataset_tag}"

    # LFW
    lfw_cfg = d1_cfg.get("evaluation", {}).get("lfw", {})
    if lfw_cfg.get("root") and Path(lfw_cfg["root"]).exists():
        print("\n[train_combiner] Evaluating on LFW ...")
        lfw_results = evaluator.evaluate_lfw()
        _print_results_table(lfw_results, f"LFW — {table_title}")
        logger.log_epoch(
            {f"lfw/{k}/{m}": v
             for k, metrics in lfw_results.items()
             for m, v in metrics.items()},
            epoch=0,
        )
    else:
        print("[train_combiner] Skipping LFW (root not configured or not found)")
        lfw_results = {}

    # CFP-FP (optional)
    cfp_cfg = d1_cfg.get("evaluation", {}).get("cfp_fp", {})
    if cfp_cfg.get("root") and Path(cfp_cfg["root"]).exists():
        print("\n[train_combiner] Evaluating on CFP-FP ...")
        try:
            cfp_results = evaluator.evaluate_cfp_fp()
            _print_results_table(cfp_results, f"CFP-FP — {table_title}")
        except NotImplementedError:
            print("[train_combiner] CFP-FP pair loading not yet implemented, skipping.")

    # AgeDB-30 (optional)
    agedb_cfg = d1_cfg.get("evaluation", {}).get("agedb", {})
    if agedb_cfg.get("root") and Path(agedb_cfg["root"]).exists():
        print("\n[train_combiner] Evaluating on AgeDB-30 ...")
        try:
            agedb_results = evaluator.evaluate_agedb30()
            _print_results_table(agedb_results, f"AgeDB-30 — {table_title}")
        except NotImplementedError:
            print("[train_combiner] AgeDB-30 pair loading not yet implemented, skipping.")

    # Graceful degradation (use LFW images as a convenience probe set if available)
    lfw_root = lfw_cfg.get("root")
    if lfw_root and Path(lfw_root).exists():
        print("\n[train_combiner] Evaluating graceful degradation ...")
        from ppi.evaluation.benchmarks import LFWBenchmark
        benchmark = LFWBenchmark(lfw_root)
        paths1, paths2, _ = benchmark.load_pairs()
        probe_paths = list(set(paths1 + paths2))
        degradation = evaluator.evaluate_graceful_degradation(probe_paths, lfw_root)
        print("\n  Graceful degradation (combine(P012) vs combine(P0) cosine similarity):")
        for k, v in degradation.items():
            print(f"    {k}: {v:.4f}")
        logger.log_epoch({f"degradation/{k}": v for k, v in degradation.items()}, epoch=0)

    logger.close()
    print("\n[train_combiner] Done.")


if __name__ == "__main__":
    main()
