"""Direction 2 boosting training script.

Execution flow:
  1. Load config, apply CLI overrides.
  2. Instantiate BoostingTrainer.
  3. trainer.train() — phases 0 through N-1.
  4. BoostingEvaluator.evaluate_all() — full eval grid.
  5. Print results table; log to wandb.

Example (CIFAR-100 smoke test):
  python scripts/train_boosting.py \\
      --config configs/direction_2_base.yaml \\
      --dataset cifar100 \\
      --backbone-state partial --frozen-stages 3 \\
      --mining-strategy topk --mining-topk 0.1 \\
      --loss triplet --combination cosine_concat \\
      --epochs-phase0 20 --epochs-per-phase 20

Example (CASIA subset gate run):
  python scripts/train_boosting.py \\
      --config configs/direction_2_base.yaml \\
      --dataset casia_subset \\
      --backbone-state partial --frozen-stages 3 \\
      --mining-strategy topk \\
      --loss triplet --combination cosine_concat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ppi.boosting.trainer import BoostingTrainer
from ppi.evaluation.boosting_eval import BoostingEvaluator, print_results_table
from ppi.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Direction 2: Boosting reformulation training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to YAML config")

    # Backbone
    p.add_argument("--backbone-state", choices=["frozen", "fine_tuned", "partial"])
    p.add_argument("--backbone-lr-multiplier", type=float)
    p.add_argument("--frozen-stages", type=int)

    # Mining
    p.add_argument("--mining-strategy", choices=["band", "topk"])
    p.add_argument("--mining-band-low", type=float)
    p.add_argument("--mining-band-high", type=float)
    p.add_argument("--mining-topk", type=float, dest="mining_topk_fraction")
    p.add_argument("--mining-refresh-every", type=int)

    # Loss
    p.add_argument(
        "--loss",
        choices=["arcface_reweighted", "arcface_margin", "triplet", "contrastive", "sub_center_arcface"],
    )
    p.add_argument("--easy-loss-weight", type=float)
    p.add_argument("--triplet-margin", type=float)
    p.add_argument("--triplet-mining", choices=["batch_hard", "semi_hard"])
    p.add_argument("--contrastive-margin", type=float)
    p.add_argument("--sub-center-K", type=int)

    # Combination
    p.add_argument("--combination", choices=["cosine_concat", "confidence_weighted", "learned_combiner"])
    p.add_argument("--confidence-source", choices=["embedding_norm", "cosine_magnitude", "scalar_head"])
    p.add_argument("--d1-combiner-path", type=str, help="Path to Direction 1 combiner checkpoint")

    # Dataset and partitions
    p.add_argument("--dataset", choices=["cifar100", "casia_subset", "casia"])
    p.add_argument("--num-partitions", type=int)
    p.add_argument("--K", type=int, dest="partition_K",
                   help="Per-partition embedding dim (overrides partitions.K)")

    # Training
    p.add_argument("--epochs-phase0", type=int)
    p.add_argument("--epochs-per-phase", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--wandb-project", type=str)
    p.add_argument("--run-name", type=str)

    # Evaluation
    p.add_argument("--baseline-results-dir", type=str, help="Dir with Variant A–D result JSONs")
    p.add_argument("--eval-only", type=str, default=None, help="Path to phase checkpoint dir; skip training")
    p.add_argument("--eval-all-subsets", action="store_true",
                   help="Also score non-P0-anchored subsets (P1, P2, P12). Diagnostic: "
                        "isolates per-partition cosine metric quality.")
    p.add_argument("--cifar-phase0-labels", choices=["superclass", "subclass"],
                   default="superclass",
                   help="Label space for CIFAR-100 phase 0. 'subclass' trains P0 "
                        "directly on the 100-way eval task — use with "
                        "--num-partitions 1 to measure the task ceiling.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply non-None CLI args as overrides into the config dict."""
    boosting = config.setdefault("boosting", {})
    training = config.setdefault("training", {})
    data = config.setdefault("data", {})
    logging_cfg = config.setdefault("logging", {})

    _set_if(boosting, "backbone_state", args.backbone_state)
    _set_if(boosting, "backbone_lr_multiplier", args.backbone_lr_multiplier)
    _set_if(boosting, "frozen_stages", args.frozen_stages)
    _set_if(boosting, "mining_strategy", args.mining_strategy)
    _set_if(boosting, "mining_band_low", args.mining_band_low)
    _set_if(boosting, "mining_band_high", args.mining_band_high)
    _set_if(boosting, "mining_topk_fraction", args.mining_topk_fraction)
    _set_if(boosting, "mining_refresh_every", args.mining_refresh_every)
    _set_if(boosting, "loss", args.loss)
    _set_if(boosting, "easy_loss_weight", args.easy_loss_weight)
    _set_if(boosting, "triplet_margin", args.triplet_margin)
    _set_if(boosting, "triplet_mining", args.triplet_mining)
    _set_if(boosting, "contrastive_margin", args.contrastive_margin)
    _set_if(boosting, "sub_center_K", args.sub_center_K)
    _set_if(boosting, "combination", args.combination)
    _set_if(boosting, "confidence_source", args.confidence_source)
    _set_if(boosting, "d1_combiner_path", args.d1_combiner_path)
    _set_if(training, "epochs_phase0", args.epochs_phase0)
    _set_if(training, "epochs_per_phase", args.epochs_per_phase)
    _set_if(data, "dataset", args.dataset)
    _set_if(logging_cfg, "wandb_project", args.wandb_project)
    _set_if(logging_cfg, "run_name", args.run_name)

    # CIFAR-100 images are 32×32; override any face-dataset input_size from the config.
    if data.get("dataset") == "cifar100":
        data["input_size"] = 32

    if args.num_partitions is not None:
        config["num_partitions"] = args.num_partitions
        config.setdefault("partitions", {})["num_partitions"] = args.num_partitions
    if args.partition_K is not None:
        config.setdefault("partitions", {})["K"] = args.partition_K
    if args.eval_all_subsets:
        config.setdefault("evaluation", {})["all_subsets"] = True
    if args.seed is not None:
        config["seed"] = args.seed

    return config


def _set_if(d: dict, key: str, value) -> None:
    if value is not None:
        d[key] = value


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def _print_flat_results(results: dict, title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"{'Key':<40}  {'pair_acc':>10}  {'pair_std':>9}  {'TAR@1e-3':>10}"
    print(header)
    print("-" * 74)
    for key in sorted(results.keys()):
        m = results[key]
        acc = m.get("pair_accuracy", float("nan"))
        std = m.get("pair_std", float("nan"))
        tar = m.get("tar_at_far_1e-3", float("nan"))
        print(f"{key:<40}  {acc:>10.4f}  {std:>9.4f}  {tar:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    config = _apply_cli_overrides(config, args)

    seed: int = config.get("seed", 42)
    torch.manual_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_boosting] device={device}", flush=True)

    # Logger
    run_name = (
        config.get("logging", {}).get("run_name")
        or f"d2_{config.get('data', {}).get('dataset', 'casia')}"
        f"_{config.get('boosting', {}).get('backbone_state', 'partial')}"
        f"_{config.get('boosting', {}).get('loss', 'triplet')}"
    )
    config.setdefault("logging", {})["run_name"] = run_name
    config["variant"] = run_name
    logger = ExperimentLogger(config)
    print(f"[train_boosting] Run dir: {logger.run_dir}", flush=True)

    # Log full config to wandb
    if logger._wandb_run is not None:
        import wandb
        wandb.config.update(config, allow_val_change=True)

    num_partitions: int = config.get("num_partitions", config.get("partitions", {}).get("num_partitions", 3))
    config["num_partitions"] = num_partitions
    config.setdefault("partitions", {})["num_partitions"] = num_partitions

    dataset_name: str = config.get("data", {}).get("dataset", "casia")

    # ---------------------------------------------------------------------------
    # CIFAR-100 special path: use the boosting adaptor
    # ---------------------------------------------------------------------------
    if dataset_name == "cifar100":
        from ppi.boosting.cifar100_adaptor import CIFAR100BoostingAdaptor

        adaptor = CIFAR100BoostingAdaptor(
            root=config["data"]["root"],
            input_size=config["data"].get("input_size", 32),
        )
        # Phase 0 normally trains on the 20 superclasses, which deliberately
        # handicaps it relative to the 100-way subclass eval so the boosting
        # phases have headroom. Training it directly on subclass instead makes
        # it a ceiling measurement for the task itself.
        p0_subclass = args.cifar_phase0_labels == "subclass"
        p0_phase = 1 if p0_subclass else 0
        p0_num_classes = (
            adaptor.num_classes_phase1plus if p0_subclass else adaptor.num_classes_phase0
        )
        config.setdefault("arcface", {})["num_classes"] = p0_num_classes
        if p0_subclass:
            print(
                f"[train_boosting] CEILING MODE: phase 0 trains on "
                f"{p0_num_classes}-way subclass labels (the eval task itself).",
                flush=True,
            )

        trainer = BoostingTrainer(config=config, device=device, logger=logger)

        if args.eval_only is None:
            epochs_phase0 = config["training"].get("epochs_phase0", 20)
            epochs_per_phase = config["training"].get("epochs_per_phase", 20)
            trainer._train_dataset = adaptor.get_train_dataset(phase=p0_phase)
            trainer.num_classes = p0_num_classes
            print(f"[BoostingTrainer] Starting Phase 0 ({epochs_phase0} epochs)", flush=True)
            trainer._train_phase_0(epochs_phase0)
            trainer._save_phase_checkpoint(0)
            # Phases 1+ use subclass dataset (100 classes) — rebuild the
            # arcface head with the larger label space before training.
            for phase_k in range(1, num_partitions):
                trainer._train_dataset = adaptor.get_train_dataset(phase=phase_k)
                trainer.rebuild_head_for_num_classes(phase_k, adaptor.num_classes_phase1plus)
                print(
                    f"\n[BoostingTrainer] Starting Phase {phase_k} ({epochs_per_phase} epochs)",
                    flush=True,
                )
                trainer._train_phase_k(phase_k, epochs_per_phase)
                trainer._save_phase_checkpoint(phase_k)
        else:
            # Without this the eval below would score a randomly-initialised
            # backbone (the CASIA branch loads, this one previously did not).
            _load_phase_checkpoints(trainer, args.eval_only, num_partitions, device)

        # CIFAR-100 verification eval
        print("\n[train_boosting] CIFAR-100 verification evaluation...", flush=True)
        imgs_a, imgs_b, is_same = adaptor.get_val_pairs()
        _eval_cifar100_verification(trainer, imgs_a, imgs_b, is_same, device)

    # ---------------------------------------------------------------------------
    # Standard CASIA / CASIA-subset path
    # ---------------------------------------------------------------------------
    else:
        trainer = BoostingTrainer(config=config, device=device, logger=logger)

        if args.eval_only is None:
            trainer.train()
        else:
            _load_phase_checkpoints(trainer, args.eval_only, num_partitions, device)

        # LFW evaluation
        lfw_cfg = config.get("evaluation", {}).get("lfw", {})
        if lfw_cfg.get("root") and Path(lfw_cfg["root"]).exists():
            print("\n[train_boosting] Running LFW evaluation...", flush=True)
            combination_strategies = _resolve_strategies(config)
            evaluator = BoostingEvaluator(
                backbone=trainer.backbone,
                partition_heads=list(trainer.partition_arcface_heads),
                combination_strategies=combination_strategies,
                config=config,
                device=device,
                d1_combiner_path=config.get("boosting", {}).get("d1_combiner_path"),
                baseline_results_dir=args.baseline_results_dir,
            )
            lfw_results = evaluator.evaluate_lfw()
            _print_flat_results(lfw_results, f"LFW — {run_name}")

            if logger._wandb_run is not None:
                import wandb
                wandb.log({f"lfw/{k}": v.get("pair_accuracy", 0.0) for k, v in lfw_results.items()})

    logger.close()
    print("[train_boosting] Done.", flush=True)


def _resolve_strategies(config: dict) -> list[str]:
    """Determine which combination strategies to evaluate."""
    primary = config.get("boosting", {}).get("combination", "cosine_concat")
    strategies = [primary]
    if primary != "cosine_concat":
        strategies.insert(0, "cosine_concat")
    if config.get("boosting", {}).get("d1_combiner_path") and "learned_combiner" not in strategies:
        strategies.append("learned_combiner")
    return strategies


def _load_phase_checkpoints(
    trainer: BoostingTrainer,
    checkpoint_dir: str,
    num_partitions: int,
    device: torch.device,
) -> None:
    """Load backbone + all partition heads from a boosting checkpoint directory."""
    ckpt_root = Path(checkpoint_dir)

    # Prefer the latest phase's backbone, but fall back to earlier phases so a
    # partial run still evaluates real weights instead of silently scoring a
    # randomly-initialised backbone.
    backbone_path = None
    for phase in range(num_partitions - 1, -1, -1):
        candidate = ckpt_root / f"phase_{phase}" / "backbone.pt"
        if candidate.exists():
            backbone_path = candidate
            break

    if backbone_path is None:
        raise FileNotFoundError(
            f"No backbone.pt found under {ckpt_root}/phase_*/. Refusing to evaluate "
            f"an untrained backbone."
        )

    ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
    # Accept both flat and D1-style nested formats
    backbone_sd = (
        ckpt["model_state_dict"]["backbone"] if "model_state_dict" in ckpt
        else ckpt["backbone"]
    )
    trainer.backbone.load_state_dict(backbone_sd)
    print(f"[train_boosting] Loaded backbone from {backbone_path}", flush=True)

    for k in range(num_partitions):
        head_path = ckpt_root / f"phase_{k}" / f"partition_{k}.pt"
        if not head_path.exists():
            continue
        ckpt = torch.load(head_path, map_location=device, weights_only=False)
        head_sd = ckpt["partition_head"]
        # Phase 1+ heads may have been rebuilt for a larger label space during
        # training (CIFAR-100: 20 superclasses at phase 0, 100 subclasses
        # after), so the constructed head can be the wrong shape here.
        ckpt_num_classes = head_sd["weight"].shape[0]
        if ckpt_num_classes != trainer.partition_arcface_heads[k].weight.shape[0]:
            print(
                f"[train_boosting] Rebuilding arcface head {k} for "
                f"{ckpt_num_classes} classes to match checkpoint",
                flush=True,
            )
            trainer.rebuild_head_for_num_classes(k, ckpt_num_classes)
        trainer.partition_arcface_heads[k].load_state_dict(head_sd)
        print(f"[train_boosting] Loaded partition head {k} from {head_path}", flush=True)


@torch.no_grad()
def _eval_cifar100_verification(
    trainer: BoostingTrainer,
    imgs_a: "torch.Tensor",
    imgs_b: "torch.Tensor",
    is_same: "torch.Tensor",
    device: torch.device,
) -> None:
    """Run same-subclass verification on CIFAR-100 val pairs.

    Honours boosting.combination — previously this path hardcoded CosineConcat,
    so --combination was silently ignored on the CIFAR-100 branch.
    """
    import numpy as np
    from ppi.evaluation.metrics import compute_pair_accuracy, compute_tar_at_far
    from ppi.boosting.combination import ConfidenceWeighted, LearnedCombiner, get_combiner

    trainer.backbone.eval()
    num_partitions = trainer.num_partitions
    cfg = trainer.config
    strategy = cfg.get("boosting", {}).get("combination", "cosine_concat")
    confidence_source = cfg.get("boosting", {}).get("confidence_source", "embedding_norm")

    # Combination runs on CPU: the partition embeddings are cached to CPU
    # above and the metrics below are numpy, so this avoids a device split.
    combine_device = torch.device("cpu")
    combiner = get_combiner(
        strategy,
        confidence_source=confidence_source,
        num_partitions=num_partitions,
        partition_dim=trainer.K,
        d1_combiner_path=cfg.get("boosting", {}).get("d1_combiner_path"),
        device=combine_device,
    )
    if isinstance(combiner, ConfidenceWeighted):
        combiner = combiner.to(combine_device).eval()
        if confidence_source == "scalar_head":
            print(
                "  [eval] WARNING: confidence_source='scalar_head' uses randomly "
                "initialised Linear heads — no training path updates them, so the "
                "confidence signal is noise. Use 'embedding_norm' instead.",
                flush=True,
            )
    print(f"  [eval] combination strategy: {strategy}"
          + (f" (confidence_source={confidence_source})"
             if strategy == "confidence_weighted" else ""), flush=True)

    batch_size = 256
    norm_a, norm_b, raw_a_l, raw_b_l, feat_a_l, feat_b_l = [], [], [], [], [], []

    for imgs, nbucket, rbucket, fbucket in [
        (imgs_a, norm_a, raw_a_l, feat_a_l), (imgs_b, norm_b, raw_b_l, feat_b_l)
    ]:
        for start in range(0, imgs.shape[0], batch_size):
            batch = imgs[start: start + batch_size].to(device)
            out = trainer.backbone(batch)
            nbucket.append(torch.stack([p.cpu() for p in out["partitions"]], dim=1))
            # Pre-normalisation outputs drive embedding_norm confidence.
            rbucket.append(torch.stack([p.cpu() for p in out["partitions_raw"]], dim=1))
            # Shared trunk feature — the representation every partition reads from.
            fbucket.append(out["features"].float().cpu())

    emb_norm_a = torch.cat(norm_a, dim=0)      # (N, P, K) unit vectors
    emb_norm_b = torch.cat(norm_b, dim=0)
    emb_raw_a = torch.cat(raw_a_l, dim=0)      # (N, P, K) pre-normalisation
    emb_raw_b = torch.cat(raw_b_l, dim=0)
    feats_a = torch.cat(feat_a_l, dim=0)       # (N, D) trunk
    feats_b = torch.cat(feat_b_l, dim=0)
    issame_np = is_same.numpy().astype(bool)

    anchored = not cfg.get("evaluation", {}).get("all_subsets", False)

    # --- Collapse check -------------------------------------------------
    # A partition trained into a degenerate solution maps every input to the
    # same direction. Mean pairwise cosine ~1.0 means collapsed: every pair
    # scores identically, so verification lands at chance no matter what the
    # combination strategy does. Catch it here rather than inferring it from
    # a suspicious accuracy.
    # The diagnostic is the SPREAD of pairwise cosines, not their mean. A
    # concentrated-but-healthy embedding can sit at high mean cosine and still
    # separate pairs perfectly well; a collapsed one gives every pair the same
    # score, leaving the threshold search nothing to work with.
    print("\n  Per-partition embedding spread (collapse check):")
    print(f"  {'Partition':<10}  {'mean cos':>10}  {'std cos':>10}  {'status':>12}")
    print("  " + "-" * 48)
    n_probe = min(512, emb_norm_a.shape[0])
    eye = torch.eye(n_probe, dtype=torch.bool)
    collapsed_partitions = []
    for i in range(num_partitions):
        e = emb_norm_a[:n_probe, i, :].float()
        off = (e @ e.T)[~eye]
        mean_cos, std_cos = off.mean().item(), off.std().item()
        if std_cos < 0.01:
            status, = ("COLLAPSED",)
            collapsed_partitions.append(i)
        elif std_cos < 0.05:
            status = "low spread"
        else:
            status = "ok"
        print(f"  {'P' + str(i):<10}  {mean_cos:>10.4f}  {std_cos:>10.4f}  {status:>12}")
    if collapsed_partitions:
        print(f"\n  WARNING: P{collapsed_partitions} collapsed to a point. Their rows below")
        print("           are meaningless and any 'improvement' from adding them is noise.")

    print("\n  CIFAR-100 Verification (same-subclass):")
    if not anchored:
        print("  (non-P0-anchored rows are diagnostic only, not deployable configs)")
    print(f"  {'Config':<10}  {'pair_acc':>10}  {'pair_std':>10}  {'TAR@1e-3':>10}")
    print("  " + "-" * 48)

    def _assemble(normed, raw, active: set[int]):
        parts = [
            normed[:, i, :].float() if i in active else None
            for i in range(num_partitions)
        ]
        raws = [
            raw[:, i, :].float() if i in active else None
            for i in range(num_partitions)
        ]
        if isinstance(combiner, ConfidenceWeighted):
            return combiner.combine(parts, raw_embeddings=raws)
        if isinstance(combiner, LearnedCombiner):
            mask = torch.zeros(normed.shape[0], num_partitions, device=combine_device)
            for i in active:
                mask[:, i] = 1.0
            return combiner.combine(parts, mask=mask)
        return combiner.combine(parts)

    for active_set_size in range(1, num_partitions + 1):
        for combo in _all_subset_combos(num_partitions, active_set_size, anchored):
            config_name = "P" + "".join(str(i) for i in combo)
            active = set(combo)
            ca = _assemble(emb_norm_a, emb_raw_a, active)
            cb = _assemble(emb_norm_b, emb_raw_b, active)
            emb_a = torch.nn.functional.normalize(ca.float().cpu(), dim=1, eps=1e-12).numpy()
            emb_b = torch.nn.functional.normalize(cb.float().cpu(), dim=1, eps=1e-12).numpy()
            mean_acc, std_acc = compute_pair_accuracy(emb_a, emb_b, issame_np)
            sims = (emb_a * emb_b).sum(axis=1)
            tar = compute_tar_at_far(sims[issame_np], sims[~issame_np], far_target=1e-3)
            print(f"  {config_name:<10}  {mean_acc:>10.4f}  {std_acc:>10.4f}  {tar:>10.4f}")
            if config_name == "P0":
                p0_acc, p0_std = mean_acc, std_acc

    if not anchored:
        # Headroom reference: the shared trunk under the same cosine metric.
        # Every partition is a projection of this vector, so it bounds what any
        # partitioning scheme can reach. Not a strict upper bound — raw pooled
        # features were never trained for cosine verification and a learned
        # projection can beat them — but if it sits at P0, this task has no room
        # for progressive improvement and cannot gate anything.
        ta = torch.nn.functional.normalize(feats_a, dim=1, eps=1e-12).numpy()
        tb_ = torch.nn.functional.normalize(feats_b, dim=1, eps=1e-12).numpy()
        trunk_acc, trunk_std = compute_pair_accuracy(ta, tb_, issame_np)
        tsims = (ta * tb_).sum(axis=1)
        trunk_tar = compute_tar_at_far(tsims[issame_np], tsims[~issame_np], far_target=1e-3)
        print("  " + "-" * 48)
        print(f"  {'TRUNK':<10}  {trunk_acc:>10.4f}  {trunk_std:>10.4f}  {trunk_tar:>10.4f}   "
              f"(dim={feats_a.shape[1]}, reference)")
        print(f"\n  Headroom above P0 on this task: {trunk_acc - p0_acc:+.4f}")

        # --- Complementarity / oracle ceiling -------------------------------
        # Progressive improvement requires the partitions to fail on DIFFERENT
        # pairs. If they fail on the same ones, averaging returns their mean
        # and no combination scheme — cosine, weighted, or learned — can beat
        # the best single partition. The oracle below is the union bound: a
        # per-pair selector that always picks a partition that gets it right.
        # It is optimistic (thresholds fitted in-sample) and therefore a true
        # UPPER bound on any combiner.
        def _correct_mask(i: int) -> np.ndarray:
            ea = torch.nn.functional.normalize(emb_norm_a[:, i, :].float(), dim=1).numpy()
            eb = torch.nn.functional.normalize(emb_norm_b[:, i, :].float(), dim=1).numpy()
            s = (ea * eb).sum(axis=1)
            best, best_thr = -1.0, 0.0
            for thr in np.linspace(s.min(), s.max(), 1000):
                acc = ((s >= thr) == issame_np).mean()
                if acc > best:
                    best, best_thr = acc, thr
            return (s >= best_thr) == issame_np

        masks = [_correct_mask(i) for i in range(num_partitions)]
        print("\n  Complementarity (do partitions fail on different pairs?):")
        print(f"  {'':<24}{'acc':>8}")
        for i, m in enumerate(masks):
            print(f"  {'P' + str(i) + ' alone':<24}{m.mean():>8.4f}")

        union_all = np.zeros_like(masks[0])
        for m in masks:
            union_all |= m
        print(f"  {'ORACLE (any partition)':<24}{union_all.mean():>8.4f}   <- ceiling "
              f"for ANY combiner")

        base = masks[0]
        for i in range(1, num_partitions):
            p0_wrong = ~base
            rescued = (p0_wrong & masks[i]).sum() / max(p0_wrong.sum(), 1)
            # Phi coefficient between the two correctness vectors.
            corr = np.corrcoef(base.astype(float), masks[i].astype(float))[0, 1]
            print(f"    P{i}: fixes {rescued * 100:5.1f}% of P0's errors | "
                  f"error correlation with P0 = {corr:+.3f}")

        # --- Specialisation profile ------------------------------------------
        # Under the boosting framing P_k is NOT expected to be good globally —
        # it is trained on P0's failure distribution, so it should look weak on
        # the bulk of pairs and strong where P0 is uncertain. Stratify by P0's
        # decision margin to see whether that specialisation actually happened.
        ea0 = torch.nn.functional.normalize(emb_norm_a[:, 0, :].float(), dim=1).numpy()
        eb0 = torch.nn.functional.normalize(emb_norm_b[:, 0, :].float(), dim=1).numpy()
        s0 = (ea0 * eb0).sum(axis=1)
        best, thr0 = -1.0, 0.0
        for thr in np.linspace(s0.min(), s0.max(), 1000):
            a = ((s0 >= thr) == issame_np).mean()
            if a > best:
                best, thr0 = a, thr
        margin0 = np.abs(s0 - thr0)
        qs = np.quantile(margin0, [0.25, 0.5, 0.75])
        strata = [
            ("Q1 (P0 least sure)", margin0 <= qs[0]),
            ("Q2", (margin0 > qs[0]) & (margin0 <= qs[1])),
            ("Q3", (margin0 > qs[1]) & (margin0 <= qs[2])),
            ("Q4 (P0 most sure)", margin0 > qs[2]),
        ]
        print("\n  Specialisation profile (accuracy by P0 decision margin):")
        hdr = f"  {'stratum':<20}" + "".join(f"{'P'+str(i):>9}" for i in range(num_partitions))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for label, sel in strata:
            row = f"  {label:<20}"
            for i in range(num_partitions):
                row += f"{masks[i][sel].mean():>9.4f}"
            print(row)
        print("    ^ complementarity looks like P_k > P0 in Q1 even if P_k < P0 overall.")

        gain = union_all.mean() - masks[0].mean()
        print(f"\n  Oracle gain over P0: {gain:+.4f}  (upper bound, noise-inflated)")

        # --- Realisable score-level combiner ---------------------------------
        # The oracle credits ANY disagreement, including noise: a noisier copy
        # of P0 flips to correct on some boundary pairs by chance. That is not
        # actionable, because at inference nothing tells you which partition to
        # trust. So fit the simplest combiner that could actually be deployed —
        # logistic regression on the per-partition similarity scores, K-folded
        # so it is scored out-of-sample. If this cannot beat P0, the oracle gain
        # is not real structure and no richer combiner will recover it either.
        def _sims(i: int) -> np.ndarray:
            ea = torch.nn.functional.normalize(emb_norm_a[:, i, :].float(), dim=1).numpy()
            eb = torch.nn.functional.normalize(emb_norm_b[:, i, :].float(), dim=1).numpy()
            return (ea * eb).sum(axis=1)

        all_sims = np.stack([_sims(i) for i in range(num_partitions)], axis=1)
        y = issame_np.astype(np.float32)
        n, n_folds = all_sims.shape[0], 10
        fold = n // n_folds
        accs_lr: list[float] = []
        for f in range(n_folds):
            va = np.zeros(n, dtype=bool)
            va[f * fold:(f + 1) * fold] = True
            tr = ~va
            xt = torch.tensor(all_sims[tr]); yt = torch.tensor(y[tr])
            xv = torch.tensor(all_sims[va]); yv = torch.tensor(y[va])
            mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True).clamp_min(1e-6)
            xt, xv = (xt - mu) / sd, (xv - mu) / sd
            lr_model = torch.nn.Linear(num_partitions, 1)
            opt = torch.optim.LBFGS(lr_model.parameters(), max_iter=200)
            def _closure():
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    lr_model(xt).squeeze(1), yt)
                loss.backward()
                return loss
            opt.step(_closure)
            with torch.no_grad():
                pred = (lr_model(xv).squeeze(1) > 0).float()
            accs_lr.append((pred == yv).float().mean().item())

        lr_acc, lr_std = float(np.mean(accs_lr)), float(np.std(accs_lr))
        lr_gain = lr_acc - p0_acc
        se = max(p0_std, lr_std) / (n_folds ** 0.5)
        print(f"\n  Realisable combiner (logistic regression on partition scores,")
        print(f"  10-fold out-of-sample): {lr_acc:.4f} +/- {lr_std:.4f}")
        print(f"  vs P0 {p0_acc:.4f}  ->  {lr_gain:+.4f}  (2*SE = {2 * se:.4f})")

        specialised = any(
            masks[i][strata[0][1]].mean() > masks[0][strata[0][1]].mean()
            for i in range(1, num_partitions)
        )
        if lr_gain > 2 * se:
            print("  -> PROGRESSIVE IMPROVEMENT IS ACHIEVABLE. A deployable combiner")
            print("     beats P0 out-of-sample; cosine averaging was discarding it.")
        elif gain >= 0.01 and not specialised:
            print("  -> Oracle gain is NOT realisable. No partition beats P0 in any")
            print("     stratum, so the union is noise-driven disagreement rather than")
            print("     complementary signal. The fix belongs in training — make the")
            print("     partitions specialise — not in the combiner.")
        else:
            print("  -> No exploitable complementarity at score level. If the Q1 row")
            print("     shows a partition beating P0, an embedding-level combiner may")
            print("     still help; otherwise the partitions are redundant.")
        if abs(trunk_acc - p0_acc) < 0.01:
            print("  -> Trunk is level with P0. This task cannot demonstrate progressive")
            print("     improvement under ANY partitioning scheme; the smoke test is")
            print("     not a valid gate. Redesign the task or move to CASIA subset.")


def _all_subset_combos(
    num_partitions: int,
    size: int,
    anchored: bool = True,
) -> list[tuple[int, ...]]:
    """Partition subsets of the given size.

    anchored=True  -> only subsets containing P0. These are the deployment-valid
                      configurations (P0 is always present on-device).
    anchored=False -> every non-empty subset, including P1-alone and P12. Not
                      deployable, but diagnostic: P_k-alone verification
                      accuracy separates "P_k has poor cosine metric structure"
                      from "the combination step is discarding its signal".
    """
    from itertools import combinations
    subsets = list(combinations(range(num_partitions), size))
    if anchored:
        subsets = [c for c in subsets if 0 in c]
    return subsets


if __name__ == "__main__":
    main()
