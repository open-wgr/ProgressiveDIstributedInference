"""Boosting evaluator for Direction 2 models."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ppi.boosting.combination import CosineConcat, ConfidenceWeighted, LearnedCombiner, get_combiner
from ppi.evaluation.benchmarks import LFWBenchmark, CFPFPBenchmark, AgeDB30Benchmark
from ppi.evaluation.metrics import compute_pair_accuracy, compute_tar_at_far


def _all_partition_configs(num_partitions: int) -> list[set[int]]:
    """All P0-anchored partition subsets (2^{N-1} subsets)."""
    optional = list(range(1, num_partitions))
    configs = []
    for r in range(num_partitions):
        for combo in combinations(optional, r):
            configs.append({0} | set(combo))
    return configs


class BoostingEvaluator:
    """Evaluate Direction 2 models across all subset × combination strategy combinations.

    Produces the full comparison table:
      - All 7 non-empty partition subsets (for N=3)
      - All combination strategies: cosine_concat, confidence_weighted,
        learned_combiner (if --d1-combiner-path provided)
      - LFW, CFP-FP, AgeDB-30 (if configured)
      - Ablation comparison against Variants A–D results (loaded from
        saved result JSON files if --baseline-results-dir is provided)

    Single eval pass: backbone runs once per image; raw partition outputs are
    cached. All subsets × strategies are then evaluated from the cache.
    """

    def __init__(
        self,
        backbone: nn.Module,
        partition_heads: list[nn.Module],
        combination_strategies: list[str],
        config: dict[str, Any],
        device: torch.device,
        d1_combiner_path: str | None = None,
        baseline_results_dir: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.partition_heads = partition_heads
        self.combination_strategies = combination_strategies
        self.config = config
        self.device = device
        self.d1_combiner_path = d1_combiner_path
        self.baseline_results_dir = baseline_results_dir

        self.num_partitions = len(partition_heads)
        self.K: int = config.get("partitions", {}).get("K", 128)

        self.combiners: dict[str, CosineConcat | ConfidenceWeighted | LearnedCombiner] = {}
        for strategy in combination_strategies:
            if strategy == "learned_combiner" and not d1_combiner_path:
                print(
                    "[BoostingEvaluator] Skipping learned_combiner: no --d1-combiner-path provided.",
                    flush=True,
                )
                continue
            self.combiners[strategy] = get_combiner(
                strategy,
                confidence_source=config.get("boosting", {}).get("confidence_source", "embedding_norm"),
                num_partitions=self.num_partitions,
                partition_dim=self.K,
                d1_combiner_path=d1_combiner_path,
                device=device,
            )

    def evaluate_all(self) -> dict[str, dict[str, dict[str, float]]]:
        """Run full eval grid.

        Returns nested dict: subset_name → strategy_name → metric_name → value.
        """
        eval_cfg = self.config.get("evaluation", {})
        results: dict[str, dict[str, dict[str, float]]] = {}

        _BENCHMARKS = [
            ("lfw",    LFWBenchmark,    "lfw"),
            ("cfp_fp", CFPFPBenchmark,  "cfp_fp"),
            ("agedb",  AgeDB30Benchmark, "agedb"),
        ]

        for cfg_key, benchmark_cls, metric_prefix in _BENCHMARKS:
            cfg = eval_cfg.get(cfg_key, {})
            if not cfg.get("root") or not Path(cfg["root"]).exists():
                continue
            print(f"[BoostingEvaluator] Evaluating on {cfg_key}...", flush=True)
            bench_res = self._evaluate_benchmark(benchmark_cls, cfg["root"])
            for subset_strategy, metrics in bench_res.items():
                subset, strategy = subset_strategy.rsplit("__", 1)
                results.setdefault(subset, {}).setdefault(strategy, {}).update(
                    {f"{metric_prefix}_{k}": v for k, v in metrics.items()}
                )

        if self.baseline_results_dir:
            self._attach_baselines(results)

        return results

    def evaluate_lfw(self) -> dict[str, dict[str, float]]:
        """Run LFW verification for all partition subsets × combination strategies.

        Returns dict keyed by "<subset>__<strategy>" → metric dict.
        """
        lfw_cfg = self.config.get("evaluation", {}).get("lfw", {})
        root = lfw_cfg.get("root")
        if root is None:
            raise ValueError("LFW evaluation requires config key evaluation.lfw.root")
        return self._evaluate_benchmark(LFWBenchmark, root)

    def evaluate_cfp_fp(self) -> dict[str, dict[str, float]]:
        """Run CFP-FP verification for all partition subsets × combination strategies.

        Returns dict keyed by "<subset>__<strategy>" → metric dict.
        """
        cfp_cfg = self.config.get("evaluation", {}).get("cfp_fp", {})
        root = cfp_cfg.get("root")
        if root is None:
            raise ValueError("CFP-FP evaluation requires config key evaluation.cfp_fp.root")
        return self._evaluate_benchmark(CFPFPBenchmark, root)

    def evaluate_agedb30(self) -> dict[str, dict[str, float]]:
        """Run AgeDB-30 verification for all partition subsets × combination strategies.

        Returns dict keyed by "<subset>__<strategy>" → metric dict.
        """
        agedb_cfg = self.config.get("evaluation", {}).get("agedb", {})
        root = agedb_cfg.get("root")
        if root is None:
            raise ValueError("AgeDB-30 evaluation requires config key evaluation.agedb.root")
        return self._evaluate_benchmark(AgeDB30Benchmark, root)

    def _evaluate_benchmark(
        self,
        benchmark_cls: type,
        root: str,
    ) -> dict[str, dict[str, float]]:
        """Shared evaluation logic for any PairBenchmark subclass.

        Returns dict keyed by "<subset>__<strategy>" → metric dict.
        """
        benchmark = benchmark_cls(root)
        paths1, paths2, issame = benchmark.load_pairs()

        all_paths = list(set(paths1 + paths2))
        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        raw_partitions = self._extract_raw_partitions(all_paths, root)
        print(
            f"[BoostingEvaluator] Raw partitions: {tuple(raw_partitions.shape)}",
            flush=True,
        )

        partition_configs = _all_partition_configs(self.num_partitions)
        results: dict[str, dict[str, float]] = {}

        for config_set in partition_configs:
            config_name = "P" + "".join(str(i) for i in sorted(config_set))
            print(f"  Evaluating {config_name}...", flush=True)

            partition_embs = self._assemble_partitions(raw_partitions, config_set)

            for strategy_name, combiner in self.combiners.items():
                if isinstance(combiner, LearnedCombiner):
                    mask = self._make_mask(config_set, raw_partitions.shape[0])
                    combined = combiner.combine(partition_embs, mask)
                else:
                    combined = combiner.combine(partition_embs)

                combined_np = combined.cpu().numpy()
                embs1 = np.array([combined_np[path_to_idx[p]] for p in paths1])
                embs2 = np.array([combined_np[path_to_idx[p]] for p in paths2])

                mean_acc, std_acc = compute_pair_accuracy(embs1, embs2, issame)
                similarities = (embs1 * embs2).sum(axis=1)
                genuine = similarities[issame]
                impostor = similarities[~issame]
                tar = compute_tar_at_far(genuine, impostor, far_target=1e-3)

                results[f"{config_name}__{strategy_name}"] = {
                    "pair_accuracy": mean_acc,
                    "pair_std": std_acc,
                    "tar_at_far_1e-3": tar,
                }

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_raw_partitions(
        self,
        image_paths: list[str],
        root: str,
        batch_size: int = 64,
    ) -> Tensor:
        """Run backbone once; return (N, num_partitions, K) on CPU."""
        from pathlib import Path as _Path
        from PIL import Image
        from torchvision import transforms

        root_p = _Path(root)
        input_size = self.config.get("data", {}).get("input_size", 112)
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        self.backbone.eval()
        for head in self.partition_heads:
            head.eval()

        all_parts: list[Tensor] = []
        n_images = len(image_paths)
        for start in range(0, n_images, batch_size):
            batch_paths = image_paths[start: start + batch_size]
            tensors = [transform(Image.open(root_p / p).convert("RGB")) for p in batch_paths]
            images = torch.stack(tensors).to(self.device)

            out = self.backbone(images)
            parts = torch.stack(out["partitions"], dim=1).cpu()  # (B, P, K)
            all_parts.append(parts)

        return torch.cat(all_parts, dim=0)

    def _assemble_partitions(
        self,
        raw_partitions: Tensor,
        active: set[int],
    ) -> list[Tensor | None]:
        """Return list of (N, K) embeddings; None for absent partitions."""
        result: list[Tensor | None] = []
        for i in range(self.num_partitions):
            if i in active:
                emb = raw_partitions[:, i, :]  # (N, K)
                result.append(F.normalize(emb.float(), dim=1))
            else:
                result.append(None)
        return result

    def _make_mask(self, active: set[int], N: int) -> Tensor:
        mask = torch.zeros(N, self.num_partitions, device=self.device)
        for i in active:
            mask[:, i] = 1.0
        return mask

    def _attach_baselines(
        self,
        results: dict[str, dict[str, dict[str, float]]],
    ) -> None:
        """Load and attach baseline (Variant A–D) results from JSON files."""
        baseline_dir = Path(self.baseline_results_dir)
        for json_path in sorted(baseline_dir.glob("*.json")):
            try:
                with open(json_path) as f:
                    baseline = json.load(f)
                variant_name = json_path.stem
                for subset, metrics in baseline.items():
                    results.setdefault(subset, {})[f"baseline_{variant_name}"] = metrics
            except Exception as e:
                print(f"[BoostingEvaluator] Warning: could not load {json_path}: {e}")


def print_results_table(
    results: dict[str, dict[str, dict[str, float]]],
    title: str = "Boosting Evaluation Results",
) -> None:
    """Pretty-print the nested results dict to stdout."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    all_strategies: set[str] = set()
    for subset_results in results.values():
        all_strategies.update(subset_results.keys())
    strategies = sorted(all_strategies)

    header = f"{'Subset':<10}" + "".join(f"  {s[:20]:<22}" for s in strategies)
    print(header)
    print("-" * min(len(header), 90))

    for subset in sorted(results.keys()):
        row = f"{subset:<10}"
        for strategy in strategies:
            metrics = results[subset].get(strategy, {})
            acc = metrics.get("lfw_pair_accuracy", metrics.get("pair_accuracy", float("nan")))
            row += f"  {acc:>8.4f}" + " " * 14
        print(row)
    print()
