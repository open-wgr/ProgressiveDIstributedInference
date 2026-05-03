"""CombinerEvaluator: evaluate a trained PartitionCombiner vs the cosine baseline."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from ppi.combiner.mlp import PartitionCombiner
from ppi.evaluation.metrics import compute_pair_accuracy, compute_tar_at_far
from ppi.training.partition_dropout import assemble_embedding


def _all_subsets(num_partitions: int) -> list[set[int]]:
    """Return all 2^N - 1 non-empty subsets of {0, ..., num_partitions-1}."""
    result = []
    for r in range(1, num_partitions + 1):
        for combo in combinations(range(num_partitions), r):
            result.append(set(combo))
    return result


class CombinerEvaluator:
    """Evaluate a trained PartitionCombiner on LFW (and optionally CFP-FP, AgeDB-30).

    For each subset, computes both:
      - Combiner output: forward through PartitionCombiner, cosine similarity on output
      - Cosine-on-concatenated baseline: the existing zero-padded + assemble_embedding path

    Both are evaluated at the same subset, giving a direct side-by-side comparison.
    The backbone is run once per image and the raw partition outputs are reused across
    all subset evaluations — same pattern as Evaluator.extract_raw_partitions_from_paths.
    """

    def __init__(
        self,
        backbone: nn.Module,
        combiner: PartitionCombiner,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.backbone = backbone
        self.combiner = combiner
        self.config = config
        self.device = device
        self.num_partitions = config["partitions"]["num_partitions"]
        self.K = config["partitions"]["K"]

        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.combiner.eval()
        self.combiner.requires_grad_(False)

        input_size = config.get("data", {}).get("input_size", 112)
        self._transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    @torch.no_grad()
    def _extract_raw_partitions(
        self,
        image_paths: list[str],
        root: str | Path,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Run the backbone once over all images; return (N, num_partitions, K) on CPU."""
        from PIL import Image

        root = Path(root)
        all_parts: list[torch.Tensor] = []
        n_batches = (len(image_paths) + batch_size - 1) // batch_size

        for b, start in enumerate(range(0, len(image_paths), batch_size)):
            if b % 10 == 0:
                print(f"  [CombinerEvaluator] batch {b}/{n_batches} ...", flush=True)
            batch_paths = image_paths[start:start + batch_size]
            tensors = [
                self._transform(Image.open(root / p).convert("RGB"))
                for p in batch_paths
            ]
            images = torch.stack(tensors).to(self.device)
            out = self.backbone(images)
            parts = torch.stack(out["partitions"], dim=1).cpu()
            all_parts.append(parts)

        return torch.cat(all_parts, dim=0)

    @torch.no_grad()
    def _combiner_embeddings(
        self,
        raw_partitions: torch.Tensor,
        active_partitions: set[int],
        chunk_size: int = 512,
    ) -> np.ndarray:
        """Compute combiner embeddings for a partition subset, processing in chunks."""
        num_p = raw_partitions.shape[1]
        K = raw_partitions.shape[2]
        outs: list[np.ndarray] = []

        for start in range(0, len(raw_partitions), chunk_size):
            chunk = raw_partitions[start:start + chunk_size]   # (B, num_p, K)
            B = chunk.shape[0]

            emb = torch.zeros(B, num_p * K, dtype=torch.float32)
            mask = torch.zeros(B, num_p, dtype=torch.float32)
            for i in active_partitions:
                normed = F.normalize(chunk[:, i, :].float(), dim=1, eps=1e-12)
                emb[:, i * K:(i + 1) * K] = normed
                mask[:, i] = 1.0

            combined = self.combiner(emb.to(self.device), mask.to(self.device))
            outs.append(combined.cpu().numpy())

        return np.concatenate(outs)

    @torch.no_grad()
    def _baseline_embeddings(
        self,
        raw_partitions: torch.Tensor,
        active_partitions: set[int],
        chunk_size: int = 512,
    ) -> np.ndarray:
        """Compute cosine-on-concatenated baseline embeddings for a partition subset.

        Replicates the existing eval exactly: zero-pad inactive partitions, then
        call assemble_embedding (concat + global L2-normalise).
        """
        num_p = raw_partitions.shape[1]
        outs: list[np.ndarray] = []

        for start in range(0, len(raw_partitions), chunk_size):
            chunk = raw_partitions[start:start + chunk_size].to(self.device)
            parts = [
                chunk[:, i, :] if i in active_partitions else torch.zeros_like(chunk[:, i, :])
                for i in range(num_p)
            ]
            emb = assemble_embedding(parts)
            outs.append(emb.cpu().numpy())

        return np.concatenate(outs)

    def _pair_metrics(
        self,
        all_embs: np.ndarray,
        paths1: list[str],
        paths2: list[str],
        issame: np.ndarray,
        path_to_idx: dict[str, int],
    ) -> dict[str, float]:
        embs1 = np.stack([all_embs[path_to_idx[p]] for p in paths1])
        embs2 = np.stack([all_embs[path_to_idx[p]] for p in paths2])
        mean_acc, std_acc = compute_pair_accuracy(embs1, embs2, issame)
        # Embeddings are already L2-normalised
        sims = (embs1 * embs2).sum(axis=1)
        tar = compute_tar_at_far(sims[issame], sims[~issame], far_target=1e-3)
        return {"pair_accuracy": mean_acc, "pair_std": std_acc, "tar_at_far_1e-3": tar}

    def _evaluate_benchmark(
        self,
        benchmark_cls: type,
        benchmark_cfg: dict[str, Any],
        benchmark_name: str,
        partition_configs: list[set[int]] | None,
    ) -> dict[str, dict[str, float]]:
        root = benchmark_cfg.get("root")
        if root is None:
            raise ValueError(f"evaluation.{benchmark_name}.root is required in config")

        benchmark = benchmark_cls(root)
        paths1, paths2, issame = benchmark.load_pairs()

        all_paths = list(set(paths1 + paths2))
        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        print(
            f"[CombinerEvaluator] Extracting raw partitions for {benchmark_name} "
            f"({len(all_paths)} images) ...",
            flush=True,
        )
        raw = self._extract_raw_partitions(all_paths, root)
        print(f"[CombinerEvaluator] Cached: shape={tuple(raw.shape)}", flush=True)

        if partition_configs is None:
            partition_configs = _all_subsets(self.num_partitions)

        results: dict[str, dict[str, float]] = {}
        for pset in partition_configs:
            name = "P" + "".join(str(i) for i in sorted(pset))
            print(f"  Evaluating {name} ...", flush=True)

            combiner_embs = self._combiner_embeddings(raw, pset)
            baseline_embs = self._baseline_embeddings(raw, pset)

            results[f"{name}_combiner"] = self._pair_metrics(
                combiner_embs, paths1, paths2, issame, path_to_idx
            )
            results[f"{name}_baseline"] = self._pair_metrics(
                baseline_embs, paths1, paths2, issame, path_to_idx
            )

        return results

    def evaluate_lfw(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run LFW pair verification for all subsets, both combiner and baseline.

        Returns dict keyed by e.g. "P012_combiner", "P012_baseline", "P0_combiner", etc.
        Each value has keys: pair_accuracy, pair_std, tar_at_far_1e-3.
        """
        from ppi.evaluation.benchmarks import LFWBenchmark
        lfw_cfg = self.config.get("evaluation", {}).get("lfw", {})
        return self._evaluate_benchmark(LFWBenchmark, lfw_cfg, "lfw", partition_configs)

    def evaluate_cfp_fp(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run CFP-FP pair verification for all subsets, both combiner and baseline."""
        from ppi.evaluation.benchmarks import CFPFPBenchmark
        cfp_cfg = self.config.get("evaluation", {}).get("cfp_fp", {})
        return self._evaluate_benchmark(CFPFPBenchmark, cfp_cfg, "cfp_fp", partition_configs)

    def evaluate_agedb30(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run AgeDB-30 pair verification for all subsets, both combiner and baseline."""
        from ppi.evaluation.benchmarks import AgeDB30Benchmark
        agedb_cfg = self.config.get("evaluation", {}).get("agedb", {})
        return self._evaluate_benchmark(AgeDB30Benchmark, agedb_cfg, "agedb", partition_configs)

    @torch.no_grad()
    def evaluate_graceful_degradation(
        self,
        image_paths: list[str],
        root: str,
    ) -> dict[str, float]:
        """Measure cosine similarity between combine({P0,P1,P2}) and combine({P0}).

        High similarity → the combiner preserves identity geometry across subsets,
        supporting gallery indexing with mixed partition availability.

        Returns per-image cosine similarity statistics.
        """
        raw = self._extract_raw_partitions(image_paths, root)
        full_set = set(range(self.num_partitions))
        anchor_set = {0}

        full_embs = torch.from_numpy(self._combiner_embeddings(raw, full_set))
        anchor_embs = torch.from_numpy(self._combiner_embeddings(raw, anchor_set))

        # Both outputs are already L2-normalised; dot product = cosine similarity
        sims = (full_embs * anchor_embs).sum(dim=1).numpy()

        return {
            "mean_cosine": float(sims.mean()),
            "std_cosine": float(sims.std()),
            "min_cosine": float(sims.min()),
            "max_cosine": float(sims.max()),
        }
