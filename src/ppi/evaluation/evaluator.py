"""Evaluator: run all partition configurations against benchmarks."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ppi.backbones import build_backbone
from ppi.heads.arcface import ArcFaceHead
from ppi.training.partition_dropout import assemble_embedding
from ppi.utils.logging import ExperimentLogger


def _all_partition_configs(num_partitions: int = 3) -> list[set[int]]:
    """Return all 7 non-degenerate partition configurations."""
    indices = list(range(num_partitions))
    configs = []
    for r in range(1, num_partitions + 1):
        for combo in combinations(indices, r):
            configs.append(set(combo))
    return configs


class Evaluator:
    """Evaluate a checkpoint at multiple partition configurations."""

    def __init__(self, config: dict[str, Any], checkpoint_path: str) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.backbone = build_backbone(config).to(self.device)

        # Load checkpoint
        ckpt = ExperimentLogger.load_checkpoint(checkpoint_path)
        self.backbone.load_state_dict(ckpt["model_state_dict"]["backbone"])
        self.backbone.eval()

        self.num_partitions = config["partitions"]["num_partitions"]
        self.K = config["partitions"]["K"]

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader,
        active_partitions: set[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings with partition masking.

        Returns (embeddings, labels) as numpy arrays.
        """
        all_embs = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            out = self.backbone(images)
            partition_outputs = out["partitions"]

            # Mask inactive partitions
            masked = []
            for idx, p in enumerate(partition_outputs):
                if idx in active_partitions:
                    masked.append(p)
                else:
                    masked.append(torch.zeros_like(p))

            embedding = assemble_embedding(masked)
            all_embs.append(embedding.cpu().numpy())
            all_labels.append(labels.numpy())

        return np.concatenate(all_embs), np.concatenate(all_labels)

    @torch.no_grad()
    def extract_embeddings_from_paths(
        self,
        image_paths: list[str],
        root: str | Path,
        active_partitions: set[int],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract embeddings for a list of image paths.

        Parameters
        ----------
        image_paths:
            Relative paths within *root*.
        root:
            Root directory containing the images.
        active_partitions:
            Which partition indices are active.
        batch_size:
            Inference batch size.

        Returns
        -------
        np.ndarray of shape (len(image_paths), embedding_dim)
        """
        from PIL import Image

        root = Path(root)
        input_size = self.config.get("data", {}).get("input_size", 112)
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        all_embs = []
        # Process in batches
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors = []
            for p in batch_paths:
                img = Image.open(root / p).convert("RGB")
                tensors.append(transform(img))
            images = torch.stack(tensors).to(self.device)

            out = self.backbone(images)
            partition_outputs = out["partitions"]

            masked = []
            for idx, po in enumerate(partition_outputs):
                if idx in active_partitions:
                    masked.append(po)
                else:
                    masked.append(torch.zeros_like(po))

            embedding = assemble_embedding(masked)
            all_embs.append(embedding.cpu().numpy())

        return np.concatenate(all_embs)

    def evaluate(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run evaluation at specified partition configurations.

        Parameters
        ----------
        partition_configs:
            List of partition index sets to evaluate. Defaults to all 7
            non-degenerate configs.

        Returns
        -------
        dict mapping config name to metric dict.
        """
        if partition_configs is None:
            partition_configs = _all_partition_configs(self.num_partitions)

        from ppi.data import build_dataloader
        val_loader, num_classes = build_dataloader(self.config, split="val")

        results = {}
        for config_set in partition_configs:
            config_name = "P" + "".join(str(i) for i in sorted(config_set))
            embeddings, labels = self.extract_embeddings(val_loader, config_set)

            # For CIFAR-100 / classification tasks: top-1 accuracy
            # Build a simple nearest-centroid classifier
            unique_labels = np.unique(labels)
            centroids = np.zeros((len(unique_labels), embeddings.shape[1]))
            for i, lbl in enumerate(unique_labels):
                centroids[i] = embeddings[labels == lbl].mean(axis=0)

            # Normalise
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
            cen_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
            sim = emb_norm @ cen_norm.T
            preds = unique_labels[sim.argmax(axis=1)]
            accuracy = float((preds == labels).mean())

            results[config_name] = {"accuracy": accuracy}

        return results

    def evaluate_lfw(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run LFW pair verification at each partition configuration.

        Requires ``config["evaluation"]["lfw"]`` with keys ``root`` and
        ``pairs`` pointing to the LFW images and ``pairs.txt`` file.

        Returns
        -------
        dict mapping config name to {pair_accuracy, pair_std, tar_at_far_1e-3}.
        """
        if partition_configs is None:
            partition_configs = _all_partition_configs(self.num_partitions)

        eval_cfg = self.config.get("evaluation", {}).get("lfw", {})
        lfw_root = eval_cfg.get("root")
        if lfw_root is None:
            raise ValueError(
                "LFW evaluation requires config key evaluation.lfw.root"
            )

        from ppi.evaluation.benchmarks import LFWBenchmark
        from ppi.evaluation.metrics import compute_pair_accuracy, compute_tar_at_far

        benchmark = LFWBenchmark(lfw_root)
        paths1, paths2, issame = benchmark.load_pairs()

        # Deduplicate paths for efficient embedding extraction
        all_paths = list(set(paths1 + paths2))
        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        results = {}
        for config_set in partition_configs:
            config_name = "P" + "".join(str(i) for i in sorted(config_set))
            print(f"  Evaluating LFW at {config_name}...", flush=True)

            # Extract embeddings for all unique images
            all_embs = self.extract_embeddings_from_paths(
                all_paths, lfw_root, config_set,
            )

            # Gather pair embeddings
            embs1 = np.array([all_embs[path_to_idx[p]] for p in paths1])
            embs2 = np.array([all_embs[path_to_idx[p]] for p in paths2])

            # Pair accuracy (10-fold)
            mean_acc, std_acc = compute_pair_accuracy(embs1, embs2, issame)

            # TAR@FAR — compute cosine similarities for genuine/impostor split
            e1 = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-12)
            e2 = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-12)
            similarities = (e1 * e2).sum(axis=1)
            genuine = similarities[issame]
            impostor = similarities[~issame]
            tar_1e3 = compute_tar_at_far(genuine, impostor, far_target=1e-3)

            results[config_name] = {
                "pair_accuracy": mean_acc,
                "pair_std": std_acc,
                "tar_at_far_1e-3": tar_1e3,
            }

        return results
